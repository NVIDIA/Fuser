import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Tuple
from single_matmul import (
    profile_config,
    estimate_matmul_size,
    analyze_json,
    NvFuserConfig,
    Layout,
)

from nvfuser import (
    FusionCache,
    FusionDefinition,
    SchedulerType,
    MatmulParams,
    ClusterDims,
    MatMulTileOptions,
    GemmTile,
    MmaMacroEncode,
    MmaMacroArch,
    MatmulTileRasterizationOrder,
)

# 139896, 808, 584, NT, 288x128_64x3_1x2_h_coopA_NTN --- HGMMA.64x96x16  -> (3, 2) -> flip cta tile (128, 288)
# 2808, 30128, 4968, TN, 160x256_64x4_2x1_v_coopA_TNN --- HGMMA.64x160x16 -> (4, 1) -> file cta tile (256, 160)
# 11528, 7960, 3784, TT, 192x208_64x4_2x1_v_coopB_NNT --- HGMMA.64x104x16 -> (3, 2) -> no flip
# 272, 8952, 360, TT, 192x144_64x5_4x2_v_coopB_NNT --- HGMMA.64x72x16 -> (3, 2) -> no flip
# 59928, 272, 16936, TN, 272x128_64x4_1x2_h_coopA_TNN --- HGMMA.64x136x16 -> (2, 2) -> flip cta tile (128, 272)

# [288, 272, 304] - MMA-N greater than 256
# 9328, 1408, 53840, TN, 128x272_64x4_1x2_h_coopA_TNT --- HGMMA.64x136x16
# 16224, 1936, 56, TT, 288x128_64x3_1x1_v_coopA_NNN --- HGMMA.64x96x16, HGMMA.64x192x16
# 131824, 912, 16256, TN, 304x128_64x3_1x2_h_coopA_TNN --- HGMMA.64x152x16


class Modifier(Enum):
    coopA = 0
    coopB = 1
    splitK = 2
    alternateM = 3
    alternateN = 4
    MAX = 5


@dataclass
class NvJetConfig:
    cta_tile: Tuple[int, int, int]
    cga: Tuple[int, int]
    stages: int
    rasterization: str
    modifiers: Tuple[Modifier]


def read_lines_from_file(file_path):
    """
    Reads all lines from a file and returns them as a list of strings.
    Each string represents a line, including the newline character if present.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            return lines
    except FileNotFoundError:
        return "File not found."


def parse(file_name):
    lines = read_lines_from_file(file_name)

    weird_coopA = set()
    weird_coopB = set()
    weird_alternate_m = set()
    weird_alternate_n = set()
    data = {layout: set() for layout in Layout if layout is not Layout.MAX}

    splitk = {}
    dp = {}
    for idx, line in enumerate(lines):
        M_str, N_str, K_str, layout, Name = lines[idx].strip().split(", ")
        M = int(M_str)
        N = int(N_str)
        K = int(K_str)
        fields = Name.split("_")

        cta_m, cta_n = [int(a) for a in fields[0].split("x")]
        cta_k, stages = [int(a) for a in fields[1].split("x")]
        cga_m, cga_n = [int(a) for a in fields[2].split("x")]
        rasterization = fields[3]
        gemm_layout = fields[-1]
        modifiers = [Modifier[m] for m in fields[4:-1]]

        # transpose cta tile if output tile is transposed
        if gemm_layout[-1] == "N":
            cta_m, cta_n = cta_n, cta_m
            cga_m, cga_n = cga_n, cga_m

        config = NvJetConfig(
            (cta_m, cta_n, cta_k), (cga_m, cga_n, 1), stages, rasterization, modifiers
        )

        if Modifier.splitK in modifiers:
            splitk[(M, N, K, layout)] = config
            continue
        else:
            dp[(M, N, K, layout)] = config

        assert len(modifiers) <= 1
        # alternate aka ping-pong means separate output tiles per warp-group
        if len(modifiers) == 0:
            if cta_m > 64:
                cond_a = cta_m % 64 == 0
                cond_b = cta_n <= 256
                cond_c = cta_n % 8 == 0
                if all([cond_a, cond_b, cond_c]):
                    modifiers.append(Modifier.alternateM)
                else:
                    weird_alternate_m.add(line)
                    continue
            else:
                cond_a = cta_n > 256
                cond_b = (cta_n / 2) % 8 == 0
                cond_c = (cta_n / 2) <= 256
                # multiple iterations along m dimension per warp-tile
                cond_d = cta_m % 64 == 0
                if all([cond_a, cond_b, cond_c, cond_d]):
                    modifiers.append(Modifier.alternateN)
                elif not cond_c:
                    weird_alternate_n.add(line)
                    continue
        elif modifiers[0] == Modifier.coopA:
            cond_a = (cta_m / 2) % 64 == 0
            # use two iterations along n dimension per warp-tile
            # for cta-n, use two different mma macros - HGMMA.64x96x16, HGMMA.64x192x16
            cond_b = cta_n % 8 == 0
            if not cond_a or not cond_b:
                weird_coopA.add(line)
                continue
        else:
            # multiple iterations along m dimension per warp-tile
            cond_a = cta_m % 64 == 0
            cond_b = (cta_n / 2) <= 256
            cond_c = (cta_n / 2) % 8 == 0
            if not cond_a or not cond_b or not cond_c:
                weird_coopB.add(line)
                continue

    print("==== Summary ====")
    print(f"dp: {len(dp)}, split-k: {len(splitk)}")
    print("total_lines:", len(lines))
    print("weird cooperativeA:", len(weird_coopA))
    print("weird cooperativB:", len(weird_coopB))
    print("weird ping-pong-M:", len(weird_alternate_m))
    print("weird ping-pong-N:", len(weird_alternate_n))
    print("==== End of Summary ====")

    return dp, splitk


def generate_nvfuser_config(config: NvJetConfig):
    # Skip Split-K and Alternate
    if len(config.modifiers) != 1:
        return None

    cta_m, cta_n, cta_k = config.cta_tile

    if config.modifiers[0] == Modifier.coopA:
        if cta_n > 256:
            return None

        warp_m = cta_m // 2
        warp_tile = GemmTile(warp_m, cta_n, cta_k)
        mma_macro = MmaMacroEncode(MmaMacroArch.hopper, 64, cta_n, 16)
    elif config.modifiers[0] == Modifier.coopB:
        warp_n = cta_n // 2
        if warp_n > 256:
            return None
        warp_tile = GemmTile(cta_m, warp_n, cta_k)
        mma_macro = MmaMacroEncode(MmaMacroArch.hopper, 64, warp_n, 16)
    else:
        return None

    cta_tile = GemmTile(cta_m, cta_n, cta_k)

    if config.rasterization == "h":
        tile_order = MatmulTileRasterizationOrder.row_major
    else:
        tile_order = MatmulTileRasterizationOrder.column_major

    # Ignore cluster dim for data parallel
    # cluster_dims = ClusterDims(config.cga[0], config.cga[1], 1)
    cluster_dims = ClusterDims(1, 1, 1)
    return NvFuserConfig(
        MatMulTileOptions(cta_tile, warp_tile),
        mma_macro,
        tile_order,
        cluster_dims,
        config.stages,
    )


def main():
    import sys

    device_properties = torch.cuda.get_device_properties(0)
    dp, splitk = parse(sys.argv[1])

    eager_data = analyze_json("gh200_matmul_eager.json")
    nvf_data = analyze_json("gh200_matmul_nvf.json")

    relative_scores = []
    for idx, (problem_config, nvjet_config) in enumerate(dp.items()):
        nvfuser_config = generate_nvfuser_config(nvjet_config)
        if nvfuser_config is None:
            continue

        # short-circuit: problem does not fit on device
        if (
            estimate_matmul_size(problem_config, torch.bfloat16)
            >= device_properties.total_memory
        ):
            continue

        m, n, k, layout = problem_config
        cta_m, cta_n, cta_k = nvjet_config.cta_tile

        # stmatrix check
        cond_a = m % 16 == 0
        cond_b = n % 16 == 0
        # k-dim check
        cond_c = k % 64 == 0
        # tma check
        cond_d = cta_m <= 256
        cond_e = cta_n <= 256
        cond_f = layout == "NT"

        # Skip all but NT because of CUDA error: cudaErrorMisalignedAddress
        if layout != "NT":
            continue

        FusionCache.reset()
        eager_result, nvf_result, normalized_result = profile_config(
            eager_data, problem_config, nvfuser_config
        )

        if nvf_result == -1:
            continue

        relative_scores.append(normalized_result)

        if problem_config[:3] in nvf_data[Layout[layout]]:
            nvf_baseline = nvf_data[Layout[problem_config[3]]][problem_config[:3]]
            nvf_normalized_result = nvf_baseline / nvf_result
            print(
                f"{idx} / {len(dp)} --- {problem_config} --- "
                f"{eager_result: .3e} out of {nvf_result: 3e} is "
                f"{normalized_result: 2f}. relative improvement: "
                f"{nvf_normalized_result :2f}"
            )

    plt.scatter(
        np.arange(len(relative_scores)),
        np.array(list(sorted(relative_scores))),
        color="blue",
        marker="o",
        s=5,
    )
    plt.ylabel("Relative %")
    plt.ylim(0, 1)
    plt.title("Hopper Matmul S-Curve")
    plt.savefig("nvfuser_NT_nvjet_config.png")
    plt.close("all")


if __name__ == "__main__":
    main()

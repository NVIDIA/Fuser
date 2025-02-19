import sys
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

# 139896, 808, 584, NT, 288x128_64x3_1x2_h_coopA_NTN --- HGMMA.64x96x16  -> (3, 2) -> flip cta tile (128, 288)
# 2808, 30128, 4968, TN, 160x256_64x4_2x1_v_coopA_TNN --- HGMMA.64x160x16 -> (4, 1) -> file cta tile (256, 160)
# 11528, 7960, 3784, TT, 192x208_64x4_2x1_v_coopB_NNT --- HGMMA.64x104x16 -> (3, 2) -> no flip
# 272, 8952, 360, TT, 192x144_64x5_4x2_v_coopB_NNT --- HGMMA.64x72x16 -> (3, 2) -> no flip
# 59928, 272, 16936, TN, 272x128_64x4_1x2_h_coopA_TNN --- HGMMA.64x136x16 -> (2, 2) -> flip cta tile (128, 272)

# [288, 272, 304] - MMA-N greater than 256
# 9328, 1408, 53840, TN, 128x272_64x4_1x2_h_coopA_TNT --- HGMMA.64x136x16
# 16224, 1936, 56, TT, 288x128_64x3_1x1_v_coopA_NNN --- HGMMA.64x96x16, HGMMA.64x192x16
# 131824, 912, 16256, TN, 304x128_64x3_1x2_h_coopA_TNN --- HGMMA.64x152x16


class Layout(Enum):
    NN = 0
    NT = 1
    TN = 2
    TT = 3
    MAX = 4


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


def main():
    import sys

    dp, splitk = parse(sys.argv[1])
    print(len(dp))


if __name__ == "__main__":
    main()

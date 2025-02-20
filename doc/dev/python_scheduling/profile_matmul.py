# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from nvfuser import (
    FusionDefinition,
    SchedulerType,
    MatmulParams,
    ClusterDims,
    MatMulTileOptions,
    GemmTile,
    MmaMacroEncode,
    MmaMacroArch,
    MatmulTileRasterizationOrder,
    MatmulCircularBufferingStrategy,
)
import torch
import math
import itertools
from enum import IntEnum


class Layout(IntEnum):
    NN = 0
    NT = 1
    TN = 2
    TT = 3
    MAX = 4


# machine_info - dict
#  * node, cpu, gpu-name - string
#  * gpu_sm_count - int
# commit_info - dict
#  * id, project, branch - string
# benchmarks - list
#  * fullname: string
#  * params - dict
#     * config: [M, N, K, Shape]
#  * stats - dict
#     * median: float
def analyze_json(filename):
    import pandas as pd
    import json

    def get_field(json_data, field):
        return pd.DataFrame(json_data[field])

    def _organize_by_layout(json_data):
        benchmarks = get_field(json_data, "benchmarks")
        data = {layout: {} for layout in Layout if layout is not Layout.MAX}
        for row in benchmarks.itertuples():
            M, N, K, layout = row.params["config"]
            shape = (M, N, K)
            time = row.stats["median"]
            data[Layout[layout]][shape] = time
        return data

    json = json.load(open(filename))
    return _organize_by_layout(json)


def estimate_matmul_size(config, dtype):
    def _estimate_size(shape, dtype):
        return math.prod(shape) * dtype.itemsize

    m, n, k, layout = config
    total_in_gbs = 0
    for shape in [[m, k], [n, k], [m, n]]:
        total_in_gbs += _estimate_size(shape, dtype)
    return total_in_gbs


def matmul_fusion(fd: FusionDefinition, inputs: list[torch.Tensor]) -> None:
    a = fd.from_pytorch(inputs[0])
    b = fd.from_pytorch(inputs[1])
    out = fd.ops.matmul(a, b)
    fd.add_output(out)


# These are the parameters we'll optimize
parameter_configurations = {
    "tile_sizes": [MatMulTileOptions(GemmTile(128, 256, 64), GemmTile(64, 256, 64))],
    "mma_macro": [MmaMacroEncode(MmaMacroArch.hopper, 64, 256, 16)],
    "tile_order": [MatmulTileRasterizationOrder.column_major],
    "cluster_dims": [ClusterDims(1, 1, 1)],
    "circular_buffer_stages": [4],
}


# Apply scheduler with custom parameters using decorator
def custom_matmul_scheduler(fd, config, verbose=False):
    def inner_fn():
        assert config is not None
        status, error = fd.sched.can_schedule(SchedulerType.matmul)
        assert status, error

        schedule_params = fd.sched.compute_matmul_heuristics()

        # Modify original parameters
        tile_sizes, macro, tile_order, cluster_dims, stages = config
        schedule_params.tile_sizes = tile_sizes
        schedule_params.mma_macro = macro.mma_macro()
        schedule_params.cta_order = tile_order
        schedule_params.cluster_dims = cluster_dims
        schedule_params.circular_buffering_strategy = (
            MatmulCircularBufferingStrategy.warp_specialized
        )
        schedule_params.circular_buffer_options.circular_buffer_smem_write = stages > 1
        schedule_params.circular_buffer_options.smem_circular_buffer_stage = stages
        if verbose:
            print(schedule_params)

        # Schedule fusion
        fd.sched.schedule()

    if config is not None:
        fd.schedule = inner_fn
    return fd


def test_matmul_nvf(
    problem_config: tuple,
    schedule_config: tuple,
    verbose: bool = False,
    validate: bool = False,
):
    m, n, k, layout = problem_config
    dtype = torch.bfloat16

    # NOTE reduced precision accumulation is not supported in nvFuser
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    if layout == "NT" or layout == "NN":
        a = a.as_strided(size=[m, k], stride=[1, m])
    if layout == "TN" or layout == "NN":
        b = b.as_strided(size=[k, n], stride=[1, k])

    with FusionDefinition() as presched_fd:
        matmul_fusion(presched_fd, [a, b])

    scheduled_fd = custom_matmul_scheduler(presched_fd, schedule_config)

    try:
        nvf_outputs = scheduled_fd.execute([a, b], profile=True)
    except Exception as e:
        if verbose:
            print(e)
        return -1

    if validate:
        baseline_output = torch.matmul(a, b)
        assert torch.allclose(nvf_outputs[0], baseline_output, atol=1e-2, rtol=1e-2)

    prof = scheduled_fd.profile()
    # convert to microseconds to match pytorch profiler units
    return prof.kernel_profiles[0].time_ms * 1e-3


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="For a single problem, run through a combination of matmul parameters and compare relative performance against nvjet."
    )
    parser.add_argument(
        "-f",
        "--baseline_filepath",
        type=str,
        help="The filepath to a json file for the nvjet baseline.",
    )
    parser.add_argument("m", type=int, help="The size of M dimension")
    parser.add_argument("n", type=int, help="The size of N dimension")
    parser.add_argument("k", type=int, help="The size of K dimension")
    parser.add_argument(
        "layout",
        type=str,
        choices=[layout.name for layout in Layout],
        help="The layout for matmul problem.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print matmul parameters and exceptions.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print matmul parameters and exceptions.",
    )
    args = parser.parse_args()

    problem_config = (args.m, args.n, args.k, args.layout)

    device_properties = torch.cuda.get_device_properties(0)
    # short-circuit: problem does not fit on device
    if (
        estimate_matmul_size(problem_config, torch.bfloat16)
        >= device_properties.total_memory
    ):
        assert False

    baseline_data = analyze_json(args.baseline_filepath)
    baseline_result = baseline_data[Layout[problem_config[3]]][problem_config[:3]]

    print(
        f"problem configuration, m: {args.m}, n: {args.n}, k: {args.k}, layout: {args.layout}"
    )
    for idx, scheduler_config in enumerate(
        itertools.product(*parameter_configurations.values())
    ):
        nvf_result = test_matmul_nvf(
            problem_config, scheduler_config, args.verbose, args.validate
        )
        normalized_result = baseline_result / nvf_result
        print(
            f"index: {idx}, baseline(us): {baseline_result: .3e}, nvfuser(us):{nvf_result: 3e}, normalized(us):{normalized_result: 2f}."
        )


# NOTE Scheduler _matmul_ ***rejected*** because : MatmulOp and LinearOp fusion
# is disabled by default. Enable it using NVFUSER_ENABLE=fuse_matmul
#
# CMD to generate pybench json:
# NVFUSER_ENABLE=fuse_matmul NVFUSER_DISABLE=matmul_expr_eval pytest benchmarks/python/test_matmul.py
# -vsk 'baseline and bf16 and fullred and not 138216-236592-50664-N and not 97952-319616-3232-TT'
# --disable-validation --benchmark-eager --benchmark-json=nvjet_pybench.json 2>&1 |
# tee ~/nvjet_pybench.log
#
# Script CMD:
# NVFUSER_ENABLE=fuse_matmul NVFUSER_DISABLE=matmul_expr_eval python single_matmul.py nvjet_pybench.json
# 1752 4720 584 NN -f ~/workspace/hopper_benchmarks/gh200_matmul_eager.json
if __name__ == "__main__":
    main()

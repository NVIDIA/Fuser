# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from nvfuser import (
    FusionDefinition,
    SchedulerType,
    ClusterDims,
    MatMulTileOptions,
    GemmTile,
    MmaMacroEncode,
    MmaMacroArch,
    MatmulTileRasterizationOrder,
    MatmulCircularBufferingStrategy,
)
import torch
from torch.autograd import DeviceType
from torch.profiler import profile, record_function, ProfilerActivity
import math
import itertools
from enum import IntEnum


class Layout(IntEnum):
    NN = 0
    NT = 1
    TN = 2
    TT = 3


def estimate_matmul_size(config, dtype):
    def _estimate_size(shape, dtype):
        return math.prod(shape) * dtype.itemsize

    m, n, k, layout = config
    total_in_gbs = 0
    for shape in [[m, k], [n, k], [m, n]]:
        total_in_gbs += _estimate_size(shape, dtype)
    return total_in_gbs


def get_kernel_time(prof_averages: torch.autograd.profiler_util.EventList):
    elapsed_cuda_time = 0
    has_cuda_event = False
    for event in prof_averages:
        if event.device_type != DeviceType.CUDA:
            continue
        has_cuda_event = True
        elapsed_cuda_time = (
            elapsed_cuda_time + event.self_device_time_total
            if hasattr(event, "self_device_time_total")
            else event.self_cuda_time_total
        )
    assert has_cuda_event, "No CUDA events found"
    return elapsed_cuda_time / 1e3


def matmul_fusion(fd: FusionDefinition, inputs: list[torch.Tensor]) -> None:
    a = fd.from_pytorch(inputs[0])
    b = fd.from_pytorch(inputs[1])
    out = fd.ops.matmul(a, b)
    fd.add_output(out)


class MatmulDefinition(FusionDefinition):
    def __init__(self, inputs, config, verbose=False):
        super().__init__()
        self.inputs = inputs
        self.config = config
        self.verbose = verbose

    def definition(self):
        matmul_fusion(self, self.inputs)

    def schedule(self):
        assert self.config is not None
        status, error = self.sched.can_schedule(SchedulerType.matmul)
        assert status, error

        schedule_params = self.sched.compute_matmul_heuristics()

        # Modify original parameters
        tile_sizes, macro, tile_order, cluster_dims, stages = self.config
        schedule_params.tile_sizes = tile_sizes
        schedule_params.mma_macro = macro.mma_macro()
        schedule_params.cta_order = tile_order
        schedule_params.cluster_dims = cluster_dims
        schedule_params.circular_buffering_strategy = (
            MatmulCircularBufferingStrategy.warp_specialized
        )
        schedule_params.circular_buffer_options.circular_buffer_smem_write = stages > 1
        schedule_params.circular_buffer_options.smem_circular_buffer_stage = stages
        if self.verbose:
            print(schedule_params)

        # Schedule fusion
        self.sched.schedule()


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

    scheduled_fd = MatmulDefinition([a, b], schedule_config, verbose)

    try:
        nvf_outputs = scheduled_fd.execute([a, b], profile=True)
    except Exception as e:
        if verbose:
            print(e)
        return -1

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("matmul"):
            baseline_output = torch.matmul(a, b)
    baseline_time = get_kernel_time(prof.key_averages())

    if validate:
        tolerance = k * 1e-6
        assert torch.allclose(
            nvf_outputs[0], baseline_output, atol=tolerance, rtol=tolerance
        )

    prof = scheduled_fd.profile()
    nvf_time = prof.kernel_profiles[0].time_ms
    return baseline_time, nvf_time


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""Run through a combination of matmul parameters and compare relative performance against nvjet for a single problem.""",
        epilog="""How to run script: NVFUSER_ENABLE=fuse_matmul NVFUSER_DISABLE=matmul_expr_eval python single_matmul.py nvjet_pybench.json 1752 4720 584 NN --verbose --validate""",
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
        help="Validate nvfuser against pytorch matmul.",
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

    # These are the parameters we'll optimize
    parameter_configurations = {
        "tile_sizes": [
            MatMulTileOptions(GemmTile(128, 256, 64), GemmTile(64, 256, 64))
        ],
        "mma_macro": [MmaMacroEncode(MmaMacroArch.hopper, 64, 256, 16)],
        "tile_order": [MatmulTileRasterizationOrder.column_major],
        "cluster_dims": [ClusterDims(1, 1, 1)],
        "circular_buffer_stages": [4],
    }

    print(
        f"problem configuration, m: {args.m}, n: {args.n}, k: {args.k}, layout: {args.layout}"
    )
    for idx, scheduler_config in enumerate(
        itertools.product(*parameter_configurations.values())
    ):
        baseline_result, nvf_result = test_matmul_nvf(
            problem_config, scheduler_config, args.verbose, args.validate
        )
        normalized_result = baseline_result / nvf_result
        print(
            f"index: {idx}, baseline(us): {baseline_result: .3e}, "
            f"nvfuser(us): {nvf_result: 3e}, normalized(us): {normalized_result: 2f}"
        )


if __name__ == "__main__":
    main()

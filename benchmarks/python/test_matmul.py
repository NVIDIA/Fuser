# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, MatmulTileRasterizationOrder
from .core import run_benchmark
import torch

import csv
import os


def matmul_fusion(fd: FusionDefinition, inputs: list[torch.Tensor]) -> None:
    a = fd.from_pytorch(inputs[0])
    b = fd.from_pytorch(inputs[1])
    out = fd.ops.matmul(a, b)
    fd.add_output(out)


def load_matmul_problems():
    with open(os.path.join(os.path.dirname(__file__), "matmul_problems.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        return list((int(m), int(n), int(k), layout) for m, n, k, layout in reader)


@pytest.mark.parametrize("half_reduction", [False, True], ids=["fullred", "halfred"])
@pytest.mark.parametrize("executor", ["eager"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "config", load_matmul_problems(), ids=lambda val: "-".join(str(v) for v in val)
)
def test_matmul_baseline_benchmark(
    benchmark,
    executor: str,
    config: tuple,
    dtype: torch.dtype,
    half_reduction: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    m, n, k, layout = config

    if (m * n + m * k + n * k) * 2 > 12 * (2**30):
        pytest.skip("Probable OOM")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = half_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = half_reduction

    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    if layout == "NT" or layout == "NN":
        a = a.as_strided(size=[m, k], stride=[1, m])
    if layout == "TN" or layout == "NN":
        b = b.as_strided(size=[k, n], stride=[1, k])

    # NOTE: we never need to validate eager, as it is our baseline
    run_benchmark(
        benchmark,
        lambda ab: torch.matmul(*ab),
        [a, b],
    )


@pytest.mark.parametrize("half_reduction", [False, True], ids=["fullred", "halfred"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "config", load_matmul_problems(), ids=lambda val: "-".join(str(v) for v in val)
)
def test_matmul_nvf_benchmark(
    benchmark,
    config: tuple,
    dtype: torch.dtype,
    half_reduction: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    m, n, k, layout = config

    if (m * n + m * k + n * k) * 2 > 12 * (2**30):
        pytest.skip("Probable OOM")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = half_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = half_reduction

    if half_reduction:
        # See https://github.com/NVIDIA/Fuser/pull/1719
        pytest.skip("Reduced precision reduction not implemented in nvFuser")

    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    if layout == "NT" or layout == "NN":
        a = a.as_strided(size=[m, k], stride=[1, m])
    if layout == "TN" or layout == "NN":
        b = b.as_strided(size=[k, n], stride=[1, k])

    with FusionDefinition() as fd:
        matmul_fusion(fd, [a, b])

    kwargs = dict(
        _enable_options=["fuse_matmul"], _disable_options=["matmul_expr_eval"]
    )

    if not disable_validation:
        eager_output = torch.matmul(a, b)
        fd.validate([a, b], [eager_output], **kwargs)

    if not disable_benchmarking:
        run_benchmark(benchmark, lambda *args: fd.execute(*args, **kwargs), [a, b])

    heuristic_params = {}

    # Record the automatic parameters
    def record_params():
        schedule_params = fd.sched.compute_matmul_heuristics()

        heuristic_params["cluster_x"] = schedule_params.cluster_dims.x
        heuristic_params["cluster_y"] = schedule_params.cluster_dims.y
        heuristic_params["cluster_z"] = schedule_params.cluster_dims.z
        heuristic_params["cta_m"] = schedule_params.tile_sizes.cta_tile.m
        heuristic_params["cta_n"] = schedule_params.tile_sizes.cta_tile.n
        heuristic_params["cta_k"] = schedule_params.tile_sizes.cta_tile.k
        heuristic_params["warp_m"] = schedule_params.tile_sizes.warp_tile.m
        heuristic_params["warp_n"] = schedule_params.tile_sizes.warp_tile.n
        heuristic_params["warp_k"] = schedule_params.tile_sizes.warp_tile.k
        heuristic_params["instruction_m"] = schedule_params.mma_macro.m
        heuristic_params["instruction_n"] = schedule_params.mma_macro.n
        heuristic_params["instruction_k"] = schedule_params.mma_macro.k
        heuristic_params[
            "stages"
        ] = schedule_params.circular_buffer_options.smem_circular_buffer_stage
        heuristic_params["cta_order"] = (
            "column_major"
            if schedule_params.cta_order == MatmulTileRasterizationOrder.column_major
            else "row_major"
        )
        heuristic_params["grid_swizzle_factor"] = schedule_params.grid_swizzle_factor
        heuristic_params["splitk_factor"] = schedule_params.splitk_factor

        # Schedule fusion
        fd.sched.schedule()

    fd.schedule = record_params
    fd.execute([a, b], **kwargs)
    benchmark.extra_info["heuristic_params"] = heuristic_params

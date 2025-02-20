# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from ..core import run_benchmark
import torch


def many_matmul_fusion(fd: FusionDefinition) -> None:
    x = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False
    )
    y = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False
    )
    a = fd.ops.add(x, y)
    for _ in range(5):
        a_transpose = fd.ops.permute(a, [1, 0])
        matmul_out = fd.ops.matmul(a_transpose, y)
        add_out = fd.ops.add(a_transpose, y)
        a = fd.ops.add(matmul_out, add_out)
    fd.add_output(a)


@pytest.mark.parametrize("host_bench_mode", ["compile", "steady", "dynamic"])
def test_many_segment_benchmark(
    benchmark,
    host_bench_mode: str,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [torch.randn(5, 5, device="cuda", dtype=torch.float) for _ in range(2)]

    # Generate multiple inputs to measure dynamic shape overhead.
    if host_bench_mode == "dynamic":
        # Note: Using these sizes to allow kernel reuse in dynamic.
        # Using sizes = [4, 8, 16, 32, 64, 128] led to heuristic mismatch and kernel recompilation.
        input_sizes = [5, 7, 9, 11]
        # Generate matrices of size x size dimensions
        inputs = [
            [
                torch.randn(size, size, device="cuda", dtype=torch.float)
                for _ in range(2)
            ]
            for size in input_sizes
        ]

    with FusionDefinition() as fd:
        many_matmul_fusion(fd)

    def validate(input):
        x, y = input
        eager_output = x + y
        for _ in range(5):
            eager_transpose = eager_output.t()
            matmul_out = torch.matmul(eager_transpose, y)
            add_out = eager_transpose + y
            eager_output = matmul_out + add_out
        fd.validate(input, [eager_output])

        # Validate number of segments
        _ = fd.execute(input, profile=True)
        num_segments = fd.profile().segments
        expected_segments = 12
        assert (
            num_segments == expected_segments
        ), f"Expected {expected_segments} fusion segments, got {num_segments}."

    if not disable_validation:
        if host_bench_mode == "dynamic":
            # Run validate for all input sizes.
            for input in inputs:
                validate(input)
        else:
            validate(inputs)

    if not disable_benchmarking:
        run_benchmark(
            benchmark,
            None,
            inputs,
            device=f"host:{host_bench_mode}",
            fusion_fn=many_matmul_fusion,
        )

# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import clear_cuda_cache
from .core import run_benchmark
import torch


def many_segment_fusion(fd: FusionDefinition) -> None:
    x = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Half, is_cpu=False
    )
    y = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Half, is_cpu=False
    )

    x = fd.ops.cast(x, dtype=DataType.Float)
    y = fd.ops.cast(y, dtype=DataType.Float)

    a = fd.ops.add(x, y)
    # Generate multiple segments using segment_set
    for _ in range(10):
        x = fd.ops.cos(a)
        y = fd.ops.sin(a)
        a = fd.ops.add(x, y)
        a = fd.ops.segment_set(a)

    a = fd.ops.cast(a, dtype=DataType.Half)
    fd.add_output(a)


@pytest.mark.parametrize("host_bench_mode", ["compile", "steady", "dynamic"])
def test_many_segment_benchmark(
    benchmark,
    host_bench_mode: str,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()

    inputs = [torch.randn(13, device="cuda", dtype=torch.float16) for _ in range(2)]

    # Generate multiple inputs to measure dynamic shape overhead.
    if host_bench_mode == "dynamic":
        input_sizes = [5, 10, 13, 15, 17, 20]
        inputs = [
            [torch.randn(size, device="cuda", dtype=torch.float16) for _ in range(2)]
            for size in input_sizes
        ]

    with FusionDefinition() as fd:
        many_segment_fusion(fd)

    def validate(input):
        eager_output = input[0] + input[1]
        for _ in range(10):
            x = torch.cos(eager_output)
            y = torch.sin(eager_output)
            eager_output = x + y
        fd.validate(input, [eager_output])

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
            fusion_fn=many_segment_fusion,
        )

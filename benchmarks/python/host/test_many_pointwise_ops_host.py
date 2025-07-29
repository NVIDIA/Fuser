# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from ..core import run_benchmark
import torch
from ..global_params import PROMOTE_DTYPES
from functools import partial


def pointwise_ops_fusion(fd: FusionDefinition, dtype: DataType, num_iters: int):
    x = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    y = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        x = fd.ops.cast(x, dtype=DataType.Float)
        y = fd.ops.cast(y, dtype=DataType.Float)

    a = fd.ops.add(x, y)
    for _ in range(num_iters):
        x = fd.ops.cos(a)
        y = fd.ops.sin(a)
        a = fd.ops.add(x, y)

    if dtype in PROMOTE_DTYPES:
        a = fd.ops.cast(a, dtype=dtype)
    fd.add_output(a)


# NOTE: num_iters restricted due to issue #1234.
@pytest.mark.parametrize("num_iters", [2, 8, 32, 128])
@pytest.mark.parametrize("host_bench_mode", ["compile", "steady", "dynamic"])
def test_pointwise_ops_benchmark(
    benchmark,
    num_iters: int,
    host_bench_mode: str,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [torch.randn(13, device="cuda", dtype=torch.float16) for _ in range(2)]

    # Generate multiple inputs to measure dynamic shape overhead.
    if host_bench_mode == "dynamic":
        input_sizes = [5, 10, 13, 15, 17, 20]
        inputs = [
            [torch.randn(size, device="cuda", dtype=torch.float16) for _ in range(2)]
            for size in input_sizes
        ]

    with FusionDefinition() as fd:
        pointwise_ops_fusion(fd, torch_dtype_to_nvfuser_dtype(torch.float16), num_iters)

    def validate(input):
        eager_output = input[0] + input[1]
        for _ in range(num_iters):
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
            fusion_fn=partial(
                pointwise_ops_fusion,
                dtype=torch_dtype_to_nvfuser_dtype(torch.float16),
                num_iters=num_iters,
            ),
        )

# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition
from nvfuser.pytorch_utils import clear_cuda_cache
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
@pytest.mark.parametrize("eager", [False, True], ids=["nvfuser", "eager"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "config", load_matmul_problems(), ids=lambda val: "-".join(str(v) for v in val)
)
def test_matmul_nvf_benchmark(
    benchmark,
    eager: bool,
    config: tuple,
    dtype: torch.dtype,
    half_reduction: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    m, n, k, layout = config

    clear_cuda_cache()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = half_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = half_reduction

    if half_reduction and not eager:
        # See https://github.com/NVIDIA/Fuser/pull/1719
        pytest.skip("Reduced precision reduction not implemented in nvFuser")

    try:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)

        if layout == "NT" or layout == "NN":
            a = a.as_strided(size=[m, k], stride=[1, m])
        if layout == "TN" or layout == "NN":
            b = b.as_strided(size=[k, n], stride=[1, k])

        if eager:
            # NOTE: we never need to validate eager, as it is our baseline
            run_benchmark(
                benchmark,
                lambda ab: torch.matmul(*ab),
                [a, b],
            )
        else:
            with FusionDefinition() as fd:
                matmul_fusion(fd, [a, b])

            if not disable_validation:
                eager_output = torch.matmul(a, b)
                fd.validate([a, b], [eager_output])

            if not disable_benchmarking:
                run_benchmark(benchmark, fd.execute, [a, b])

    except torch.OutOfMemoryError:
        pytest.skip("Test failed due to OutOfMemoryError")

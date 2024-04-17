# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch

import csv
import os


def matmul_fusion(fd: FusionDefinition, dtype: DataType) -> None:
    # Decide contiguity based on layout
    a = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    b = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    out = fd.ops.matmul(a, b)
    fd.add_output(out)


precision_types = {"H": torch.float16, "S": torch.float32, "T": torch.bfloat16}


def load_matmul_problems():
    with open(os.path.join(os.path.dirname(__file__), "matmul_problems.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        return list((int(m), int(n), int(k), layout) for m, n, k, layout in reader)


@pytest.mark.parametrize("config", load_matmul_problems())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduction_nvf_benchmark(
    benchmark,
    config: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()
    m, n, k, layout = config

    # a_dtype, b_dtype, out_dtype = [precision_types[prec] for prec in precision]
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    if layout == "NT" or layout == "NN":
        a = a.as_strided(size=[m, k], stride=[1, k])
    if layout == "TN" or layout == "NN":
        b = b.as_strided(size=[k, n], stride=[1, k])

    with FusionDefinition() as fd:
        matmul_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.matmul(a, b)
        fd.validate([a, b], [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [a, b])

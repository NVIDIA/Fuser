# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from nvfuser import FusionDefinition
import torch
import math
import csv
import os
from enum import Enum


def load_matmul_problems():
    with open(os.path.join(os.path.dirname(__file__), "matmul_problems.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        return list((int(m), int(n), int(k), layout) for m, n, k, layout in reader)


def estimate_matmul_size(config, dtype):
    def _estimate_size(shape, dtype):
        return math.prod(shape) * dtype.itemsize

    m, n, k, layout = config
    total_in_gbs = 0
    for shape in [[m, k], [n, k], [m, n]]:
        total_in_gbs += _estimate_size(shape, dtype)
    return total_in_gbs


def test_matmul(config: tuple):
    m, n, k, layout = config
    dtype = torch.bfloat16

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    if layout == "NT" or layout == "NN":
        a = a.as_strided(size=[m, k], stride=[1, m])
    if layout == "TN" or layout == "NN":
        b = b.as_strided(size=[k, n], stride=[1, k])

    return torch.matmul(a, b)


def profile_configuration(config):
    device_properties = torch.cuda.get_device_properties(0)
    # short-circuit: problem does not fit on device
    if estimate_matmul_size(config, torch.bfloat16) >= device_properties.total_memory:
        return
    test_matmul(config)


def main():
    import sys

    configurations = load_matmul_problems()
    for config in configurations:
        print("****", config)
        profile_configuration(config)


if __name__ == "__main__":
    main()

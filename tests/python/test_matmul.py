# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]
import torch
from nvfuser import (
    FusionDefinition,
    DataType
)


def test_issue_2532():
    def fusion_func(fd : FusionDefinition) -> None :
        T0 = fd.define_tensor(shape=[-1, -1, 1], contiguity=[True, None, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 0, 1])
        T1 = fd.define_tensor(shape=[-1, 1, -1], contiguity=[True, None, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
        T2 = fd.ops.sum(T1, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T3 = fd.ops.matmul(T0, T1)
        T4 = fd.ops.sum(T3, dims=[0], keepdim=False, dtype=DataType.Null)
        fd.add_output(T2)
        fd.add_output(T4)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.randn((262400,), dtype=torch.float32, device='cuda:0').as_strided((1025, 256, 1), (256, 1, 256)),
        torch.randn((1049600,), dtype=torch.float32, device='cuda:0').as_strided((1025, 1, 1024), (1024, 1024, 1)),
    ]
    _ = fd.execute(inputs)
# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import DataType
from direct_fusion_definition import FusionDefinition

inputs = [
    torch.ones(2, 4, 8, device="cuda"),
    torch.ones(2, 4, 8, device="cuda"),
]

with FusionDefinition() as fd:
    tv0 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    tv1 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    tv2 = fd.ops.add(tv0, tv1)
    fd.add_output(tv2)

outputs = fd.execute(inputs)
print(outputs)

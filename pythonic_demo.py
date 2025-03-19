# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import direct, DataType
from direct_fusion_definition import FusionDefinition

inputs = [
    torch.ones(2, 4, 8, device="cuda"),
    torch.ones(2, 4, 8, device="cuda"),
]

with FusionDefinition() as fd:
    tv0 = fd.from_pytorch(inputs[0])
    tv1 = fd.from_pytorch(inputs[1])
    tv2 = fd.ops.add(tv0, tv1)
    fd.add_output(tv2)

outputs = fd.execute(inputs)
print(outputs)

fd_str = direct.translate_fusion(fd.fusion)
print(fd_str)
exec(fd_str)
func_name = "nvfuser_fusion"

# Execute the python definition that was captured
with FusionDefinition() as fd_cap:
    eval(func_name)(fd_cap)

print(fd_cap.fusion.print_math())
captured_outputs = fd_cap.execute(inputs)
print(captured_outputs)

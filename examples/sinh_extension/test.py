# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import nvfuser_extension  # noqa: F401

torch.manual_seed(0)
t = torch.randn((5, 5), device="cuda")
expected = torch.sinh(t)
output = torch.ops.myop.sinh_nvfuser(t)

print("Expected:", expected)
print("Output:", output)

assert torch.allclose(output, expected)
print("They match!")

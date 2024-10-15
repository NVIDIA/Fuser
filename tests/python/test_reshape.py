# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from nvfuser import FusionDefinition

def test_unequal_size():
    with FusionDefinition() as fd:
        inp = fd.define_tensor([2, 3])
        out = fd.ops.reshape(inp, [5])
        fd.add_output(out)

    inp_tensor = torch.randn(2, 3, device="cuda")
    fd.execute([inp_tensor])

# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from nvfuser import FusionDefinition


def test_squeeze():
    with FusionDefinition() as fd:
        inp = fd.define_tensor([1, 1, 2], contiguity=True)
        out = fd.ops.squeeze(inp, [0, 1])
        out = fd.ops.slice(out, [1], [2])
        fd.add_output(out)

    inp_tensor = torch.randn(1, 1, 2, device="cuda")
    out_tensor = fd.execute([inp_tensor])[0]
    torch.testing.assert_close(
        out_tensor, inp_tensor.squeeze([0, 1])[1:2], rtol=0, atol=0
    )

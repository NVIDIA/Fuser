# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from nvfuser import FusionDefinition


def test_define_tensor_contiguous():
    with FusionDefinition() as fd:
        inp = fd.define_tensor([2, 3], contiguity=True)
        out = fd.ops.add(inp, inp)
        fd.add_output(out)

    inp_tensor = torch.randn(2, 3, device="cuda")
    out_tensor = fd.execute([inp_tensor])[0]
    torch.testing.assert_close(out_tensor, inp_tensor * 2)


def test_define_tensor_noncontiguous():
    with FusionDefinition() as fd:
        inp = fd.define_tensor([2, 3])
        out = fd.ops.add(inp, inp)
        fd.add_output(out)

    inp_tensor = torch.randn(8, device="cuda").as_strided([2, 3], [4, 1])
    out_tensor = fd.execute([inp_tensor])[0]
    torch.testing.assert_close(out_tensor, inp_tensor * 2)


def test_define_tensor_broadcast():
    with FusionDefinition() as fd:
        inp = fd.define_tensor([1, 2, 1], contiguity=True)
        out = fd.ops.add(inp, inp)
        fd.add_output(out)

    inp_tensor = torch.randn(1, 2, 1, device="cuda").as_strided([1, 2, 1], [0, 1, 0])
    out_tensor = fd.execute([inp_tensor])[0]
    torch.testing.assert_close(out_tensor, inp_tensor * 2)

# SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from nvfuser_direct import FusionDefinition, DataType


def test_contiguous():
    with FusionDefinition() as fd:
        inp = fd.define_tensor([-1, -1], dtype=DataType.Float, contiguity=True)
        out = fd.ops.sum(inp, [-1])
        fd.add_output(out)

    # It might not be obvious but this tensor is indeed contiguous by
    # definition. inp.size(0) == 1 so inp.stride(0) shouldn't matter.
    #
    # I ran into such a tensor when I shard a reference tensor for a particular
    # device using slicing:
    # ```
    # inp = torch.arange(4, device="cuda").view(2, 2)
    # inp = inp[0:1, 0:1].contiguous()
    # ```
    # Despite `.contiguous()`, inp.stride(0) remains to be 2.
    inp = torch.as_strided(torch.ones([1], device="cuda"), [1, 1], [2, 1])
    (out,) = fd.execute([inp])
    torch.testing.assert_close(out.cpu(), torch.ones([1]))

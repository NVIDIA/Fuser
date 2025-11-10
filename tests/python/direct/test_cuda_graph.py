# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from contextlib import contextmanager

import torch

from nvfuser_direct import DataType, FusionDefinition


@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def test_capture_and_replay():
    with FusionDefinition() as fd:
        x = fd.define_tensor(shape=[-1, -1], dtype=DataType.Float)
        y = fd.define_tensor(shape=[-1, -1], dtype=DataType.Float)
        z = fd.ops.add(x, y)
        fd.add_output(z)

    with nvtx_range("run"):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(5):
                x = torch.randn(2, 3, device="cuda")
                y = torch.randn(2, 3, device="cuda")
                outs = fd.execute([x, y], _disable_options=["kernel_reuse"])
        stream.synchronize()

    torch.testing.assert_close(outs[0], x + y)

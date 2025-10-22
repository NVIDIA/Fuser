# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch

from nvfuser_direct import DataType, FusionDefinition


def test_capture_and_replay():
    with FusionDefinition() as fd:
        x = fd.define_tensor(shape=[-1, -1], dtype=DataType.Float)
        y = fd.define_tensor(shape=[-1, -1], dtype=DataType.Float)
        z = fd.ops.add(x, y)
        fd.add_output(z)

    x = torch.randn(2, 3, device="cuda")
    y = torch.randn(2, 3, device="cuda")
    for _ in range(5):
        out = fd.execute([x, y], _disable_options=["kernel_reuse"])
    torch.testing.assert_close(out[0], x + y)

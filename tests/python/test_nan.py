# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
from nvfuser import FusionDefinition, DataType


def test_validate_precomputed_values():
    def compare() -> FusionDefinition:
        with FusionDefinition() as fd:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )

            S1 = fd.define_scalar(None, dtype=DataType.Double)
            T2 = fd.ops.ge(T0, S1)
            fd.add_output(T2)
        return fd

    fd = compare()
    outs = fd.execute(
        [
            torch.randn((10,), dtype=torch.float32, device="cuda:0").as_strided(
                (2, 5), (5, 1)
            ),
            float("nan"),
        ]
    )
    # Cmoparing any number to NaN results in False.
    torch.testing.assert_close(outs[0].cpu(), torch.full((2, 5), False))


def test_dynamic_reshape():
    e = 768

    with FusionDefinition() as fd:
        inp = fd.define_tensor([-1, -1, e], contiguity=True)
        out = fd.ops.reshape(inp, [-1, e])
        fd.add_output(out)

    inp = torch.randn(2, 3, e, device="cuda")
    (out,) = fd.execute([inp])

    torch.testing.assert_close(out, inp.view(-1, e))

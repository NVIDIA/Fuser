# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import nvtx
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

    ins = [
        torch.randn((10,), dtype=torch.float32, device="cuda:0").as_strided(
            (2, 5), (5, 1)
        ),
        float("nan"),
    ]

    # nsys profile --capture-range=nvtx --capture-range-end=stop --nvtx-capture=test_nan --stats=true pytest tests/python/test_nan.py -s
    with nvtx.annotate("test_nan"):
        outs = fd.execute(ins)
        torch.cuda.synchronize()
        torch.testing.assert_close(outs[0].cpu(), torch.full((2, 5), False))

    outs = fd.execute(ins)
    torch.cuda.synchronize()
    # Cmoparing any number to NaN results in False.
    torch.testing.assert_close(outs[0].cpu(), torch.full((2, 5), False))

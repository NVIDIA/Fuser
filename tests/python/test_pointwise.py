# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
from nvfuser import FusionDefinition, DataType


def test_issue_2395():
    def create_fusion(fd: FusionDefinition) -> None:
        cond0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, None],
            dtype=DataType.Bool,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        values = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        cond1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Bool,
            is_cpu=False,
            stride_order=[1, 0],
        )
        cond1 = fd.ops.broadcast_in_dim(
            cond1, shape=[16, 16, 32], broadcast_dims=[0, 1]
        )
        sliced = fd.ops.slice(
            values,
            start_indices=[0, 0, 16],
            end_indices=[16, 16, 32],
            strides=[1, 1, 1],
        )
        zero = fd.define_scalar(0.00000, dtype=DataType.Double)
        masked = fd.ops.where(cond1, zero, values)
        masked = fd.ops.where(cond0, zero, masked)
        fd.add_output(sliced)
        fd.add_output(masked)

    with FusionDefinition() as fd:
        create_fusion(fd)
    ins = [
        torch.randint(0, 2, (256,), dtype=torch.bool, device="cuda:0").as_strided(
            (16, 16, 32), (16, 1, 0)
        ),
        torch.randn((8192,), dtype=torch.float32, device="cuda:0").as_strided(
            (16, 16, 32), (512, 32, 1)
        ),
        torch.randint(0, 2, (256,), dtype=torch.bool, device="cuda:0").as_strided(
            (16, 16), (16, 1)
        ),
    ]
    outs = fd.execute(ins)

    torch.testing.assert_close(outs[0], ins[1][:, :, 16:], rtol=0, atol=0)
    torch.testing.assert_close(
        outs[1],
        torch.where(
            torch.logical_or(ins[0] == 1, ins[2].unsqueeze(-1) == 1), 0, ins[1]
        ),
    )


# Tests that CPU scalar tensor can be instantiated using fd.from_pytorch
def test_cpu_add():
    inputs = [
        torch.tensor(2.0, device="cpu", dtype=torch.float),
        torch.randn(3, device="cuda", dtype=torch.float),
    ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        s0 = fd.from_pytorch(inputs[1])
        t1 = fd.ops.add(t0, s0)
        fd.add_output(t1)

    with FusionDefinition() as fd:
        fusion_func(fd)
    nvf_out = fd.execute(inputs)
    torch.testing.assert_close(nvf_out[0], inputs[0] + inputs[1])

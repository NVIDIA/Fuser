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


# Test bcast to different extents, issue-3227.
def test_bcast_different_extent():
    def nvfuser_fusion(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, 4, 2, 3],
            contiguity=[None, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[1, 5, 2, 3],
            contiguity=[None, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[1, 2, 3],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        S1 = fd.define_scalar(1, dtype=DataType.Int)
        S2 = fd.define_scalar(1, dtype=DataType.Int)
        S3 = fd.define_scalar(2, dtype=DataType.Int)
        S4 = fd.define_scalar(3, dtype=DataType.Int)
        T3 = fd.ops.broadcast_in_dim(
            T2, shape=[S1, S2, S3, S4], broadcast_dims=[1, 2, 3]
        )
        # bcast T2 to [1, 4, 2, 3]
        S5 = fd.define_scalar(1, dtype=DataType.Int)
        S6 = fd.define_scalar(4, dtype=DataType.Int)
        S7 = fd.define_scalar(2, dtype=DataType.Int)
        S8 = fd.define_scalar(3, dtype=DataType.Int)
        T4 = fd.ops.broadcast_in_dim(
            T3, shape=[S5, S6, S7, S8], broadcast_dims=[0, 1, 2, 3]
        )
        # bcast T2 to [1, 5, 2, 3]
        S9 = fd.define_scalar(1, dtype=DataType.Int)
        S10 = fd.define_scalar(5, dtype=DataType.Int)
        S11 = fd.define_scalar(2, dtype=DataType.Int)
        S12 = fd.define_scalar(3, dtype=DataType.Int)
        T5 = fd.ops.broadcast_in_dim(
            T3, shape=[S9, S10, S11, S12], broadcast_dims=[0, 1, 2, 3]
        )
        # add with T0
        T6 = fd.ops.add(T4, T0)
        # add with T1
        T7 = fd.ops.add(T5, T1)
        fd.add_output(T6)
        fd.add_output(T7)

    with FusionDefinition() as fd:
        nvfuser_fusion(fd)

    inputs = [
        torch.rand(24, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 4, 2, 3), (24, 6, 3, 1)
        ),
        torch.rand(30, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 5, 2, 3), (30, 6, 3, 1)
        ),
        torch.rand(6, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2, 3), (6, 3, 1)
        ),
    ]
    nvf_out = fd.execute(inputs)
    torch.testing.assert_close(nvf_out[0], inputs[0] + inputs[2])
    torch.testing.assert_close(nvf_out[1], inputs[1] + inputs[2])


# Test bcast to different extents, issue-3227.
def test_bcast_different_extent():
    def nvfuser_fusion(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, 4, 2, 3],
            contiguity=[None, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[1, 5, 2, 3],
            contiguity=[None, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[1, 2, 3],
            contiguity=[None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        S1 = fd.define_scalar(1, dtype=DataType.Int)
        S2 = fd.define_scalar(1, dtype=DataType.Int)
        S3 = fd.define_scalar(2, dtype=DataType.Int)
        S4 = fd.define_scalar(3, dtype=DataType.Int)
        T3 = fd.ops.broadcast_in_dim(
            T2, shape=[S1, S2, S3, S4], broadcast_dims=[1, 2, 3]
        )
        # bcast T2 to [1, 4, 2, 3]
        S5 = fd.define_scalar(1, dtype=DataType.Int)
        S6 = fd.define_scalar(4, dtype=DataType.Int)
        S7 = fd.define_scalar(2, dtype=DataType.Int)
        S8 = fd.define_scalar(3, dtype=DataType.Int)
        T4 = fd.ops.broadcast_in_dim(
            T3, shape=[S5, S6, S7, S8], broadcast_dims=[0, 1, 2, 3]
        )
        # bcast T2 to [1, 5, 2, 3]
        S9 = fd.define_scalar(1, dtype=DataType.Int)
        S10 = fd.define_scalar(5, dtype=DataType.Int)
        S11 = fd.define_scalar(2, dtype=DataType.Int)
        S12 = fd.define_scalar(3, dtype=DataType.Int)
        T5 = fd.ops.broadcast_in_dim(
            T3, shape=[S9, S10, S11, S12], broadcast_dims=[0, 1, 2, 3]
        )
        # add with T0
        T6 = fd.ops.add(T4, T0)
        # add with T1
        T7 = fd.ops.add(T5, T1)
        fd.add_output(T6)
        fd.add_output(T7)

    with FusionDefinition() as fd:
        nvfuser_fusion(fd)

    inputs = [
        torch.rand(24, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 4, 2, 3), (24, 6, 3, 1)
        ),
        torch.rand(30, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 5, 2, 3), (30, 6, 3, 1)
        ),
        torch.rand(6, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2, 3), (6, 3, 1)
        ),
    ]
    nvf_out = fd.execute(inputs)
    torch.testing.assert_close(nvf_out[0], inputs[0] + inputs[2])
    torch.testing.assert_close(nvf_out[1], inputs[1] + inputs[2])

def test_inplace_issue2664():
    def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
        T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False, stride_order=[0])
        T2 = fd.define_tensor(shape=[], contiguity=[], dtype=DataType.Float, is_cpu=False)
        S3 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T4 = fd.ops.add(T2, S3)
        S5 = fd.define_scalar(4194304, dtype=DataType.Int)
        V6 = fd.define_vector([S5], dtype=DataType.Int)
        T7 = fd.ops.broadcast_in_dim(T4, shape=V6, broadcast_dims=[])
        # T7_seg = fd.ops.segment_set(T7)
        T8 = fd.ops.mul(T1, T7)
        fd.add_output(T4, T2)
        fd.add_output(T8)

    with FusionDefinition() as fd:
        nvfuser_fusion_id0(fd)

    inputs = [
        torch.randn((4194304,), dtype=torch.float32, device='cuda:0').as_strided((4194304,), (1,)),
        torch.randn((1,), dtype=torch.float32, device='cuda:0').as_strided((), ()),
    ]
    ref_out = (inputs[-1] + 1.0) * inputs[0]
    
    out = fd.execute(inputs)
    torch.testing.assert_close(out[0], ref_out)

def test_inplace_bcast():
    def fusion_func(fd: FusionDefinition) -> None:
        T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False, stride_order=[0])
        T2 = fd.define_tensor(shape=[], contiguity=[], dtype=DataType.Float, is_cpu=False)
        S5 = fd.define_scalar(4194304, dtype=DataType.Int)
        V6 = fd.define_vector([S5], dtype=DataType.Int)
        T7 = fd.ops.broadcast_in_dim(T2, shape=V6, broadcast_dims=[])
        T8 = fd.ops.sum(T7, dims=[0], keepdim=False)
        T9 = fd.ops.mul(T1, T7)
        T10 = fd.ops.add (T1, T7)
        fd.add_output(T8, T2)
        fd.add_output(T9)
        fd.add_output(T10)
    with FusionDefinition() as fd:
        fusion_func(fd)
        
    inputs = [
        torch.randn((4194304,), dtype=torch.float32, device='cuda:0').as_strided((4194304,), (1,)),
        torch.randn((1,), dtype=torch.float32, device='cuda:0').as_strided((), ()),
    ]
    ref_out = [inputs[-1] * inputs[0], inputs[0] + inputs[1]]
    
    out = fd.execute(inputs)
    
    torch.testing.assert_close(out[0], ref_out[0])
    torch.testing.assert_close(out[1], ref_out[1])

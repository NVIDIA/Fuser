# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
from nvfuser import FusionDefinition, DataType
import pytest


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


# Example 1: Repro from https://github.com/NVIDIA/Fuser/issues/2664
# T4 (scalar) is broadcasted and used in mul computation. It is also used to update T2 inplace.
# This causes a RW race. In this case, the aliased tensor is a producer of bcast op.
def test_inplace_issue2664():
    def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
        T1 = fd.define_tensor(
            shape=[-1],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        T2 = fd.define_tensor(
            shape=[], contiguity=[], dtype=DataType.Float, is_cpu=False
        )
        S3 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T4 = fd.ops.add(T2, S3)
        S5 = fd.define_scalar(4194304, dtype=DataType.Int)
        V6 = fd.define_vector([S5], dtype=DataType.Int)
        T7 = fd.ops.broadcast_in_dim(T4, shape=V6, broadcast_dims=[])
        T8 = fd.ops.mul(T1, T7)
        fd.add_output(T4, T2)
        fd.add_output(T8)

    with FusionDefinition() as fd:
        nvfuser_fusion_id0(fd)

    inputs = [
        torch.randn((4194304,), dtype=torch.float32, device="cuda:0").as_strided(
            (4194304,), (1,)
        ),
        torch.randn((1,), dtype=torch.float32, device="cuda:0").as_strided((), ()),
    ]
    # Reference out = T4 (aliased to inputs[-1]), T8
    ref_out = [inputs[-1] + 1.0, (inputs[-1] + 1.0) * inputs[0]]

    out = fd.execute(inputs, profile=False)

    # Disabled due to CUDA 13 compatibility
    # assert fd.profile().segments == 2

    torch.testing.assert_close(inputs[-1], ref_out[0])
    torch.testing.assert_close(out[0], ref_out[1])


# Example 2 for Issue 2664:
# T2 is broadcasted and used in mul/add compute. It is also summed (T8) and used to inplace update T2.
# In this case, the aliased tensor (T8) is a consumer of the bcast op.
def test_inplace_post_bcast():
    def fusion_func(fd: FusionDefinition) -> None:
        T1 = fd.define_tensor(
            shape=[-1],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        T2 = fd.define_tensor(
            shape=[], contiguity=[], dtype=DataType.Float, is_cpu=False
        )
        S5 = fd.define_scalar(4194304, dtype=DataType.Int)
        V6 = fd.define_vector([S5], dtype=DataType.Int)
        T7 = fd.ops.broadcast_in_dim(T2, shape=V6, broadcast_dims=[])
        T8 = fd.ops.sum(T7, dims=[0], keepdim=False)
        T9 = fd.ops.mul(T1, T7)
        T10 = fd.ops.add(T1, T7)
        fd.add_output(T8, T2)
        fd.add_output(T9)
        fd.add_output(T10)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.randn((4194304,), dtype=torch.float32, device="cuda:0").as_strided(
            (4194304,), (1,)
        ),
        torch.randn((1,), dtype=torch.float32, device="cuda:0").as_strided((), ()),
    ]

    # Reference out = T8 (aliased to inputs[-1]), T9, T10
    ref_out = [
        inputs[-1] * inputs[0].size(0),
        inputs[-1] * inputs[0],
        inputs[0] + inputs[1],
    ]

    out = fd.execute(inputs, profile=False)

    # Disabled due to CUDA 13 compatibility
    # assert fd.profile().segments == 2

    torch.testing.assert_close(inputs[-1], ref_out[0])
    torch.testing.assert_close(out[0], ref_out[1])
    torch.testing.assert_close(out[1], ref_out[2])


# Example 3 for Issue 2664: This case involves two inplace updates.
# T7 is aliased to T2: T7 is not a producer/consumer of the bcast op, but the aliased input T2 is a producer of the bcast op.
# T6 is aliased to T3: T6 is a consumer of the bcast op.
def test_multi_inplace():
    def fusion_func(fd: FusionDefinition) -> None:
        T1 = fd.define_tensor(
            shape=[-1],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        T2 = fd.define_tensor(
            shape=[], contiguity=[], dtype=DataType.Float, is_cpu=False
        )
        T3 = fd.define_tensor(
            shape=[], contiguity=[], dtype=DataType.Float, is_cpu=False
        )
        T4 = fd.ops.broadcast_in_dim(T2, shape=T1.shape(), broadcast_dims=[])
        T5 = fd.ops.add(T1, T4)
        T6 = fd.ops.sum(T5, dims=[0], keepdim=False)
        S0 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T7 = fd.ops.add(T3, S0)
        fd.add_output(T6, T3)
        fd.add_output(T7, T2)

    with FusionDefinition() as fd:
        fusion_func(fd)

    inputs = [
        torch.randn((4194304,), dtype=torch.float32, device="cuda:0").as_strided(
            (4194304,), (1,)
        ),
        torch.randn((1,), dtype=torch.float32, device="cuda:0").as_strided((), ()),
        torch.randn((1,), dtype=torch.float32, device="cuda:0").as_strided((), ()),
    ]

    # Reference out = T6 (aliased to inputs[2]), T7 (aliased to inputs[1])
    ref_out = [inputs[-1] + 1.0, (inputs[0] + inputs[1]).sum(dim=-1)]

    fd.execute(inputs, profile=False)
    # Disabled due to CUDA 13 compatibility
    # assert fd.profile().segments == 4

    torch.testing.assert_close(inputs[1], ref_out[0])
    torch.testing.assert_close(inputs[2], ref_out[1])


# Example 4 for Issue 2664: There is no explicit broadcast. However, the aliased input has a broadcast dimension that is concretized in the fusion.
# T0 has a implicit broadcast which is used in add(T3) and neg (T4). T4 is used to inplace update T0, which causes RW race.
def test_implicit_bcast_inplace():
    def fusion_func(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, 1],
            contiguity=[True, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T3 = fd.ops.add(T1, T0)
        T4 = fd.ops.neg(T0)
        fd.add_output(T3)
        fd.add_output(T4, T0)

    inputs = [
        torch.randn((4194304, 1), dtype=torch.float32, device="cuda:0"),
        torch.randn((4194304, 128), dtype=torch.float32, device="cuda:0"),
    ]
    with FusionDefinition() as fd:
        fusion_func(fd)
    ref_out = [inputs[0] + inputs[1], -inputs[0]]
    out = fd.execute(inputs)

    torch.testing.assert_close(ref_out[0], out[0])
    torch.testing.assert_close(ref_out[1], inputs[0])


# Test that an error is raised if there are segments
# with CPU outputs.
# See https://github.com/NVIDIA/Fuser/issues/2853.
def test_issue2853():
    inputs = [
        torch.tensor(2.0, device="cpu", dtype=torch.float),
        torch.randn(3, device="cuda", dtype=torch.float),
    ]

    def fusion_func(fd: FusionDefinition):
        tv0 = fd.from_pytorch(inputs[0])  # CPU scalar tensor
        tv1 = fd.from_pytorch(inputs[1])  # CUDA input
        s0 = fd.define_scalar(3.0)
        # CPU scalar only segment that should raise an error
        t2 = fd.ops.add(tv0, s0)  # Should be a CPU scalar tensor
        t3 = fd.ops.add(tv1, s0)
        fd.add_output(t2)
        fd.add_output(t3)

    with FusionDefinition() as fd:
        fusion_func(fd)
    with pytest.raises(
        RuntimeError, match="KernelExecutor does not support the Fusion provided."
    ):
        _ = fd.execute(inputs)


# This example contains CPU scalar only fusion inputs.
# The `full` op does not take any fusion inputs but generates a
# CUDA tensor. This is a nvFuser supported fusion since the final
# output is a CUDA tensor.
def test_full_with_cpu_inputs():
    inputs = [
        torch.tensor(2.0, device="cpu", dtype=torch.float),
    ]

    def fusion_func(fd: FusionDefinition):
        tv0 = fd.from_pytorch(inputs[0])
        s0 = fd.define_scalar(3.0)
        tv1 = fd.ops.full(shape=[2, 2], fill_value=s0, dtype=DataType.Float)
        t2 = fd.ops.mul(tv0, tv1)  # CPU scalar * CUDA tensor = CUDA tensor
        fd.add_output(t2)

    with FusionDefinition() as fd:
        fusion_func(fd)
    _ = fd.execute(inputs)


# If fusion segment do not consist of any exprs, no kernel is
# launched and the output is on the correct device.
def test_input_forwarding_device():
    inputs = [torch.tensor(2.0, device="cpu", dtype=torch.float)]

    def fusion_func(fd: FusionDefinition):
        tv0 = fd.from_pytorch(inputs[0])
        fd.add_output(tv0)

    with FusionDefinition() as fd:
        fusion_func(fd)

    out = fd.execute(inputs)
    assert out[0].is_cpu


# Test single segment with CPU and CUDA outputs
def test_single_segment_multi_device():
    inputs = [
        torch.tensor(2.0, device="cpu", dtype=torch.float),
        torch.tensor(3.0, device="cuda", dtype=torch.float),
    ]

    def fusion_func(fd: FusionDefinition):
        tv0 = fd.from_pytorch(inputs[0])
        s0 = fd.define_scalar(3.0)
        tv1 = fd.ops.add(tv0, s0)
        tv2 = fd.from_pytorch(inputs[1])
        tv3 = fd.ops.add(tv1, tv2)
        fd.add_output(tv1)
        fd.add_output(tv2)

    with FusionDefinition() as fd:
        fusion_func(fd)

    with pytest.raises(
        RuntimeError, match="KernelExecutor does not support the Fusion provided."
    ):
        _ = fd.execute(inputs)

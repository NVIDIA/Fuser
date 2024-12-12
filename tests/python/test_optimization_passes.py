# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
from nvfuser import FusionDefinition, DataType


# this example hits a segfault in nvfuser::preseg_passes::MovePadPass::replaceCat, where the old optimization pass updates the fusion within the filterByType generator, causing a dynamic cast on a dangling pointer.
def test_litgpt_variants_gpt_neox_like():
    def nvfuser_fusion_id4(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[128, 16],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[128, 16],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.define_tensor(
            shape=[5, 5, 192],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T12 = fd.ops.slice(
            T0, start_indices=[0, 0], end_indices=[5, 16], strides=[1, 1]
        )
        T22 = fd.ops.slice(
            T1, start_indices=[0, 0], end_indices=[5, 16], strides=[1, 1]
        )
        T29 = fd.ops.reshape(T2, new_shape=[5, 5, 4, 3, 16])
        T30 = fd.ops.permute(T29, dims=[0, 2, 3, 1, 4])
        T49 = fd.ops.slice(
            T30,
            start_indices=[0, 0, 0, 0, 0],
            end_indices=[5, 4, 1, 5, 16],
            strides=[1, 1, 1, 1, 1],
        )
        T68 = fd.ops.slice(
            T30,
            start_indices=[0, 0, 1, 0, 0],
            end_indices=[5, 4, 2, 5, 16],
            strides=[1, 1, 1, 1, 1],
        )
        T87 = fd.ops.slice(
            T30,
            start_indices=[0, 0, 2, 0, 0],
            end_indices=[5, 4, 3, 5, 16],
            strides=[1, 1, 1, 1, 1],
        )
        T93 = fd.ops.reshape(T49, new_shape=[5, 4, 5, 16])
        T99 = fd.ops.reshape(T68, new_shape=[5, 4, 5, 16])
        T105 = fd.ops.reshape(T87, new_shape=[5, 4, 5, 16])
        T121 = fd.ops.slice(
            T93,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 4, 5, 8],
            strides=[1, 1, 1, 1],
        )
        T137 = fd.ops.slice(
            T93,
            start_indices=[0, 0, 0, 8],
            end_indices=[5, 4, 5, 16],
            strides=[1, 1, 1, 1],
        )
        T138 = fd.ops.neg(T137)
        T139 = fd.ops.cat([T138, T121], dim=-1)
        S140 = fd.define_scalar(5, dtype=DataType.Int)
        S141 = fd.define_scalar(4, dtype=DataType.Int)
        S142 = fd.define_scalar(5, dtype=DataType.Int)
        S143 = fd.define_scalar(16, dtype=DataType.Int)
        T145 = fd.ops.broadcast_in_dim(
            T12, shape=[S140, S141, S142, S143], broadcast_dims=[2, 3]
        )
        T146 = fd.ops.mul(T93, T145)
        S147 = fd.define_scalar(5, dtype=DataType.Int)
        S148 = fd.define_scalar(4, dtype=DataType.Int)
        S149 = fd.define_scalar(5, dtype=DataType.Int)
        S150 = fd.define_scalar(16, dtype=DataType.Int)
        T152 = fd.ops.broadcast_in_dim(
            T22, shape=[S147, S148, S149, S150], broadcast_dims=[2, 3]
        )
        T153 = fd.ops.mul(T139, T152)
        T154 = fd.ops.add(T146, T153)
        T170 = fd.ops.slice(
            T99,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 4, 5, 8],
            strides=[1, 1, 1, 1],
        )
        T186 = fd.ops.slice(
            T99,
            start_indices=[0, 0, 0, 8],
            end_indices=[5, 4, 5, 16],
            strides=[1, 1, 1, 1],
        )
        T187 = fd.ops.neg(T186)
        T188 = fd.ops.cat([T187, T170], dim=-1)
        T189 = fd.ops.mul(T99, T145)
        T190 = fd.ops.mul(T188, T152)
        T191 = fd.ops.add(T189, T190)
        T207 = fd.ops.slice(
            T93,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 4, 5, 0],
            strides=[1, 1, 1, 1],
        )
        T208 = fd.ops.cat([T154, T207], dim=-1)
        T224 = fd.ops.slice(
            T99,
            start_indices=[0, 0, 0, 0],
            end_indices=[5, 4, 5, 0],
            strides=[1, 1, 1, 1],
        )
        T225 = fd.ops.cat([T191, T224], dim=-1)
        S226 = fd.define_scalar(0.500000, dtype=DataType.Double)
        T227 = fd.ops.mul(T208, S226)
        T228 = fd.ops.permute(T225, dims=[0, 1, 3, 2])
        S229 = fd.define_scalar(0.500000, dtype=DataType.Double)
        T230 = fd.ops.mul(T228, S229)
        fd.add_output(T105)
        fd.add_output(T145)
        fd.add_output(T152)
        fd.add_output(T227)
        fd.add_output(T230)

    with FusionDefinition() as fd:
        nvfuser_fusion_id4(fd)

    inputs = [
        torch.testing.make_tensor((128, 16), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((128, 16), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((5, 5, 192), dtype=torch.float32, device="cuda:0"),
    ]
    # TODO: I wish we have an easy way for validation like in cpp tests.
    fd.execute(inputs)


# https://github.com/NVIDIA/Fuser/issues/3369
# don't need to replace constants in the same Id set
def test_square_linear():
    def nvfuser_fusion_id28(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[5, 5],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[5, 5],
            contiguity=[True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.define_tensor(
            shape=[5],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        T3 = fd.ops.linear(T0, T1, T2)
        fd.add_output(T3)

    with FusionDefinition() as fd:
        nvfuser_fusion_id28(fd)

    inputs = [
        torch.testing.make_tensor((5, 5), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((5, 5), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((5,), dtype=torch.float32, device="cuda:0"),
    ]
    fd.execute(inputs)

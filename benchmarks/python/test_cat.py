# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from nvfuser import FusionDefinition, DataType
from .core import run_benchmark, with_executor

# These tests are sourced from automation added into thunder.
#
# To recreate:
#   * Use the tfogal/benchmarking-dumps branch in thunder.
#   * Pass nv_store_fusion_inputs=True to your thunderfx call
#   * Disable the torch.compile executor in the thunderfx call, which ensures
#     that all fusions go to nvFuser.
#   * After running the network of interest, invoke thunder.tests.dump.traces()
# This particular file is aimed at concatenation operations, so it is an
# agglomeration of cases that included `torch.cat` in the fusion.


def cat_qwen2_fwd_11(t17353, t17351, cos_2, sin_2, query_states_1, key_states_1):
    # t17353: "cuda:0 bf16[2048, 512]"
    # t17351: "cuda:0 bf16[1, 2048, 512]"
    # cos_2: "cuda:0 bf16[1, 2048, 128]"
    # sin_2: "cuda:0 bf16[1, 2048, 128]"
    # query_states_1: "cuda:0 bf16[1, 28, 2048, 128]"
    # key_states_1: "cuda:0 bf16[1, 4, 2048, 128]"
    t0 = torch.reshape(t17353, (1, 2048, 512))  # t0: "cuda:0 bf16[1, 2048, 512]"
    t1 = torch.Tensor.to(t0, torch.float32, copy=True)  # t1: "cuda:0 f32[1, 2048, 512]"
    # t1 = prims.convert_element_type(t0, dtypes.float32)  # t1: "cuda:0 f32[1, 2048, 512]"
    t2 = torch.mul(t1, 2.0)  # t2: "cuda:0 f32[1, 2048, 512]"
    t3 = torch.Tensor.to(
        t17351, torch.float32, copy=True
    )  # t3: "cuda:0 f32[1, 2048, 512]"
    # t3 = prims.convert_element_type(t17351, dtypes.float32)  # t3: "cuda:0 f32[1, 2048, 512]"
    t4 = torch.add(t3, t2)  # t4: "cuda:0 f32[1, 2048, 512]"
    t5 = torch.Tensor.to(
        t4, torch.bfloat16, copy=True
    )  # t5: "cuda:0 bf16[1, 2048, 512]"
    # t5 = prims.convert_element_type(t4, dtypes.bfloat16)  # t5: "cuda:0 bf16[1, 2048, 512]"
    t6 = torch.reshape(t5, (1, 2048, 4, 128))  # t6: "cuda:0 bf16[1, 2048, 4, 128]"
    t7 = torch.permute(t6, (0, 2, 1, 3))  # t7: "cuda:0 bf16[1, 4, 2048, 128]"
    # t7 = prims.transpose(t6, (0, 2, 1, 3))  # t7: "cuda:0 bf16[1, 4, 2048, 128]"
    t8 = torch.unsqueeze(cos_2, 1)  # t8: "cuda:0 bf16[1, 1, 2048, 128]"
    # t8 = prims.broadcast_in_dim(cos_2, [1, 1, 2048, 128], [0, 2, 3])  # t8: "cuda:0 bf16[1, 1, 2048, 128]"
    t9 = torch.Tensor.expand(
        t8, [1, 1, 2048, 128]
    )  # t9: "cuda:0 bf16[1, 1, 2048, 128]"
    # t9 = prims.broadcast_in_dim(t8, (1, 1, 2048, 128), (0, 1, 2, 3))  # t9: "cuda:0 bf16[1, 1, 2048, 128]"

    t10 = torch.unsqueeze(sin_2, 1)  # t10: "cuda:0 bf16[1, 1, 2048, 128]"
    # t10 = prims.broadcast_in_dim(sin_2, [1, 1, 2048, 128], [0, 2, 3])  # t10: "cuda:0 bf16[1, 1, 2048, 128]"
    t11 = torch.Tensor.expand(
        t10, [1, 1, 2048, 128]
    )  # t11: "cuda:0 bf16[1, 1, 2048, 128]"
    # t11 = prims.broadcast_in_dim(t10, (1, 1, 2048, 128), (0, 1, 2, 3))  # t11: "cuda:0 bf16[1, 1, 2048, 128]"
    t12 = torch.Tensor.expand(
        t9, (1, 28, 2048, 128)
    )  # t12: "cuda:0 bf16[1, 28, 2048, 128]"
    # t12 = prims.broadcast_in_dim(t9, (1, 28, 2048, 128), (0, 1, 2, 3))  # t12: "cuda:0 bf16[1, 28, 2048, 128]"
    t13 = torch.Tensor.to(
        query_states_1, torch.float32, copy=True
    )  # t13: "cuda:0 f32[1, 28, 2048, 128]"
    # t13 = prims.convert_element_type(query_states_1, dtypes.float32)  # t13: "cuda:0 f32[1, 28, 2048, 128]"
    t14 = torch.Tensor.to(
        t12, torch.float32, copy=True
    )  # t14: "cuda:0 f32[1, 28, 2048, 128]"
    # t14 = prims.convert_element_type(t12, dtypes.float32)  # t14: "cuda:0 f32[1, 28, 2048, 128]"
    t15 = torch.mul(t13, t14)  # t15: "cuda:0 f32[1, 28, 2048, 128]"
    t16 = query_states_1[0:1, 0:28, 0:2048, 0:64]
    t17 = query_states_1[0:1, 0:28, 0:2048, 64:128]
    # t16 = torch.slice(query_states_1, [0, 0, 0, 0], [1, 28, 2048, 64], [1, 1, 1, 1])  # t16: "cuda:0 bf16[1, 28, 2048, 64]"
    # t17 = torch.slice(query_states_1, [0, 0, 0, 64], [1, 28, 2048, 128], [1, 1, 1, 1])  # t17: "cuda:0 bf16[1, 28, 2048, 64]"
    t18 = torch.Tensor.to(
        t17, torch.float32, copy=True
    )  # t18: "cuda:0 f32[1, 28, 2048, 64]"
    # t18 = prims.convert_element_type(t17, dtypes.float32)  # t18: "cuda:0 f32[1, 28, 2048, 64]"
    t19 = torch.neg(t18)  # t19: "cuda:0 f32[1, 28, 2048, 64]"
    t20 = torch.Tensor.to(
        t19, torch.bfloat16, copy=True
    )  # t20: "cuda:0 bf16[1, 28, 2048, 64]"
    # t20 = prims.convert_element_type(t19, dtypes.bfloat16)  # t20: "cuda:0 bf16[1, 28, 2048, 64]"
    t21 = torch.cat([t20, t16], -1)  # t21: "cuda:0 bf16[1, 28, 2048, 128]"
    t22 = torch.Tensor.expand(
        t11, (1, 28, 2048, 128)
    )  # t22: "cuda:0 bf16[1, 28, 2048, 128]"
    # t22 = prims.broadcast_in_dim(t11, (1, 28, 2048, 128), (0, 1, 2, 3))  # t22: "cuda:0 bf16[1, 28, 2048, 128]"
    t23 = torch.Tensor.to(
        t21, torch.float32, copy=True
    )  # t23: "cuda:0 f32[1, 28, 2048, 128]"
    # t23 = prims.convert_element_type(t21, dtypes.float32)  # t23: "cuda:0 f32[1, 28, 2048, 128]"
    t24 = torch.Tensor.to(
        t22, torch.float32, copy=True
    )  # t24: "cuda:0 f32[1, 28, 2048, 128]"
    # t24 = prims.convert_element_type(t22, dtypes.float32)  # t24: "cuda:0 f32[1, 28, 2048, 128]"
    t25 = torch.mul(t23, t24)  # t25: "cuda:0 f32[1, 28, 2048, 128]"
    t26 = torch.add(t15, t25)  # t26: "cuda:0 f32[1, 28, 2048, 128]"
    t27 = torch.Tensor.to(
        t26, torch.bfloat16, copy=True
    )  # t27: "cuda:0 bf16[1, 28, 2048, 128]"
    # t27 = prims.convert_element_type(t26, dtypes.bfloat16)  # t27: "cuda:0 bf16[1, 28, 2048, 128]"
    t28 = torch.Tensor.expand(
        t9, (1, 4, 2048, 128)
    )  # t28: "cuda:0 bf16[1, 4, 2048, 128]"
    # t28 = prims.broadcast_in_dim(t9, (1, 4, 2048, 128), (0, 1, 2, 3))  # t28: "cuda:0 bf16[1, 4, 2048, 128]"
    t29 = torch.Tensor.to(
        key_states_1, torch.float32, copy=True
    )  # t29: "cuda:0 f32[1, 4, 2048, 128]"
    # t29 = prims.convert_element_type(key_states_1, dtypes.float32)  # t29: "cuda:0 f32[1, 4, 2048, 128]"
    t30 = torch.Tensor.to(
        t28, torch.float32, copy=True
    )  # t30: "cuda:0 f32[1, 4, 2048, 128]"
    # t30 = prims.convert_element_type(t28, dtypes.float32)  # t30: "cuda:0 f32[1, 4, 2048, 128]"
    t31 = torch.mul(t29, t30)  # t31: "cuda:0 f32[1, 4, 2048, 128]"
    # t32 = torch.slice(key_states_1, [0, 0, 0, 0], [1, 4, 2048, 64], [1, 1, 1, 1])  # t32: "cuda:0 bf16[1, 4, 2048, 64]"
    # t33 = torch.slice(key_states_1, [0, 0, 0, 64], [1, 4, 2048, 128], [1, 1, 1, 1])  # t33: "cuda:0 bf16[1, 4, 2048, 64]"
    t32 = key_states_1[0:1, 0:4, 0:2048, 0:64]
    t33 = key_states_1[0:1, 0:4, 0:2048, 64:128]
    t34 = torch.Tensor.to(
        t33, torch.float32, copy=True
    )  # t34: "cuda:0 f32[1, 4, 2048, 64]"
    # t34 = prims.convert_element_type(t33, dtypes.float32)  # t34: "cuda:0 f32[1, 4, 2048, 64]"
    t35 = torch.neg(t34)  # t35: "cuda:0 f32[1, 4, 2048, 64]"
    t36 = torch.Tensor.to(
        t35, torch.bfloat16, copy=True
    )  # t36: "cuda:0 bf16[1, 4, 2048, 64]"
    # t36 = prims.convert_element_type(t35, dtypes.bfloat16)  # t36: "cuda:0 bf16[1, 4, 2048, 64]"
    t37 = torch.cat([t36, t32], -1)  # t37: "cuda:0 bf16[1, 4, 2048, 128]"
    t38 = torch.Tensor.expand(
        t11, (1, 4, 2048, 128)
    )  # t38: "cuda:0 bf16[1, 4, 2048, 128]"
    # t38 = prims.broadcast_in_dim(t11, (1, 4, 2048, 128), (0, 1, 2, 3))  # t38: "cuda:0 bf16[1, 4, 2048, 128]"
    t39 = torch.Tensor.to(
        t37, torch.float32, copy=True
    )  # t39: "cuda:0 f32[1, 4, 2048, 128]"
    # t39 = prims.convert_element_type(t37, dtypes.float32)  # t39: "cuda:0 f32[1, 4, 2048, 128]"
    t40 = torch.Tensor.to(
        t38, torch.float32, copy=True
    )  # t40: "cuda:0 f32[1, 4, 2048, 128]"
    # t40 = prims.convert_element_type(t38, dtypes.float32)  # t40: "cuda:0 f32[1, 4, 2048, 128]"
    t41 = torch.mul(t39, t40)  # t41: "cuda:0 f32[1, 4, 2048, 128]"
    t42 = torch.add(t31, t41)  # t42: "cuda:0 f32[1, 4, 2048, 128]"
    t43 = torch.Tensor.to(
        t42, torch.bfloat16, copy=True
    )  # t43: "cuda:0 bf16[1, 4, 2048, 128]"
    # t43 = prims.convert_element_type(t42, dtypes.bfloat16)  # t43: "cuda:0 bf16[1, 4, 2048, 128]"
    t44 = torch.unsqueeze(t43, 2)  # t44: "cuda:0 bf16[1, 4, 1, 2048, 128]"
    # t44 = prims.broadcast_in_dim(t43, [1, 4, 1, 2048, 128], [0, 1, 3, 4])  # t44: "cuda:0 bf16[1, 4, 1, 2048, 128]"
    t45 = torch.Tensor.expand(
        t44, [1, 4, 1, 2048, 128]
    )  # t45: "cuda:0 bf16[1, 4, 1, 2048, 128]"
    # t45 = prims.broadcast_in_dim(t44, (1, 4, 1, 2048, 128), (0, 1, 2, 3, 4))  # t45: "cuda:0 bf16[1, 4, 1, 2048, 128]"
    t46 = torch.Tensor.expand(
        t45, (1, 4, 7, 2048, 128)
    )  # t46: "cuda:0 bf16[1, 4, 7, 2048, 128]"
    # t46 = prims.broadcast_in_dim(t45, (1, 4, 7, 2048, 128), (0, 1, 2, 3, 4))  # t46: "cuda:0 bf16[1, 4, 7, 2048, 128]"
    t47 = torch.reshape(t46, (1, 28, 2048, 128))  # t47: "cuda:0 bf16[1, 28, 2048, 128]"
    t48 = torch.unsqueeze(t7, 2)  # t48: "cuda:0 bf16[1, 4, 1, 2048, 128]"
    # t48 = prims.broadcast_in_dim(t7, [1, 4, 1, 2048, 128], [0, 1, 3, 4])  # t48: "cuda:0 bf16[1, 4, 1, 2048, 128]"
    t49 = torch.Tensor.expand(
        t48, [1, 4, 1, 2048, 128]
    )  # t49: "cuda:0 bf16[1, 4, 1, 2048, 128]"
    # t49 = prims.broadcast_in_dim(t48, (1, 4, 1, 2048, 128), (0, 1, 2, 3, 4))  # t49: "cuda:0 bf16[1, 4, 1, 2048, 128]"
    t50 = torch.Tensor.expand(
        t49, (1, 4, 7, 2048, 128)
    )  # t50: "cuda:0 bf16[1, 4, 7, 2048, 128]"
    # t50 = prims.broadcast_in_dim(t49, (1, 4, 7, 2048, 128), (0, 1, 2, 3, 4))  # t50: "cuda:0 bf16[1, 4, 7, 2048, 128]"
    t51 = torch.reshape(t50, (1, 28, 2048, 128))  # t51: "cuda:0 bf16[1, 28, 2048, 128]"
    # t52 = torch_stride_order_prim_impl(t27, (3, 2, 1, 0))  # t52: "cuda:0 bf16[1, 28, 2048, 128]"
    # t53 = torch_stride_order_prim_impl(t47, (3, 2, 1, 0))  # t53: "cuda:0 bf16[1, 28, 2048, 128]"
    # t54 = torch_stride_order_prim_impl(t51, (3, 2, 1, 0))  # t54: "cuda:0 bf16[1, 28, 2048, 128]"
    t52 = torch.as_strided(t27, (1, 28, 2048, 128), (7340032, 7340032, 262144, 128))
    t53 = torch.as_strided(t47, (1, 28, 2048, 128), (7340032, 7340032, 262144, 128))
    t54 = torch.as_strided(t51, (1, 28, 2048, 128), (7340032, 7340032, 262144, 128))
    return [t7, t43, t52, t53, t54]


# Unfortunately Thunder does not support 'torch.as_strided' yet; until it
# does, we use this manual nvFuser definition.
def cat_qwen2_fwd_11_fusion(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[2048, 512],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T1 = fd.define_tensor(
        shape=[1, 2048, 512],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[1, 2048, 128],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 0, 1],
    )
    T3 = fd.define_tensor(
        shape=[1, 2048, 128],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 0, 1],
    )
    T4 = fd.define_tensor(
        shape=[1, 28, 2048, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    T5 = fd.define_tensor(
        shape=[1, 4, 2048, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    T10 = fd.ops.reshape(T0, new_shape=[1, 2048, 512])
    T11 = fd.ops.cast(T10, dtype=DataType.Float)
    S12 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T13 = fd.ops.mul(T11, S12)
    T14 = fd.ops.cast(T1, dtype=DataType.Float)
    T15 = fd.ops.add(T14, T13)
    T16 = fd.ops.cast(T15, dtype=DataType.BFloat16)
    T22 = fd.ops.reshape(T16, new_shape=[1, 2048, 4, 128])
    T23 = fd.ops.permute(T22, dims=[0, 2, 1, 3])
    T29 = fd.ops.broadcast_in_dim(T2, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3])
    T35 = fd.ops.broadcast_in_dim(T3, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3])
    T41 = fd.ops.broadcast_in_dim(
        T29, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T42 = fd.ops.cast(T4, dtype=DataType.Float)
    T43 = fd.ops.cast(T41, dtype=DataType.Float)
    T44 = fd.ops.mul(T42, T43)
    T60 = fd.ops.slice(
        T4,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 28, 2048, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T76 = fd.ops.slice(
        T4,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 28, 2048, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T77 = fd.ops.cast(T76, dtype=DataType.Float)
    T78 = fd.ops.neg(T77)
    T79 = fd.ops.cast(T78, dtype=DataType.BFloat16)
    T80 = fd.ops.cat([T79, T60], dim=-1, manual_padding=0)
    T86 = fd.ops.broadcast_in_dim(
        T35, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T87 = fd.ops.cast(T80, dtype=DataType.Float)
    T88 = fd.ops.cast(T86, dtype=DataType.Float)
    T89 = fd.ops.mul(T87, T88)
    T90 = fd.ops.add(T44, T89)
    T91 = fd.ops.cast(T90, dtype=DataType.BFloat16)
    T97 = fd.ops.broadcast_in_dim(
        T29, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T98 = fd.ops.cast(T5, dtype=DataType.Float)
    T99 = fd.ops.cast(T97, dtype=DataType.Float)
    T100 = fd.ops.mul(T98, T99)
    T116 = fd.ops.slice(
        T5,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 4, 2048, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T132 = fd.ops.slice(
        T5,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 4, 2048, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T133 = fd.ops.cast(T132, dtype=DataType.Float)
    T134 = fd.ops.neg(T133)
    T135 = fd.ops.cast(T134, dtype=DataType.BFloat16)
    T136 = fd.ops.cat([T135, T116], dim=-1, manual_padding=0)
    T142 = fd.ops.broadcast_in_dim(
        T35, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T143 = fd.ops.cast(T136, dtype=DataType.Float)
    T144 = fd.ops.cast(T142, dtype=DataType.Float)
    T145 = fd.ops.mul(T143, T144)
    T146 = fd.ops.add(T100, T145)
    T147 = fd.ops.cast(T146, dtype=DataType.BFloat16)
    T154 = fd.ops.broadcast_in_dim(
        T147, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4]
    )
    T161 = fd.ops.broadcast_in_dim(
        T154, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4]
    )
    T167 = fd.ops.reshape(T161, new_shape=[1, 28, 2048, 128])
    T174 = fd.ops.broadcast_in_dim(
        T23, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4]
    )
    T181 = fd.ops.broadcast_in_dim(
        T174, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4]
    )
    T187 = fd.ops.reshape(T181, new_shape=[1, 28, 2048, 128])
    T188 = fd.ops.stride_order(T91, stride_order=[3, 2, 1, 0])
    T189 = fd.ops.stride_order(T167, stride_order=[3, 2, 1, 0])
    T190 = fd.ops.stride_order(T187, stride_order=[3, 2, 1, 0])
    fd.add_output(T23)
    fd.add_output(T147)
    fd.add_output(T188)
    fd.add_output(T189)
    fd.add_output(T190)


def get_cat_qwen2_inputs() -> list[torch.Tensor]:
    inputs = [
        torch.randn(
            size=(2048, 512), dtype=torch.bfloat16, device="cuda", requires_grad=False
        ).as_strided((2048, 512), (512, 1)),
        torch.randn(
            size=(1, 2048, 512),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=False,
        ).as_strided((1, 2048, 512), (1048576, 512, 1)),
        torch.randn(
            size=(1, 2048, 128),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=False,
        ).as_strided((1, 2048, 128), (262144, 1, 2048)),
        torch.randn(
            size=(1, 2048, 128),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=False,
        ).as_strided((1, 2048, 128), (262144, 1, 2048)),
        torch.randn(
            size=(1, 28, 2048, 128),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=False,
        ).as_strided((1, 28, 2048, 128), (7340032, 128, 3584, 1)),
        torch.randn(
            size=(1, 4, 2048, 128),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=False,
        ).as_strided((1, 4, 2048, 128), (1048576, 128, 512, 1)),
    ]
    return inputs


def test_cat_qwen2_fwd_11_nvf_benchmark(
    benchmark, disable_validation: bool, disable_benchmarking: bool
):
    with FusionDefinition() as fd:
        cat_qwen2_fwd_11_fusion(fd)

    inputs = get_cat_qwen2_inputs()

    def benchmark_fn(inputs):
        return fd.execute(inputs)

    if not disable_validation:
        # torch.compile required; works around an issue in eager mode.
        tc = torch.compile(cat_qwen2_fwd_11)
        reference = tc(*inputs)
        # Temporarily disabled: leads to an illegal memory access.
        if False:
            fd.validate(inputs, reference)
    if not disable_benchmarking:
        run_benchmark(benchmark, benchmark_fn, inputs)


# Qwen2 fusion that involves concatenation. Note that there are no Mistral-Nemo
# benchmarks in this file, because they were all equivalent to this fusion. The
# fusion below appears repeatedly in both networks.
#
# The numbers are arbitrary; this was the 11th fusion in the forward pass.
#
# We don't use a "thunder" executor here because thunder cannot accept the
# torch.as_strided call yet. There's a separate benchmark for thunder for now.
@pytest.mark.parametrize("executor", ["torchcompile"])
@pytest.mark.expr_eval
@pytest.mark.pointwise
@pytest.mark.resize
@pytest.mark.transpose
def test_cat_qwen2_fwd_11_baseline_benchmark(benchmark, executor: str) -> None:
    inputs = get_cat_qwen2_inputs()

    def benchmark_fn(inputs):
        return cat_qwen2_fwd_11(*inputs)

    benchmark_fn = with_executor(executor, benchmark_fn)

    run_benchmark(benchmark, benchmark_fn, inputs)


@pytest.mark.parametrize("executor", ["thunder", "torchcompile"])
@pytest.mark.expr_eval
@pytest.mark.pointwise
@pytest.mark.resize
@pytest.mark.transpose
def test_cat_phi3_1_baseline_benchmark(
    benchmark, executor: str, disable_validation: bool
) -> None:
    def to_be_compiled(t14):
        # t14: "cuda:0 f32[1, 48, 2048]"
        t0 = torch.permute(t14, (0, 2, 1))  # t0: "cuda:0 f32[1, 2048, 48]"
        # t0 = prims.transpose(t14, (0, 2, 1))  # t0: "cuda:0 f32[1, 2048, 48]"
        t1 = torch.cat([t0, t0], -1)  # t1: "cuda:0 f32[1, 2048, 96]"
        t2 = torch.cos(t1)  # t2: "cuda:0 f32[1, 2048, 96]"
        t3 = torch.sin(t1)  # t3: "cuda:0 f32[1, 2048, 96]"
        t4 = torch.mul(t2, 1.1902380714238083)  # t4: "cuda:0 f32[1, 2048, 96]"
        t5 = torch.mul(t3, 1.1902380714238083)  # t5: "cuda:0 f32[1, 2048, 96]"
        t6 = torch.Tensor.to(
            t4, torch.bfloat16, copy=True
        )  # t6: "cuda:0 bf16[1, 2048, 96]"
        # t6 = prims.convert_element_type(t4, dtypes.bfloat16)  # t6: "cuda:0 bf16[1, 2048, 96]"
        t7 = torch.Tensor.to(
            t5, torch.bfloat16, copy=True
        )  # t7: "cuda:0 bf16[1, 2048, 96]"
        # t7 = prims.convert_element_type(t5, dtypes.bfloat16)  # t7: "cuda:0 bf16[1, 2048, 96]"
        return [t6, t7]

    inputs = [
        torch.randn(
            size=(1, 48, 2048),
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        ),
    ]

    def benchmark_fn(inputs):
        return to_be_compiled(*inputs)

    benchmark_fn = with_executor(executor, benchmark_fn)

    if not disable_validation and executor == "thunder":
        # The FusionDefinition can be pulled from thunder, but `with_executor`
        # hides thunder from us so we need to manually run things to pull the
        # FusionDefinition ourself.
        import thunder

        reference_outputs = to_be_compiled(*inputs)
        from thunder.executors.nvfuserex import nvfuserex

        tmp = thunder.jit(to_be_compiled, executors=[nvfuserex])
        tmp(*inputs)
        traces: [thunder.core.trace.TraceCtx] = thunder.last_traces(tmp)
        assert traces is not None
        trace: thunder.core.trace.TraceCtx = traces[-1]
        assert (
            "nvFusion1" not in trace.python_ctx()
        ), "thunder split the fusion, so the validation no longer fits."
        fusion: FusionDefinition = trace.python_ctx()["nvFusion0"].last_used
        fusion.validate(inputs, reference_outputs)
    run_benchmark(benchmark, benchmark_fn, inputs)


# Nanogpt has no concat operations, but because it has split operations concat
# ops appear in the backward pass. The kernel shown below appears multiple
# times in the backward pass. The '6' is arbitrary: this is the 6th fusion
# generated by the network.
@pytest.mark.parametrize("executor", ["thunder", "torchcompile"])
@pytest.mark.expr_eval
@pytest.mark.pointwise
@pytest.mark.resize
@pytest.mark.transpose
def test_cat_nanogpt_bwd_6_baseline_benchmark(
    benchmark, executor: str, disable_validation: bool
) -> None:
    def nanogpt_bwd_fusion_6_torch(bw_t3476: torch.Tensor, bw_t3475, bw_t3474):
        # bw_t3476: "cuda:0 f32[4, 6, 128, 64]"
        # bw_t3475: "cuda:0 f32[4, 6, 64, 128]"
        # bw_t3474: "cuda:0 f32[4, 6, 64, 128]"
        t0 = torch.mul(0.29730177875068026, bw_t3476)  # t0: "cuda:0 f32[4, 6, 128, 64]"
        t1 = torch.permute(t0, (0, 1, 3, 2))  # t1: "cuda:0 f32[4, 6, 64, 128]"
        # t1 = prims.transpose(t0, (0, 1, 3, 2))  # t1: "cuda:0 f32[4, 6, 64, 128]"
        t2 = torch.mul(0.29730177875068026, bw_t3475)  # t2: "cuda:0 f32[4, 6, 64, 128]"
        t3 = torch.permute(bw_t3474, (0, 2, 1, 3))  # t3: "cuda:0 f32[4, 64, 6, 128]"
        # t3 = prims.transpose(bw_t3474, (0, 2, 1, 3))  # t3: "cuda:0 f32[4, 64, 6, 128]"
        t4 = torch.reshape(t3, (4, 64, 768))  # t4: "cuda:0 f32[4, 64, 768]"
        t5 = torch.permute(t2, (0, 2, 1, 3))  # t5: "cuda:0 f32[4, 64, 6, 128]"
        # t5 = prims.transpose(t2, (0, 2, 1, 3))  # t5: "cuda:0 f32[4, 64, 6, 128]"
        t6 = torch.reshape(t5, (4, 64, 768))  # t6: "cuda:0 f32[4, 64, 768]"
        t7 = torch.permute(t1, (0, 2, 1, 3))  # t7: "cuda:0 f32[4, 64, 6, 128]"
        # t7 = prims.transpose(t1, (0, 2, 1, 3))  # t7: "cuda:0 f32[4, 64, 6, 128]"
        t8 = torch.reshape(t7, (4, 64, 768))  # t8: "cuda:0 f32[4, 64, 768]"
        t9 = torch.cat([t6, t8, t4], 2)  # t9: "cuda:0 f32[4, 64, 2304]"
        t10 = torch.reshape(t9, (256, 2304))  # t10: "cuda:0 f32[256, 2304]"
        return [t9, t10]

    inputs = [
        torch.testing.make_tensor(
            (4, 6, 128, 64), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (4, 6, 64, 128), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (4, 6, 64, 128), dtype=torch.float32, device="cuda:0"
        ),
    ]

    def benchmark_fn(inputs):
        return nanogpt_bwd_fusion_6_torch(*inputs)

    benchmark_fn = with_executor(executor, benchmark_fn)

    if not disable_validation and executor == "thunder":
        # The FusionDefinition can be pulled from thunder, but `with_executor`
        # hides thunder from us so we need to manually run things to pull the
        # FusionDefinition ourself.
        import thunder

        reference_outputs = nanogpt_bwd_fusion_6_torch(*inputs)
        from thunder.executors.nvfuserex import nvfuserex

        tmp = thunder.jit(nanogpt_bwd_fusion_6_torch, executors=[nvfuserex])
        tmp(*inputs)
        traces: [thunder.core.trace.TraceCtx] = thunder.last_traces(tmp)
        assert traces is not None
        trace: thunder.core.trace.TraceCtx = traces[-1]
        assert (
            "nvFusion1" not in trace.python_ctx()
        ), "thunder split the fusion, so the validation no longer fits."
        fusion: FusionDefinition = trace.python_ctx()["nvFusion0"].last_used
        fusion.validate(inputs, reference_outputs)
    run_benchmark(benchmark, benchmark_fn, inputs)

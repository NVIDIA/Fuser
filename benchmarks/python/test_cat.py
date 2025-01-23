# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from nvfuser import FusionDefinition, DataType
import core


# Qwen2 fusion that involves concatenation. Note that there are no Mistral-Nemo
# benchmarks in this file, because they were all equivalent to this fusion. The
# fusion below appears repeatedly in both networks.
#
# The numbers are arbitrary; this was the 11th fusion in the forward pass.
def test_cat_qwen2_fwd_11_nvfuser(benchmark):
    def nvfuser_fusion_id4(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(shape=[2048, 512], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
        T1 = fd.define_tensor(shape=[1, 2048, 512], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
        T2 = fd.define_tensor(shape=[1, 2048, 128], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 0, 1])
        T3 = fd.define_tensor(shape=[1, 2048, 128], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 0, 1])
        T4 = fd.define_tensor(shape=[1, 28, 2048, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
        T5 = fd.define_tensor(shape=[1, 4, 2048, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
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
        T41 = fd.ops.broadcast_in_dim(T29, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3])
        T42 = fd.ops.cast(T4, dtype=DataType.Float)
        T43 = fd.ops.cast(T41, dtype=DataType.Float)
        T44 = fd.ops.mul(T42, T43)
        T60 = fd.ops.slice(T4, start_indices=[0, 0, 0, 0], end_indices=[1, 28, 2048, 64], strides=[1, 1, 1, 1], manual_normalization=0)
        T76 = fd.ops.slice(T4, start_indices=[0, 0, 0, 64], end_indices=[1, 28, 2048, 128], strides=[1, 1, 1, 1], manual_normalization=0)
        T77 = fd.ops.cast(T76, dtype=DataType.Float)
        T78 = fd.ops.neg(T77)
        T79 = fd.ops.cast(T78, dtype=DataType.BFloat16)
        T80 = fd.ops.cat([T79, T60], dim=-1, manual_padding=0)
        T86 = fd.ops.broadcast_in_dim(T35, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3])
        T87 = fd.ops.cast(T80, dtype=DataType.Float)
        T88 = fd.ops.cast(T86, dtype=DataType.Float)
        T89 = fd.ops.mul(T87, T88)
        T90 = fd.ops.add(T44, T89)
        T91 = fd.ops.cast(T90, dtype=DataType.BFloat16)
        T97 = fd.ops.broadcast_in_dim(T29, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3])
        T98 = fd.ops.cast(T5, dtype=DataType.Float)
        T99 = fd.ops.cast(T97, dtype=DataType.Float)
        T100 = fd.ops.mul(T98, T99)
        T116 = fd.ops.slice(T5, start_indices=[0, 0, 0, 0], end_indices=[1, 4, 2048, 64], strides=[1, 1, 1, 1], manual_normalization=0)
        T132 = fd.ops.slice(T5, start_indices=[0, 0, 0, 64], end_indices=[1, 4, 2048, 128], strides=[1, 1, 1, 1], manual_normalization=0)
        T133 = fd.ops.cast(T132, dtype=DataType.Float)
        T134 = fd.ops.neg(T133)
        T135 = fd.ops.cast(T134, dtype=DataType.BFloat16)
        T136 = fd.ops.cat([T135, T116], dim=-1, manual_padding=0)
        T142 = fd.ops.broadcast_in_dim(T35, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3])
        T143 = fd.ops.cast(T136, dtype=DataType.Float)
        T144 = fd.ops.cast(T142, dtype=DataType.Float)
        T145 = fd.ops.mul(T143, T144)
        T146 = fd.ops.add(T100, T145)
        T147 = fd.ops.cast(T146, dtype=DataType.BFloat16)
        T154 = fd.ops.broadcast_in_dim(T147, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4])
        T161 = fd.ops.broadcast_in_dim(T154, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4])
        T167 = fd.ops.reshape(T161, new_shape=[1, 28, 2048, 128])
        T174 = fd.ops.broadcast_in_dim(T23, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4])
        T181 = fd.ops.broadcast_in_dim(T174, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4])
        T187 = fd.ops.reshape(T181, new_shape=[1, 28, 2048, 128])
        T188 = fd.ops.stride_order(T91, stride_order=[3, 2, 1, 0])
        T189 = fd.ops.stride_order(T167, stride_order=[3, 2, 1, 0])
        T190 = fd.ops.stride_order(T187, stride_order=[3, 2, 1, 0])
        fd.add_output(T23)
        fd.add_output(T147)
        fd.add_output(T188)
        fd.add_output(T189)
        fd.add_output(T190)

    with FusionDefinition() as fd:
        nvfuser_fusion_id4(fd)

    inputs = [
        torch.testing.make_tensor((2048, 512), dtype=torch.bfloat16, device='cuda:0'),
        torch.testing.make_tensor((1, 2048, 512), dtype=torch.bfloat16, device='cuda:0'),
        torch.randn(262144, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 2048, 128), (262144, 1, 2048)),
        torch.randn(262144, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 2048, 128), (262144, 1, 2048)),
        torch.randn(7340032, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 28, 2048, 128), (7340032, 128, 3584, 1)),
        torch.randn(1048576, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 4, 2048, 128), (1048576, 128, 512, 1)),
    ]

    def benchmark_fn(inputs): fd.execute(inputs)
    core.run_benchmark(benchmark, benchmark_fn, inputs)


# _tc tests are "torch.compile"-based tests. They are the equivalent set of ops
# to the _nvfuser variants but sent to dynamo/inductor instead.
def test_cat_qwen2_fwd_11_tc(benchmark):
    def to_be_compiled(t17353, t17351, cos_2, sin_2, query_states_1, key_states_1):
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
        t3 = torch.Tensor.to(t17351, torch.float32, copy=True)  # t3: "cuda:0 f32[1, 2048, 512]"
            # t3 = prims.convert_element_type(t17351, dtypes.float32)  # t3: "cuda:0 f32[1, 2048, 512]"
        t4 = torch.add(t3, t2)  # t4: "cuda:0 f32[1, 2048, 512]"
        t5 = torch.Tensor.to(t4, torch.bfloat16, copy=True)  # t5: "cuda:0 bf16[1, 2048, 512]"
            # t5 = prims.convert_element_type(t4, dtypes.bfloat16)  # t5: "cuda:0 bf16[1, 2048, 512]"
        t6 = torch.reshape(t5, (1, 2048, 4, 128))  # t6: "cuda:0 bf16[1, 2048, 4, 128]"
        t7 = torch.permute(t6, (0, 2, 1, 3))  # t7: "cuda:0 bf16[1, 4, 2048, 128]"
            # t7 = prims.transpose(t6, (0, 2, 1, 3))  # t7: "cuda:0 bf16[1, 4, 2048, 128]"
        t8 = torch.unsqueeze(cos_2, 1)  # t8: "cuda:0 bf16[1, 1, 2048, 128]"
            # t8 = prims.broadcast_in_dim(cos_2, [1, 1, 2048, 128], [0, 2, 3])  # t8: "cuda:0 bf16[1, 1, 2048, 128]"
        t9 = torch.Tensor.expand(t8, [1, 1, 2048, 128])  # t9: "cuda:0 bf16[1, 1, 2048, 128]"
            # t9 = prims.broadcast_in_dim(t8, (1, 1, 2048, 128), (0, 1, 2, 3))  # t9: "cuda:0 bf16[1, 1, 2048, 128]"

        t10 = torch.unsqueeze(sin_2, 1)  # t10: "cuda:0 bf16[1, 1, 2048, 128]"
            # t10 = prims.broadcast_in_dim(sin_2, [1, 1, 2048, 128], [0, 2, 3])  # t10: "cuda:0 bf16[1, 1, 2048, 128]"
        t11 = torch.Tensor.expand(t10, [1, 1, 2048, 128])  # t11: "cuda:0 bf16[1, 1, 2048, 128]"
            # t11 = prims.broadcast_in_dim(t10, (1, 1, 2048, 128), (0, 1, 2, 3))  # t11: "cuda:0 bf16[1, 1, 2048, 128]"
        t12 = torch.Tensor.expand(t9, (1, 28, 2048, 128))  # t12: "cuda:0 bf16[1, 28, 2048, 128]"
            # t12 = prims.broadcast_in_dim(t9, (1, 28, 2048, 128), (0, 1, 2, 3))  # t12: "cuda:0 bf16[1, 28, 2048, 128]"
        t13 = torch.Tensor.to(query_states_1, torch.float32, copy=True)  # t13: "cuda:0 f32[1, 28, 2048, 128]"
            # t13 = prims.convert_element_type(query_states_1, dtypes.float32)  # t13: "cuda:0 f32[1, 28, 2048, 128]"
        t14 = torch.Tensor.to(t12, torch.float32, copy=True)  # t14: "cuda:0 f32[1, 28, 2048, 128]"
            # t14 = prims.convert_element_type(t12, dtypes.float32)  # t14: "cuda:0 f32[1, 28, 2048, 128]"
        t15 = torch.mul(t13, t14)  # t15: "cuda:0 f32[1, 28, 2048, 128]"
        t16 = query_states_1[0:1, 0:28, 0:2048, 0:64]
        t17 = query_states_1[0:1, 0:28, 0:2048, 64:128]
        #t16 = torch.slice(query_states_1, [0, 0, 0, 0], [1, 28, 2048, 64], [1, 1, 1, 1])  # t16: "cuda:0 bf16[1, 28, 2048, 64]"
        #t17 = torch.slice(query_states_1, [0, 0, 0, 64], [1, 28, 2048, 128], [1, 1, 1, 1])  # t17: "cuda:0 bf16[1, 28, 2048, 64]"
        t18 = torch.Tensor.to(t17, torch.float32, copy=True)  # t18: "cuda:0 f32[1, 28, 2048, 64]"
            # t18 = prims.convert_element_type(t17, dtypes.float32)  # t18: "cuda:0 f32[1, 28, 2048, 64]"
        t19 = torch.neg(t18)  # t19: "cuda:0 f32[1, 28, 2048, 64]"
        t20 = torch.Tensor.to(t19, torch.bfloat16, copy=True)  # t20: "cuda:0 bf16[1, 28, 2048, 64]"
            # t20 = prims.convert_element_type(t19, dtypes.bfloat16)  # t20: "cuda:0 bf16[1, 28, 2048, 64]"
        t21 = torch.cat([t20, t16], -1)  # t21: "cuda:0 bf16[1, 28, 2048, 128]"
        t22 = torch.Tensor.expand(t11, (1, 28, 2048, 128))  # t22: "cuda:0 bf16[1, 28, 2048, 128]"
            # t22 = prims.broadcast_in_dim(t11, (1, 28, 2048, 128), (0, 1, 2, 3))  # t22: "cuda:0 bf16[1, 28, 2048, 128]"
        t23 = torch.Tensor.to(t21, torch.float32, copy=True)  # t23: "cuda:0 f32[1, 28, 2048, 128]"
            # t23 = prims.convert_element_type(t21, dtypes.float32)  # t23: "cuda:0 f32[1, 28, 2048, 128]"
        t24 = torch.Tensor.to(t22, torch.float32, copy=True)  # t24: "cuda:0 f32[1, 28, 2048, 128]"
            # t24 = prims.convert_element_type(t22, dtypes.float32)  # t24: "cuda:0 f32[1, 28, 2048, 128]"
        t25 = torch.mul(t23, t24)  # t25: "cuda:0 f32[1, 28, 2048, 128]"
        t26 = torch.add(t15, t25)  # t26: "cuda:0 f32[1, 28, 2048, 128]"
        t27 = torch.Tensor.to(t26, torch.bfloat16, copy=True)  # t27: "cuda:0 bf16[1, 28, 2048, 128]"
            # t27 = prims.convert_element_type(t26, dtypes.bfloat16)  # t27: "cuda:0 bf16[1, 28, 2048, 128]"
        t28 = torch.Tensor.expand(t9, (1, 4, 2048, 128))  # t28: "cuda:0 bf16[1, 4, 2048, 128]"
            # t28 = prims.broadcast_in_dim(t9, (1, 4, 2048, 128), (0, 1, 2, 3))  # t28: "cuda:0 bf16[1, 4, 2048, 128]"
        t29 = torch.Tensor.to(key_states_1, torch.float32, copy=True)  # t29: "cuda:0 f32[1, 4, 2048, 128]"
            # t29 = prims.convert_element_type(key_states_1, dtypes.float32)  # t29: "cuda:0 f32[1, 4, 2048, 128]"
        t30 = torch.Tensor.to(t28, torch.float32, copy=True)  # t30: "cuda:0 f32[1, 4, 2048, 128]"
            # t30 = prims.convert_element_type(t28, dtypes.float32)  # t30: "cuda:0 f32[1, 4, 2048, 128]"
        t31 = torch.mul(t29, t30)  # t31: "cuda:0 f32[1, 4, 2048, 128]"
        #t32 = torch.slice(key_states_1, [0, 0, 0, 0], [1, 4, 2048, 64], [1, 1, 1, 1])  # t32: "cuda:0 bf16[1, 4, 2048, 64]"
        #t33 = torch.slice(key_states_1, [0, 0, 0, 64], [1, 4, 2048, 128], [1, 1, 1, 1])  # t33: "cuda:0 bf16[1, 4, 2048, 64]"
        t32 = key_states_1[0:1, 0:4, 0:2048, 0:64]
        t33 = key_states_1[0:1, 0:4, 0:2048, 64:128]
        t34 = torch.Tensor.to(t33, torch.float32, copy=True)  # t34: "cuda:0 f32[1, 4, 2048, 64]"
            # t34 = prims.convert_element_type(t33, dtypes.float32)  # t34: "cuda:0 f32[1, 4, 2048, 64]"
        t35 = torch.neg(t34)  # t35: "cuda:0 f32[1, 4, 2048, 64]"
        t36 = torch.Tensor.to(t35, torch.bfloat16, copy=True)  # t36: "cuda:0 bf16[1, 4, 2048, 64]"
            # t36 = prims.convert_element_type(t35, dtypes.bfloat16)  # t36: "cuda:0 bf16[1, 4, 2048, 64]"
        t37 = torch.cat([t36, t32], -1)  # t37: "cuda:0 bf16[1, 4, 2048, 128]"
        t38 = torch.Tensor.expand(t11, (1, 4, 2048, 128))  # t38: "cuda:0 bf16[1, 4, 2048, 128]"
            # t38 = prims.broadcast_in_dim(t11, (1, 4, 2048, 128), (0, 1, 2, 3))  # t38: "cuda:0 bf16[1, 4, 2048, 128]"
        t39 = torch.Tensor.to(t37, torch.float32, copy=True)  # t39: "cuda:0 f32[1, 4, 2048, 128]"
            # t39 = prims.convert_element_type(t37, dtypes.float32)  # t39: "cuda:0 f32[1, 4, 2048, 128]"
        t40 = torch.Tensor.to(t38, torch.float32, copy=True)  # t40: "cuda:0 f32[1, 4, 2048, 128]"
            # t40 = prims.convert_element_type(t38, dtypes.float32)  # t40: "cuda:0 f32[1, 4, 2048, 128]"
        t41 = torch.mul(t39, t40)  # t41: "cuda:0 f32[1, 4, 2048, 128]"
        t42 = torch.add(t31, t41)  # t42: "cuda:0 f32[1, 4, 2048, 128]"
        t43 = torch.Tensor.to(t42, torch.bfloat16, copy=True)  # t43: "cuda:0 bf16[1, 4, 2048, 128]"
            # t43 = prims.convert_element_type(t42, dtypes.bfloat16)  # t43: "cuda:0 bf16[1, 4, 2048, 128]"
        t44 = torch.unsqueeze(t43, 2)  # t44: "cuda:0 bf16[1, 4, 1, 2048, 128]"
            # t44 = prims.broadcast_in_dim(t43, [1, 4, 1, 2048, 128], [0, 1, 3, 4])  # t44: "cuda:0 bf16[1, 4, 1, 2048, 128]"
        t45 = torch.Tensor.expand(t44, [1, 4, 1, 2048, 128])  # t45: "cuda:0 bf16[1, 4, 1, 2048, 128]"
            # t45 = prims.broadcast_in_dim(t44, (1, 4, 1, 2048, 128), (0, 1, 2, 3, 4))  # t45: "cuda:0 bf16[1, 4, 1, 2048, 128]"
        t46 = torch.Tensor.expand(t45, (1, 4, 7, 2048, 128))  # t46: "cuda:0 bf16[1, 4, 7, 2048, 128]"
            # t46 = prims.broadcast_in_dim(t45, (1, 4, 7, 2048, 128), (0, 1, 2, 3, 4))  # t46: "cuda:0 bf16[1, 4, 7, 2048, 128]"
        t47 = torch.reshape(t46, (1, 28, 2048, 128))  # t47: "cuda:0 bf16[1, 28, 2048, 128]"
        t48 = torch.unsqueeze(t7, 2)  # t48: "cuda:0 bf16[1, 4, 1, 2048, 128]"
            # t48 = prims.broadcast_in_dim(t7, [1, 4, 1, 2048, 128], [0, 1, 3, 4])  # t48: "cuda:0 bf16[1, 4, 1, 2048, 128]"
        t49 = torch.Tensor.expand(t48, [1, 4, 1, 2048, 128])  # t49: "cuda:0 bf16[1, 4, 1, 2048, 128]"
            # t49 = prims.broadcast_in_dim(t48, (1, 4, 1, 2048, 128), (0, 1, 2, 3, 4))  # t49: "cuda:0 bf16[1, 4, 1, 2048, 128]"
        t50 = torch.Tensor.expand(t49, (1, 4, 7, 2048, 128))  # t50: "cuda:0 bf16[1, 4, 7, 2048, 128]"
            # t50 = prims.broadcast_in_dim(t49, (1, 4, 7, 2048, 128), (0, 1, 2, 3, 4))  # t50: "cuda:0 bf16[1, 4, 7, 2048, 128]"
        t51 = torch.reshape(t50, (1, 28, 2048, 128))  # t51: "cuda:0 bf16[1, 28, 2048, 128]"
        #t52 = torch_stride_order_prim_impl(t27, (3, 2, 1, 0))  # t52: "cuda:0 bf16[1, 28, 2048, 128]"
        #t53 = torch_stride_order_prim_impl(t47, (3, 2, 1, 0))  # t53: "cuda:0 bf16[1, 28, 2048, 128]"
        #t54 = torch_stride_order_prim_impl(t51, (3, 2, 1, 0))  # t54: "cuda:0 bf16[1, 28, 2048, 128]"
        t52 = torch.as_strided(t27, (1,28,2048,128), (7340032, 7340032, 262144, 128))
        t53 = torch.as_strided(t47, (1,28,2048,128), (7340032, 7340032, 262144, 128))
        t54 = torch.as_strided(t51, (1,28,2048,128), (7340032, 7340032, 262144, 128))
        return [t7, t43, t52, t53, t54]

    inputs = [
        torch.randn(size=(2048, 512), dtype=torch.bfloat16, layout=torch.strided, device="cuda", requires_grad=False),
        torch.randn(size=(1, 2048, 512), dtype=torch.bfloat16, layout=torch.strided, device="cuda", requires_grad=False),
        torch.randn(size=(1, 2048, 128), dtype=torch.bfloat16, layout=torch.strided, device="cuda", requires_grad=False),
        torch.randn(size=(1, 2048, 128), dtype=torch.bfloat16, layout=torch.strided, device="cuda", requires_grad=False),
        torch.randn(size=(1, 28, 2048, 128), dtype=torch.bfloat16, layout=torch.strided, device="cuda", requires_grad=False),
        torch.randn(size=(1, 4, 2048, 128), dtype=torch.bfloat16, layout=torch.strided, device="cuda", requires_grad=False),
    ]

    func = torch.compile(to_be_compiled)
    def benchmark_fn(inputs): func(*inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)


def test_cat_phi3_1_nvfuser(benchmark):
    def nvfuser_fusion_id3(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(shape=[1, 48, 2048], contiguity=[None, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
        T1 = fd.ops.permute(T0, dims=[0, 2, 1])
        T2 = fd.ops.cat([T1, T1], dim=-1, manual_padding=0)
        T3 = fd.ops.cos(T2)
        T4 = fd.ops.sin(T2)
        S5 = fd.define_scalar(1.19024, dtype=DataType.Double)
        T6 = fd.ops.mul(T3, S5)
        S7 = fd.define_scalar(1.19024, dtype=DataType.Double)
        T8 = fd.ops.mul(T4, S7)
        T9 = fd.ops.cast(T6, dtype=DataType.BFloat16)
        T10 = fd.ops.cast(T8, dtype=DataType.BFloat16)
        fd.add_output(T9)
        fd.add_output(T10)

    with FusionDefinition() as fd:
        nvfuser_fusion_id3(fd)

    inputs = [
        torch.testing.make_tensor((1, 48, 2048), dtype=torch.float32, device='cuda:0'),
    ]

    def benchmark_fn(inputs): fd.execute(inputs)
    core.run_benchmark(benchmark, benchmark_fn, inputs)


def test_cat_phi3_1_tc(benchmark):
    def to_be_compiled(t14):
      # t14: "cuda:0 f32[1, 48, 2048]"
      t0 = torch.permute(t14, (0, 2, 1))  # t0: "cuda:0 f32[1, 2048, 48]"
          # t0 = prims.transpose(t14, (0, 2, 1))  # t0: "cuda:0 f32[1, 2048, 48]"
      t1 = torch.cat([t0, t0], -1)  # t1: "cuda:0 f32[1, 2048, 96]"
      t2 = torch.cos(t1)  # t2: "cuda:0 f32[1, 2048, 96]"
      t3 = torch.sin(t1)  # t3: "cuda:0 f32[1, 2048, 96]"
      t4 = torch.mul(t2, 1.1902380714238083)  # t4: "cuda:0 f32[1, 2048, 96]"
      t5 = torch.mul(t3, 1.1902380714238083)  # t5: "cuda:0 f32[1, 2048, 96]"
      t6 = torch.Tensor.to(t4, torch.bfloat16, copy=True)  # t6: "cuda:0 bf16[1, 2048, 96]"
          # t6 = prims.convert_element_type(t4, dtypes.bfloat16)  # t6: "cuda:0 bf16[1, 2048, 96]"
      t7 = torch.Tensor.to(t5, torch.bfloat16, copy=True)  # t7: "cuda:0 bf16[1, 2048, 96]"
          # t7 = prims.convert_element_type(t5, dtypes.bfloat16)  # t7: "cuda:0 bf16[1, 2048, 96]"
      return [t6, t7]

    inputs = [
      torch.randn(size=(1, 48, 2048), dtype=torch.float32, layout=torch.strided, device="cuda", requires_grad=False),
    ]

    func = torch.compile(to_be_compiled)
    def benchmark_fn(inputs): func(*inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)


@pytest.mark.skip("issue 3740")
def test_cat_qwen2_v2(benchmark):
    def qwen2_cat_fusion_2(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(shape=[2048, 512], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
        S1 = fd.define_scalar(None, dtype=DataType.Int)
        S2 = fd.define_scalar(None, dtype=DataType.Int)
        T3 = fd.define_tensor(shape=[1, 4, 2048, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
        T4 = fd.define_tensor(shape=[1, 2048, 128], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 0, 1])
        T5 = fd.define_tensor(shape=[1, 2048, 128], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 0, 1])
        T6 = fd.define_tensor(shape=[1, 28, 2048, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
        T7 = fd.define_tensor(shape=[1, 2048, 512], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
        T12 = fd.ops.reshape(T0, new_shape=[1, 2048, 512])
        T13 = fd.ops.cast(T12, dtype=DataType.Float)
        S14 = fd.define_scalar(0.00000, dtype=DataType.Double)
        S15 = fd.define_scalar(1.00000, dtype=DataType.Double)
        S16 = fd.define_scalar(1, dtype=DataType.Int)
        S17 = fd.define_scalar(2048, dtype=DataType.Int)
        S18 = fd.define_scalar(512, dtype=DataType.Int)
        T20 = fd.ops.uniform(S14, S15, shape=[S16, S17, S18], rng_seed=S2, rng_offset=S1, dtype=DataType.BFloat16)
        S21 = fd.define_scalar(4.00000, dtype=DataType.Double)
        T22 = fd.ops.mul(T13, S21)
        S23 = fd.define_scalar(0.900000, dtype=DataType.Double)
        T24 = fd.ops.lt(T20, S23)
        T25 = fd.ops.cast(T24, dtype=DataType.Float)
        T41 = fd.ops.slice(T3, start_indices=[0, 0, 0, 64], end_indices=[1, 4, 2048, 128], strides=[1, 1, 1, 1], manual_normalization=0)
        T42 = fd.ops.mul(T22, T25)
        T43 = fd.ops.cast(T41, dtype=DataType.Float)
        T44 = fd.ops.neg(T43)
        T50 = fd.ops.broadcast_in_dim(T4, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3])
        T66 = fd.ops.slice(T3, start_indices=[0, 0, 0, 0], end_indices=[1, 4, 2048, 64], strides=[1, 1, 1, 1], manual_normalization=0)
        T67 = fd.ops.cast(T44, dtype=DataType.BFloat16)
        T73 = fd.ops.broadcast_in_dim(T5, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3])
        T89 = fd.ops.slice(T6, start_indices=[0, 0, 0, 64], end_indices=[1, 28, 2048, 128], strides=[1, 1, 1, 1], manual_normalization=0)
        S90 = fd.define_scalar(1.11111, dtype=DataType.Double)
        T91 = fd.ops.mul(T42, S90)
        T97 = fd.ops.broadcast_in_dim(T50, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3])
        T98 = fd.ops.cat([T67, T66], dim=-1, manual_padding=0)
        T104 = fd.ops.broadcast_in_dim(T73, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3])
        T105 = fd.ops.cast(T89, dtype=DataType.Float)
        T106 = fd.ops.cast(T97, dtype=DataType.Float)
        T107 = fd.ops.cast(T98, dtype=DataType.Float)
        T108 = fd.ops.cast(T104, dtype=DataType.Float)
        T109 = fd.ops.cast(T3, dtype=DataType.Float)
        T110 = fd.ops.neg(T105)
        T111 = fd.ops.cast(T7, dtype=DataType.Float)
        T112 = fd.ops.mul(T107, T106)
        T113 = fd.ops.mul(T109, T108)
        T129 = fd.ops.slice(T6, start_indices=[0, 0, 0, 0], end_indices=[1, 28, 2048, 64], strides=[1, 1, 1, 1], manual_normalization=0)
        T130 = fd.ops.cast(T110, dtype=DataType.BFloat16)
        T131 = fd.ops.add(T111, T91)
        T137 = fd.ops.broadcast_in_dim(T50, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3])
        T138 = fd.ops.cat([T130, T129], dim=-1, manual_padding=0)
        T144 = fd.ops.broadcast_in_dim(T73, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3])
        T145 = fd.ops.cast(T131, dtype=DataType.BFloat16)
        T146 = fd.ops.cast(T137, dtype=DataType.Float)
        T147 = fd.ops.cast(T138, dtype=DataType.Float)
        T148 = fd.ops.cast(T144, dtype=DataType.Float)
        T149 = fd.ops.cast(T6, dtype=DataType.Float)
        T155 = fd.ops.reshape(T145, new_shape=[1, 2048, 4, 128])
        T156 = fd.ops.add(T113, T112)
        T157 = fd.ops.mul(T147, T146)
        T158 = fd.ops.mul(T149, T148)
        T159 = fd.ops.permute(T155, dims=[0, 2, 1, 3])
        T160 = fd.ops.cast(T156, dtype=DataType.BFloat16)
        T167 = fd.ops.broadcast_in_dim(T159, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4])
        T174 = fd.ops.broadcast_in_dim(T160, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4])
        T181 = fd.ops.broadcast_in_dim(T167, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4])
        T188 = fd.ops.broadcast_in_dim(T174, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4])
        T189 = fd.ops.add(T158, T157)
        T195 = fd.ops.reshape(T181, new_shape=[1, 28, 2048, 128])
        T201 = fd.ops.reshape(T188, new_shape=[1, 28, 2048, 128])
        T202 = fd.ops.cast(T189, dtype=DataType.BFloat16)
        T203 = fd.ops.stride_order(T195, stride_order=[3, 2, 1, 0])
        T204 = fd.ops.stride_order(T201, stride_order=[3, 2, 1, 0])
        T205 = fd.ops.stride_order(T202, stride_order=[3, 2, 1, 0])
        fd.add_output(T159)
        fd.add_output(T160)
        fd.add_output(T203)
        fd.add_output(T204)
        fd.add_output(T205)

    inputs = [
        torch.testing.make_tensor((2048, 512), dtype=torch.bfloat16, device='cuda:0'),
        25546,
        1400552702872758,
        torch.randn(1048576, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 4, 2048, 128), (1048576, 128, 512, 1)),
        torch.randn(262144, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 2048, 128), (262144, 1, 2048)),
        torch.randn(262144, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 2048, 128), (262144, 1, 2048)),
        torch.randn(7340032, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 28, 2048, 128), (7340032, 128, 3584, 1)),
        torch.testing.make_tensor((1, 2048, 512), dtype=torch.bfloat16, device='cuda:0'),
    ]

    with FusionDefinition() as fd:
        qwen2_cat_fusion_2(fd)

    def benchmark_fn(inputs):
        fd.execute(inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)

# Nanogpt has no concat operations, but because it has split operations concat
# ops appear in the backward pass. The kernel shown below appears 6 times in
# Thunder's test. The '6' is arbitrary; this is the 6th fusion generated by the
# network.
def test_cat_nanogpt_bwd_6_nvfuser(benchmark):
    def nanogpt_bwd_fusion_6(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(shape=[4, 6, 128, 64], contiguity=[True, True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
        T1 = fd.define_tensor(shape=[4, 6, 64, 128], contiguity=[True, True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
        T2 = fd.define_tensor(shape=[4, 6, 64, 128], contiguity=[True, True, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[3, 2, 1, 0])
        S3 = fd.define_scalar(0.297302, dtype=DataType.Double)
        T4 = fd.ops.mul(S3, T0)
        T5 = fd.ops.permute(T4, dims=[0, 1, 3, 2])
        S6 = fd.define_scalar(0.297302, dtype=DataType.Double)
        T7 = fd.ops.mul(S6, T1)
        T8 = fd.ops.permute(T2, dims=[0, 2, 1, 3])
        T13 = fd.ops.reshape(T8, new_shape=[4, 64, 768])
        T14 = fd.ops.permute(T7, dims=[0, 2, 1, 3])
        T19 = fd.ops.reshape(T14, new_shape=[4, 64, 768])
        T20 = fd.ops.permute(T5, dims=[0, 2, 1, 3])
        T25 = fd.ops.reshape(T20, new_shape=[4, 64, 768])
        T26 = fd.ops.cat([T19, T25, T13], dim=2, manual_padding=0)
        T30 = fd.ops.reshape(T26, new_shape=[256, 2304])
        fd.add_output(T26)
        fd.add_output(T30)

    inputs = [
        torch.testing.make_tensor((4, 6, 128, 64), dtype=torch.float32, device='cuda:0'),
        torch.testing.make_tensor((4, 6, 64, 128), dtype=torch.float32, device='cuda:0'),
        torch.testing.make_tensor((4, 6, 64, 128), dtype=torch.float32, device='cuda:0'),
    ]

    with FusionDefinition() as fd:
        nanogpt_bwd_fusion_6(fd)

    def benchmark_fn(inputs): fd.execute(inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)

def test_cat_nanogpt_bwd_6_tc(benchmark):
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
        torch.testing.make_tensor((4, 6, 128, 64), dtype=torch.float32, device='cuda:0'),
        torch.testing.make_tensor((4, 6, 64, 128), dtype=torch.float32, device='cuda:0'),
        torch.testing.make_tensor((4, 6, 64, 128), dtype=torch.float32, device='cuda:0'),
    ]

    func = torch.compile(nanogpt_bwd_fusion_6_torch)
    def benchmark_fn(inputs): func(*inputs)

    core.run_benchmark(benchmark, benchmark_fn, inputs)

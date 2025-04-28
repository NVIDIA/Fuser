# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import pytest
import nvfuser_next
from nvfuser_next import FusionDefinition, DataType


def test_fusion_definition():
    fd = FusionDefinition()
    tv0 = (
        nvfuser_next.TensorViewBuilder()
        .num_dims(3)
        .shape([2, 4, 8])
        .contiguity(True)
        .build()
    )
    tv1 = (
        nvfuser_next.TensorViewBuilder()
        .num_dims(3)
        .shape([2, 4, 8])
        .contiguity(True)
        .build()
    )

    fd.add_input(tv0)
    fd.add_input(tv1)
    tv2 = fd.ops.add(tv0, tv1)
    fd.add_output(tv2)

    # Test TensorView string representation
    assert str(tv0) == r"T0_g_float[iS0{2}, iS1{4}, iS2{8}]"
    assert str(tv1) == r"T1_g_float[iS3{2}, iS4{4}, iS5{8}]"
    assert str(tv2) == r"T2_g_float[iS6{2}, iS7{4}, iS8{8}]"

    # Test TensorDomain string representation
    assert str(tv0.domain()) == r"[iS0{2}, iS1{4}, iS2{8}]"
    assert str(tv1.domain()) == r"[iS3{2}, iS4{4}, iS5{8}]"
    assert str(tv2.domain()) == r"[iS6{2}, iS7{4}, iS8{8}]"

    # Test IterDomain string representation
    assert str(tv0.axis(0)) == r"iS0{2}"
    assert str(tv1.axis(0)) == r"iS3{2}"
    assert str(tv2.axis(0)) == r"iS6{2}"

    # Test axis extents
    assert str(tv0.axis(0).extent()) == r"2"
    assert str(tv1.axis(0).extent()) == r"2"
    assert str(tv2.axis(0).extent()) == r"2"


def test_fusion_execution():
    fd = FusionDefinition()
    tv0 = (
        nvfuser_next.TensorViewBuilder()
        .num_dims(3)
        .shape([2, 4, 8])
        .contiguity(True)
        .build()
    )
    tv1 = (
        nvfuser_next.TensorViewBuilder()
        .num_dims(3)
        .shape([2, 4, 8])
        .contiguity(True)
        .build()
    )

    fd.add_input(tv0)
    fd.add_input(tv1)
    tv2 = fd.ops.add(tv0, tv1)
    fd.add_output(tv2)

    # Test fusion math representation
    prescheduled_fusion_definition = """Inputs:
  T0_g_float[iS0{2}, iS1{4}, iS2{8}]
  T1_g_float[iS3{2}, iS4{4}, iS5{8}]
Outputs:
  T2_g_float[iS6{2}, iS7{4}, iS8{8}]

%kernel_math {
T2_g_float[iS6{2}, iS7{4}, iS8{8}]
   = T0_g_float[iS0{2}, iS1{4}, iS2{8}]
   + T1_g_float[iS3{2}, iS4{4}, iS5{8}];
} // %kernel_math \n\n"""
    assert fd.fusion.print_math() == prescheduled_fusion_definition

    # Test fusion kernel representation
    prescheduled_kernel = """// Codegen generated code
__global__ void CUDAGeneratedKernel(Tensor<float, 3, 3> T0, Tensor<float, 3, 3> T1, Tensor<float, 3, 3> T2) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i0 = 0LL; i0 < 2LL; ++i0) {
    nvfuser_index_t i1;
    i1 = 32LL * i0;
    #pragma unroll
    for(nvfuser_index_t i2 = 0LL; i2 < 4LL; ++i2) {
      nvfuser_index_t i3;
      i3 = i1 + (8LL * i2);
      #pragma unroll
      for(nvfuser_index_t i4 = 0LL; i4 < 8LL; ++i4) {
        nvfuser_index_t i5;
        i5 = i3 + (i4 + nvfuser_zero);
        T2[i5]
          = T0[i5]
          + T1[i5];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
}\n"""
    assert fd.fusion.print_kernel() == prescheduled_kernel

    # Test fusion transforms
    prescheduled_transforms = """T0_g_float[iS0{2}, iS1{4}, iS2{8}]
 logical domain : (iS0{2}, iS1{4}, iS2{8})
 contiguity: t t t
 loop domain : (iS0{2}, iS1{4}, iS2{8})
T1_g_float[iS3{2}, iS4{4}, iS5{8}]
 logical domain : (iS3{2}, iS4{4}, iS5{8})
 contiguity: t t t
 loop domain : (iS3{2}, iS4{4}, iS5{8})
T2_g_float[iS6{2}, iS7{4}, iS8{8}]
 logical domain : (iS6{2}, iS7{4}, iS8{8})
 contiguity: t t t
 loop domain : (iS6{2}, iS7{4}, iS8{8})
"""
    assert fd.fusion.print_transforms() == prescheduled_transforms

    fec = nvfuser_next.FusionExecutorCache(fd.fusion)
    inputs = [
        torch.randn(2, 4, 8, device="cuda"),
        torch.randn(2, 4, 8, device="cuda"),
    ]

    # Test fusion execution
    results = fec.execute(inputs)
    assert torch.allclose(results[0], inputs[0] + inputs[1])

    # Test fusion string representation
    assert fec.print_fusion() == prescheduled_fusion_definition

    # Test compilation status
    assert fec.is_compiled(inputs)

    # Test scheduled IR representation
    scheduled_ir = """Inputs:
  T0_g_float[iS58{1}, iS59{1}, iS57{128}]
  T1_g_float[iS46{1}, iS47{1}, iS45{128}]
Outputs:
  T2_g_float[iblockIdx.x28{1}, iUS29{1}, ithreadIdx.x27{128}] ca_pos( 2 ) produce_pos( 3 )

%kernel {
T3_l_float[iblockIdx.x52{1}, iUS53{1}, ithreadIdx.x51{128}] ca_pos( 2 )
   = Set( T0_g_float[iS58{1}, iS59{1}, iS57{128}], cache_op=Streaming )
T4_l_float[iblockIdx.x40{1}, iUS41{1}, ithreadIdx.x39{128}] ca_pos( 2 )
   = Set( T1_g_float[iS46{1}, iS47{1}, iS45{128}], cache_op=Streaming )
T5_l_float[iblockIdx.x34{1}, iUS35{1}, ithreadIdx.x33{128}] ca_pos( 3 ) produce_pos( 2 )
   = T3_l_float[iblockIdx.x52{1}, iUS53{1}, ithreadIdx.x51{128}] ca_pos( 2 )
   + T4_l_float[iblockIdx.x40{1}, iUS41{1}, ithreadIdx.x39{128}] ca_pos( 2 );
T2_g_float[iblockIdx.x28{1}, iUS29{1}, ithreadIdx.x27{128}] ca_pos( 2 ) produce_pos( 3 )
   = Set( T5_l_float[iblockIdx.x34{1}, iUS35{1}, ithreadIdx.x33{128}] ca_pos( 3 ) produce_pos( 2 ), cache_op=Streaming )
} // %kernel\n"""
    assert fec.get_scheduled_ir(inputs) == scheduled_ir
    assert fec.get_most_recent_scheduled_ir() == scheduled_ir

    # Test CUDA kernel representation
    cuda_kernel = """// Codegen generated code
__global__ void nvfuser_pointwise_f0_c1_r0_g0(Tensor<float, 3, 3> T0, Tensor<float, 3, 3> T1, Tensor<float, 3, 3> T2) {
  nvfuser_index_t i0;
  i0 = ((nvfuser_index_t)threadIdx.x) + (128 * ((nvfuser_index_t)blockIdx.x));
  if ((i0 < 64)) {
    Array<float, 1, 1> T4;
    T4[0] = 0;
    T4[0]
       = T1[i0];
    Array<float, 1, 1> T3;
    T3[0] = 0;
    T3[0]
       = T0[i0];
    Array<float, 1, 1> T5;
    T5[0]
      = T3[0]
      + T4[0];
    T2[i0]
       = T5[0];
  }
}\n"""
    assert fec.get_cuda_kernel(inputs) == cuda_kernel


def test_define_tensor():
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        tv1 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        tv2 = fd.ops.add(tv0, tv1)
        fd.add_output(tv2)

    # Test fusion execution with dynamic shapes
    inputs = [
        torch.randn(2, 4, 8, device="cuda"),
        torch.randn(2, 4, 8, device="cuda"),
    ]
    outputs = fd.execute(inputs)
    assert torch.allclose(outputs[0], inputs[0] + inputs[1])

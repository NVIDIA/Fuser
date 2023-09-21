// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <inlining.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class LoopRotationTest : public NVFuserTest {};

TEST_F(LoopRotationTest, RotateInner) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineMost();
  scheduler_utils::rotateLoop(tv4, -1, {tv1, tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i0 = 0; i0 < T0.logical_size[0]; ++i0) {
    nvfuser_index_t i1;
    i1 = T0.alloc_stride[0] * i0;
    nvfuser_index_t i2;
    i2 = 3 * i0;
    float T1[1];
    float T2[1];
    T1[0] = 0;
    T1[0]
       = T0[i1];
    T2[0]
       = T1[0];
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
      nvfuser_index_t i4;
      i4 = (1 + i3) + nvfuser_zero;
      float T3[1];
      T3[0]
         = T2[0];
      T4[(i2 + (i3 + nvfuser_zero))]
         = T3[0];
      T1[0] = 0;
      if ((i4 < 3)) {
        T1[0]
           = T0[(i1 + (T0.alloc_stride[1] * i4))];
      }
      T2[0]
         = T1[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";

  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, RotateOuter) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  float T1[3];
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i0 = 0; i0 < 3; ++i0) {
    T1[i0] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i0 = 0; i0 < 3; ++i0) {
    T1[i0]
       = T0[(T0.alloc_stride[1] * (i0 + nvfuser_zero))];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i1 = 0; i1 < 3; ++i1) {
    T2[i1]
       = T1[i1];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i2 = 0; i2 < T0.logical_size[0]; ++i2) {
    nvfuser_index_t i3;
    i3 = 3 * i2;
    nvfuser_index_t i4;
    i4 = T0.alloc_stride[0] + (T0.alloc_stride[0] * i2);
    bool b5;
    b5 = (1 + i2) < T0.logical_size[0];
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
      T3[i6]
         = T2[i6];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i7 = 0; i7 < 3; ++i7) {
      T4[(i3 + (i7 + nvfuser_zero))]
         = T3[i7];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i0 = 0; i0 < 3; ++i0) {
      T1[i0] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i0 = 0; i0 < 3; ++i0) {
      if (b5) {
        T1[i0]
           = T0[(i4 + (T0.alloc_stride[1] * (i0 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i1 = 0; i1 < 3; ++i1) {
      T2[i1]
         = T1[i1];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, NonDivisibleSplit) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, -1});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  for (auto tv : {tv0, tv1, tv2, tv3, tv4}) {
    tv->merge(0);
    tv->split(0, 5);
  }
  inlineAllAt(tv4, 1);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  nvfuser_index_t i0;
  i0 = T0.logical_size[0] * T0.logical_size[1];
  nvfuser_index_t i1;
  i1 = ceilDiv(i0, 5);
  float T1[5];
  float T2[5];
  #pragma unroll
  for(nvfuser_index_t i2 = 0; i2 < 5; ++i2) {
    T1[i2] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i2 = 0; i2 < 5; ++i2) {
    nvfuser_index_t i3;
    i3 = i2 + nvfuser_zero;
    if ((i3 < i0)) {
      T1[i2]
         = T0[((T0.alloc_stride[0] * (i3 / T0.logical_size[1])) + (T0.alloc_stride[1] * (i3 % T0.logical_size[1])))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i4 = 0; i4 < 5; ++i4) {
    T2[i4]
       = T1[i4];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i5 = 0; i5 < i1; ++i5) {
    nvfuser_index_t i6;
    i6 = 5 * i5;
    nvfuser_index_t i7;
    i7 = 5 + i6;
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i8 = 0; i8 < 5; ++i8) {
      T3[i8]
         = T2[i8];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i9 = 0; i9 < 5; ++i9) {
      nvfuser_index_t i10;
      i10 = i6 + (i9 + nvfuser_zero);
      if ((i10 < i0)) {
        T4[i10]
           = T3[i9];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i2 = 0; i2 < 5; ++i2) {
      T1[i2] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i2 = 0; i2 < 5; ++i2) {
      nvfuser_index_t i11;
      i11 = i7 + (i2 + nvfuser_zero);
      if ((i11 < i0)) {
        T1[i2]
           = T0[((T0.alloc_stride[0] * (i11 / T0.logical_size[1])) + (T0.alloc_stride[1] * (i11 % T0.logical_size[1])))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i4 = 0; i4 < 5; ++i4) {
      T2[i4]
         = T1[i4];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, DoubleBuffered) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  tv1->circularBuffer(5);
  scheduler_utils::rotateLoop(tv4, 0, {tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  nvfuser_index_t i0;
  i0 = 4 * T0.alloc_stride[0];
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i1 = 0; i1 < 4; ++i1) {
    nvfuser_index_t i2;
    i2 = 3 * i1;
    nvfuser_index_t i3;
    i3 = T0.alloc_stride[0] * i1;
    bool b4;
    b4 = (i1 + nvfuser_zero) < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i5 = 0; i5 < 3; ++i5) {
      T1[(i2 + i5)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i5 = 0; i5 < 3; ++i5) {
      if (b4) {
        T1[(i2 + i5)]
           = T0[(i3 + (T0.alloc_stride[1] * (i5 + nvfuser_zero)))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
    T2[i6]
       = T1[i6];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i7 = 0; i7 < T0.logical_size[0]; ++i7) {
    nvfuser_index_t i8;
    i8 = 4 + i7;
    nvfuser_index_t i9;
    i9 = 3 * (i8 % 5);
    nvfuser_index_t i10;
    i10 = i0 + (T0.alloc_stride[0] * i7);
    nvfuser_index_t i11;
    i11 = 3 * i7;
    nvfuser_index_t i12;
    i12 = 3 * ((1 + i7) % 5);
    bool b13;
    b13 = i8 < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i5 = 0; i5 < 3; ++i5) {
      T1[(i9 + i5)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i5 = 0; i5 < 3; ++i5) {
      if (b13) {
        T1[(i9 + i5)]
           = T0[(i10 + (T0.alloc_stride[1] * (i5 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i14 = 0; i14 < 3; ++i14) {
      T3[i14]
         = T2[i14];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i15 = 0; i15 < 3; ++i15) {
      T4[(i11 + (i15 + nvfuser_zero))]
         = T3[i15];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
      T2[i6]
         = T1[(i12 + i6)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, SelectDoubleBufferLoad) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  tv1->circularBuffer(5);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  nvfuser_index_t i0;
  i0 = 4 * T0.alloc_stride[0];
  nvfuser_index_t i1;
  i1 = 5 * T0.alloc_stride[0];
  bool b2;
  b2 = 4 < T0.logical_size[0];
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
    T1[i3] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
    T1[i3]
       = T0[(T0.alloc_stride[1] * (i3 + nvfuser_zero))];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i4 = 0; i4 < 4; ++i4) {
    nvfuser_index_t i5;
    i5 = 3 + (3 * i4);
    nvfuser_index_t i6;
    i6 = T0.alloc_stride[0] + (T0.alloc_stride[0] * i4);
    bool b7;
    b7 = ((1 + i4) + nvfuser_zero) < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
      T1[(i5 + i3)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
      if (b7) {
        T1[(i5 + i3)]
           = T0[(i6 + (T0.alloc_stride[1] * (i3 + nvfuser_zero)))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
    T1[(12 + i3)] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
    if (b2) {
      T1[(12 + i3)]
         = T0[(i0 + (T0.alloc_stride[1] * (i3 + nvfuser_zero)))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i8 = 0; i8 < 3; ++i8) {
    T2[i8]
       = T1[i8];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i9 = 0; i9 < T0.logical_size[0]; ++i9) {
    nvfuser_index_t i10;
    i10 = 3 * i9;
    nvfuser_index_t i11;
    i11 = 3 * (i9 % 5);
    nvfuser_index_t i12;
    i12 = i1 + (T0.alloc_stride[0] * i9);
    nvfuser_index_t i13;
    i13 = 3 * ((1 + i9) % 5);
    bool b14;
    b14 = (5 + i9) < T0.logical_size[0];
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i15 = 0; i15 < 3; ++i15) {
      T3[i15]
         = T2[i15];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i16 = 0; i16 < 3; ++i16) {
      T4[(i10 + (i16 + nvfuser_zero))]
         = T3[i16];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
      T1[(i11 + i3)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
      if (b14) {
        T1[(i11 + i3)]
           = T0[(i12 + (T0.alloc_stride[1] * (i3 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i8 = 0; i8 < 3; ++i8) {
      T2[i8]
         = T1[(i13 + i8)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

// This is a case similar to matmul, where we have
// tv4 = set(tv0) // cp.async for matmul
// tv1 = set(tv4) // ld.matrix for matmul
// and both are double buffered
TEST_F(LoopRotationTest, MultipleDoubleBuffer) {
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  auto tv4 = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv4->setMemoryType(MemoryType::Shared);

  inlineAllAt(tv3, 1);
  inlineSelectedAt({tv1, tv2, tv3}, tv3, 2);

  tv4->circularBuffer(5);
  tv1->doubleBuffer();
  scheduler_utils::rotateLoop(tv3, 0, {tv1});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T3) {
  alignas(16) extern __shared__ char array[];
  const unsigned smem_offset = 0;
  NVFUSER_DEFINE_MAGIC_ZERO;
  float* ptr0;
  ptr0 = T0.data + (4 * T0.alloc_stride[0]);
  float* T4 = reinterpret_cast<float*>(array + smem_offset + 0);
  #pragma unroll
  for(nvfuser_index_t i1 = 0; i1 < 4; ++i1) {
    float* ptr2;
    ptr2 = T0.data + (T0.alloc_stride[0] * i1);
    unsigned i3;
    i3 = toSmem(T4) + (12 * i1);
    bool b4;
    b4 = (i1 + nvfuser_zero) < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i5 = 0; i5 < 3; ++i5) {
      Ampere::cpAsyncCa<float, 1>((i3 + (4 * i5)), (ptr2 + (T0.alloc_stride[1] * (i5 + nvfuser_zero))), b4);
    }
    Ampere::cpAsyncCommit();
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  Ampere::cpAsyncPartialBarrier<3>();
  float T1[2];
  T1[0]
     = T4[0];
  #pragma unroll 1
  for(nvfuser_index_t i6 = 0; i6 < T0.logical_size[0]; ++i6) {
    float* ptr7;
    ptr7 = ptr0 + (T0.alloc_stride[0] * i6);
    nvfuser_index_t i8;
    i8 = 4 + i6;
    unsigned i9;
    i9 = toSmem(T4) + (12 * (i8 % 5));
    nvfuser_index_t i10;
    i10 = 1 + (3 * (i6 % 5));
    nvfuser_index_t i11;
    i11 = 3 * i6;
    bool b12;
    b12 = i8 < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i5 = 0; i5 < 3; ++i5) {
      Ampere::cpAsyncCa<float, 1>((i9 + (4 * i5)), (ptr7 + (T0.alloc_stride[1] * (i5 + nvfuser_zero))), b12);
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    Ampere::cpAsyncCommit();
    #pragma unroll
    for(nvfuser_index_t i13 = 0; i13 < 2; ++i13) {
      T1[((1 + i13) % 2)]
         = T4[(i10 + i13)];
      float T2[1];
      T2[0]
         = T1[(i13 % 2)];
      T3[(i11 + (i13 + nvfuser_zero))]
         = T2[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    float T2[1];
    T2[0]
       = T1[0];
    T3[(2 + i11)]
       = T2[0];
    NVFUSER_UPDATE_MAGIC_ZERO;
    Ampere::cpAsyncPartialBarrier<3>();
    T1[0]
       = T4[(3 * ((1 + i6) % 5))];
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}
} // namespace nvfuser

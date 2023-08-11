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
  Array<nvfuser_index_t, 2, 1> a0;
  a0 = (T0).alloc_stride;
  int64_t i1;
  i1 = a0[0];
  int64_t i2;
  i2 = a0[1];
  #pragma unroll 1
  for(nvfuser_index_t i3 = 0; i3 < T0.logical_size[0]; ++i3) {
    int64_t i4;
    i4 = i1 * i3;
    int64_t i5;
    i5 = 3 * i3;
    float T1[1];
    float T2[1];
    T1[0] = 0;
    T1[0]
       = T0[i4];
    T2[0]
       = T1[0];
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
      int64_t i7;
      i7 = (1 + i6) + nvfuser_zero;
      float T3[1];
      T3[0]
         = T2[0];
      T4[(i5 + (i6 + nvfuser_zero))]
         = T3[0];
      T1[0] = 0;
      if ((i7 < 3)) {
        T1[0]
           = T0[(i4 + (i2 * i7))];
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
  Array<nvfuser_index_t, 2, 1> a0;
  a0 = (T0).alloc_stride;
  int64_t i1;
  i1 = a0[1];
  int64_t i2;
  i2 = a0[0];
  float T1[3];
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
    T1[i3] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
    T1[i3]
       = T0[(i1 * (i3 + nvfuser_zero))];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i4 = 0; i4 < 3; ++i4) {
    T2[i4]
       = T1[i4];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i5 = 0; i5 < T0.logical_size[0]; ++i5) {
    int64_t i6;
    i6 = 3 * i5;
    int64_t i7;
    i7 = i2 + (i2 * i5);
    bool b8;
    b8 = (1 + i5) < T0.logical_size[0];
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i9 = 0; i9 < 3; ++i9) {
      T3[i9]
         = T2[i9];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i10 = 0; i10 < 3; ++i10) {
      T4[(i6 + (i10 + nvfuser_zero))]
         = T3[i10];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
      T1[i3] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i3 = 0; i3 < 3; ++i3) {
      if (b8) {
        T1[i3]
           = T0[(i7 + (i1 * (i3 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i4 = 0; i4 < 3; ++i4) {
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
  Array<nvfuser_index_t, 2, 1> a0;
  a0 = (T0).alloc_stride;
  int64_t i1;
  i1 = a0[0];
  int64_t i2;
  i2 = a0[1];
  int64_t i3;
  i3 = T0.logical_size[0] * T0.logical_size[1];
  float T1[5];
  float T2[5];
  #pragma unroll
  for(nvfuser_index_t i4 = 0; i4 < 5; ++i4) {
    T1[i4] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i4 = 0; i4 < 5; ++i4) {
    int64_t i5;
    i5 = i4 + nvfuser_zero;
    if ((i5 < i3)) {
      T1[i4]
         = T0[((i1 * (i5 / T0.logical_size[1])) + (i2 * (i5 % T0.logical_size[1])))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i6 = 0; i6 < 5; ++i6) {
    T2[i6]
       = T1[i6];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i7 = 0; i7 < (ceilDiv((T0.logical_size[0] * T0.logical_size[1]), 5)); ++i7) {
    int64_t i8;
    i8 = 5 * i7;
    int64_t i9;
    i9 = 5 + i8;
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i10 = 0; i10 < 5; ++i10) {
      T3[i10]
         = T2[i10];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i11 = 0; i11 < 5; ++i11) {
      int64_t i12;
      i12 = i8 + (i11 + nvfuser_zero);
      if ((i12 < i3)) {
        T4[i12]
           = T3[i11];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i4 = 0; i4 < 5; ++i4) {
      T1[i4] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i4 = 0; i4 < 5; ++i4) {
      int64_t i13;
      i13 = i9 + (i4 + nvfuser_zero);
      if ((i13 < i3)) {
        T1[i4]
           = T0[((i1 * (i13 / T0.logical_size[1])) + (i2 * (i13 % T0.logical_size[1])))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i6 = 0; i6 < 5; ++i6) {
      T2[i6]
         = T1[i6];
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
  Array<nvfuser_index_t, 2, 1> a0;
  a0 = (T0).alloc_stride;
  int64_t i1;
  i1 = a0[0];
  int64_t i2;
  i2 = a0[1];
  int64_t i3;
  i3 = 4 * i1;
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i4 = 0; i4 < 4; ++i4) {
    int64_t i5;
    i5 = 3 * i4;
    int64_t i6;
    i6 = i1 * i4;
    bool b7;
    b7 = (i4 + nvfuser_zero) < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i8 = 0; i8 < 3; ++i8) {
      T1[(i5 + i8)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i8 = 0; i8 < 3; ++i8) {
      if (b7) {
        T1[(i5 + i8)]
           = T0[(i6 + (i2 * (i8 + nvfuser_zero)))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i9 = 0; i9 < 3; ++i9) {
    T2[i9]
       = T1[i9];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i10 = 0; i10 < T0.logical_size[0]; ++i10) {
    int64_t i11;
    i11 = 4 + i10;
    int64_t i12;
    i12 = 3 * (i11 % 5);
    int64_t i13;
    i13 = i3 + (i1 * i10);
    int64_t i14;
    i14 = 3 * i10;
    int64_t i15;
    i15 = 3 * ((1 + i10) % 5);
    bool b16;
    b16 = i11 < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i8 = 0; i8 < 3; ++i8) {
      T1[(i12 + i8)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i8 = 0; i8 < 3; ++i8) {
      if (b16) {
        T1[(i12 + i8)]
           = T0[(i13 + (i2 * (i8 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      T3[i17]
         = T2[i17];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i18 = 0; i18 < 3; ++i18) {
      T4[(i14 + (i18 + nvfuser_zero))]
         = T3[i18];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i9 = 0; i9 < 3; ++i9) {
      T2[i9]
         = T1[(i15 + i9)];
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
  Array<nvfuser_index_t, 2, 1> a0;
  a0 = (T0).alloc_stride;
  int64_t i1;
  i1 = a0[1];
  int64_t i2;
  i2 = a0[0];
  int64_t i3;
  i3 = 4 * i2;
  int64_t i4;
  i4 = 5 * i2;
  bool b5;
  b5 = 4 < T0.logical_size[0];
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
    T1[i6] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
    T1[i6]
       = T0[(i1 * (i6 + nvfuser_zero))];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i7 = 0; i7 < 4; ++i7) {
    int64_t i8;
    i8 = 3 + (3 * i7);
    int64_t i9;
    i9 = i2 + (i2 * i7);
    bool b10;
    b10 = ((1 + i7) + nvfuser_zero) < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
      T1[(i8 + i6)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
      if (b10) {
        T1[(i8 + i6)]
           = T0[(i9 + (i1 * (i6 + nvfuser_zero)))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
    T1[(12 + i6)] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
    if (b5) {
      T1[(12 + i6)]
         = T0[(i3 + (i1 * (i6 + nvfuser_zero)))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i11 = 0; i11 < 3; ++i11) {
    T2[i11]
       = T1[i11];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i12 = 0; i12 < T0.logical_size[0]; ++i12) {
    int64_t i13;
    i13 = 3 * i12;
    int64_t i14;
    i14 = 3 * (i12 % 5);
    int64_t i15;
    i15 = i4 + (i2 * i12);
    int64_t i16;
    i16 = 3 * ((1 + i12) % 5);
    bool b17;
    b17 = (5 + i12) < T0.logical_size[0];
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i18 = 0; i18 < 3; ++i18) {
      T3[i18]
         = T2[i18];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i19 = 0; i19 < 3; ++i19) {
      T4[(i13 + (i19 + nvfuser_zero))]
         = T3[i19];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
      T1[(i14 + i6)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i6 = 0; i6 < 3; ++i6) {
      if (b17) {
        T1[(i14 + i6)]
           = T0[(i15 + (i1 * (i6 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i11 = 0; i11 < 3; ++i11) {
      T2[i11]
         = T1[(i16 + i11)];
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

  auto tv4 = tv0->cacheAfter(LoadStoreOpType::CpAsyncCa);
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
  Tensor<float, 2, 2> s0;
  s0.data = T0.data;
  s0.logical_size = T0.logical_size;
  s0.alloc_stride = T0.alloc_stride;
  float* ptr1;
  ptr1 = s0.data;
  Array<nvfuser_index_t, 2, 1> a2;
  a2 = s0.alloc_stride;
  int64_t i3;
  i3 = a2[0];
  int64_t i4;
  i4 = a2[1];
  float* ptr5;
  ptr5 = ptr1 + (4 * i3);
  float* T4 = reinterpret_cast<float*>(array + smem_offset + 0);
  #pragma unroll
  for(nvfuser_index_t i6 = 0; i6 < 4; ++i6) {
    float* ptr7;
    ptr7 = ptr1 + (i3 * i6);
    unsigned i8;
    i8 = (toSmem((T4))) + (12 * i6);
    bool b9;
    b9 = (i6 + nvfuser_zero) < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i10 = 0; i10 < 3; ++i10) {
      Ampere::cpAsyncCa<float, 1>((i8 + (4 * i10)), (ptr7 + (i4 * (i10 + nvfuser_zero))), b9);
    }
    Ampere::cpAsyncCommit();
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  Ampere::cpAsyncPartialBarrier<3>();
  float T1[2];
  T1[0]
     = T4[0];
  #pragma unroll 1
  for(nvfuser_index_t i11 = 0; i11 < T0.logical_size[0]; ++i11) {
    float* ptr12;
    ptr12 = ptr5 + (i3 * i11);
    int64_t i13;
    i13 = 4 + i11;
    unsigned i14;
    i14 = (toSmem((T4))) + (12 * (i13 % 5));
    int64_t i15;
    i15 = 1 + (3 * (i11 % 5));
    int64_t i16;
    i16 = 3 * i11;
    bool b17;
    b17 = i13 < T0.logical_size[0];
    #pragma unroll
    for(nvfuser_index_t i10 = 0; i10 < 3; ++i10) {
      Ampere::cpAsyncCa<float, 1>((i14 + (4 * i10)), (ptr12 + (i4 * (i10 + nvfuser_zero))), b17);
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    Ampere::cpAsyncCommit();
    #pragma unroll
    for(nvfuser_index_t i18 = 0; i18 < 2; ++i18) {
      T1[((1 + i18) % 2)]
         = T4[(i15 + i18)];
      float T2[1];
      T2[0]
         = T1[(i18 % 2)];
      T3[(i16 + (i18 + nvfuser_zero))]
         = T2[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    float T2[1];
    T2[0]
       = T1[0];
    T3[(2 + i16)]
       = T2[0];
    NVFUSER_UPDATE_MAGIC_ZERO;
    Ampere::cpAsyncPartialBarrier<3>();
    T1[0]
       = T4[(3 * ((1 + i11) % 5))];
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

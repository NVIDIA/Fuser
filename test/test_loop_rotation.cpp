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

class LoopRotationTest : public NVFuserTest {
 private:
  // Please see note [Limitation of boundary assert]
  EnableOutOfBoundAssert guard;
};

TEST_F(LoopRotationTest, RotateInner_CUDA) {
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i21 = 0; i21 < T0.size[0]; ++i21) {
    int64_t i286;
    i286 = T0.stride[0] * i21;
    int64_t i1290;
    i1290 = 3 * i21;
    float T1[1];
    float T2[1];
    T1[0] = 0;
    T1[0]
       = T0[i286];
    T2[0]
       = T1[0];
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      int64_t i1929;
      i1929 = (1 + i22) + nvfuser_zero;
      float T3[1];
      T3[0]
         = T2[0];
      T4[(i1290 + (i22 + nvfuser_zero))]
         = T3[0];
      T1[0] = 0;
      if ((i1929 < 3)) {
        T1[0]
           = T0[(i286 + (T0.stride[1] * i1929))];
      }
      T2[0]
         = T1[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";

  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, RotateOuter_CUDA) {
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  bool b3402;
  b3402 = 0 < T0.size[0];
  float T1[3];
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    if (b3402) {
      T1[i21]
         = T0[(T0.stride[1] * (i21 + nvfuser_zero))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
    T2[i22]
       = T1[i22];
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i24 = 0; i24 < T0.size[0]; ++i24) {
    int64_t i1494;
    i1494 = 3 * i24;
    int64_t i2157;
    i2157 = T0.stride[0] + (T0.stride[0] * i24);
    bool b4473;
    b4473 = (1 + i24) < T0.size[0];
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i25 = 0; i25 < 3; ++i25) {
      T4[(i1494 + (i25 + nvfuser_zero))]
         = T3[i25];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[i21] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b4473) {
        T1[i21]
           = T0[(i2157 + (T0.stride[1] * (i21 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[i22];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, NonDivisibleSplit_CUDA) {
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i6157;
  i6157 = T0.size[0] * T0.size[1];
  float T1[5];
  float T2[5];
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    T1[i36] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    int64_t i266;
    i266 = i36 + nvfuser_zero;
    if ((i266 < i6157)) {
      T1[i36]
         = T0[((T0.stride[0] * (i266 / T0.size[1])) + (T0.stride[1] * (i266 % T0.size[1])))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i37 = 0; i37 < 5; ++i37) {
    T2[i37]
       = T1[i37];
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i39 = 0; i39 < (ceilDiv((T0.size[0] * T0.size[1]), 5)); ++i39) {
    int64_t i2020;
    i2020 = 5 * i39;
    int64_t i4590;
    i4590 = 5 + i2020;
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i38 = 0; i38 < 5; ++i38) {
      T3[i38]
         = T2[i38];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i40 = 0; i40 < 5; ++i40) {
      int64_t i2021;
      i2021 = i2020 + (i40 + nvfuser_zero);
      if ((i2021 < i6157)) {
        T4[i2021]
           = T3[i40];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
      T1[i36] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
      int64_t i4671;
      i4671 = i4590 + (i36 + nvfuser_zero);
      if ((i4671 < i6157)) {
        T1[i36]
           = T0[((T0.stride[0] * (i4671 / T0.size[1])) + (T0.stride[1] * (i4671 % T0.size[1])))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i37 = 0; i37 < 5; ++i37) {
      T2[i37]
         = T1[i37];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, DoubleBuffered_CUDA) {
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i2539;
  i2539 = T0.stride[0] * 4;
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i24 = 0; i24 < 4; ++i24) {
    int64_t i278;
    i278 = 3 * i24;
    int64_t i637;
    i637 = T0.stride[0] * i24;
    bool b5903;
    b5903 = (i24 + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i278 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b5903) {
        T1[(i278 + i21)]
           = T0[(i637 + (T0.stride[1] * (i21 + nvfuser_zero)))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
    T2[i22]
       = T1[i22];
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i25 = 0; i25 < T0.size[0]; ++i25) {
    int64_t i1879;
    i1879 = 4 + i25;
    int64_t i1881;
    i1881 = 3 * (i1879 % 5);
    int64_t i2541;
    i2541 = i2539 + (T0.stride[0] * i25);
    int64_t i4528;
    i4528 = 3 * i25;
    int64_t i5045;
    i5045 = 3 * ((1 + i25) % 5);
    bool b6884;
    b6884 = i1879 < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i1881 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b6884) {
        T1[(i1881 + i21)]
           = T0[(i2541 + (T0.stride[1] * (i21 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i27 = 0; i27 < 3; ++i27) {
      T4[(i4528 + (i27 + nvfuser_zero))]
         = T3[i27];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i5045 + i22)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, SelectDoubleBufferLoad_CUDA) {
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i2471;
  i2471 = 4 * T0.stride[0];
  int64_t i4858;
  i4858 = T0.stride[0] * 5;
  bool b7080;
  b7080 = 0 < T0.size[0];
  bool b8125;
  b8125 = 4 < T0.size[0];
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    if (b7080) {
      T1[i21]
         = T0[(T0.stride[1] * (i21 + nvfuser_zero))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i24 = 0; i24 < 4; ++i24) {
    int64_t i717;
    i717 = 3 + (3 * i24);
    int64_t i1214;
    i1214 = T0.stride[0] + (T0.stride[0] * i24);
    bool b7685;
    b7685 = ((1 + i24) + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i717 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b7685) {
        T1[(i717 + i21)]
           = T0[(i1214 + (T0.stride[1] * (i21 + nvfuser_zero)))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[(12 + i21)] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    if (b8125) {
      T1[(12 + i21)]
         = T0[(i2471 + (T0.stride[1] * (i21 + nvfuser_zero)))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
    T2[i22]
       = T1[i22];
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i25 = 0; i25 < T0.size[0]; ++i25) {
    int64_t i3817;
    i3817 = 3 * i25;
    int64_t i4354;
    i4354 = 3 * (i25 % 5);
    int64_t i4860;
    i4860 = i4858 + (T0.stride[0] * i25);
    int64_t i6294;
    i6294 = 3 * ((1 + i25) % 5);
    bool b9230;
    b9230 = (5 + i25) < T0.size[0];
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i27 = 0; i27 < 3; ++i27) {
      T4[(i3817 + (i27 + nvfuser_zero))]
         = T3[i27];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i4354 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b9230) {
        T1[(i4354 + i21)]
           = T0[(i4860 + (T0.stride[1] * (i21 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i6294 + i22)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
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
TEST_F(LoopRotationTest, MultipleDoubleBuffer_CUDA) {
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T3) {
  alignas(16) extern __shared__ char array[];
  unsigned smem_offset = 0;
  NVFUSER_DEFINE_MAGIC_ZERO
  float* ptr171;
  ptr171 = T0.data;
  float* ptr2237;
  ptr2237 = (T0.stride[0] * 4) + ptr171;
  smem_offset = alignBufferSize(smem_offset, 16);
  float* T4 = reinterpret_cast<float*>(array + smem_offset);
  smem_offset += (15 * sizeof(float));
  #pragma unroll
  for(nvfuser_index_t i18 = 0; i18 < 4; ++i18) {
    float* ptr310;
    ptr310 = ptr171 + (T0.stride[0] * i18);
    unsigned i988;
    i988 = (toSmem(T4)) + (12 * i18);
    bool b6852;
    b6852 = (i18 + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      Ampere::cpAsyncCa<float, 1>((i988 + (4 * i17)),(ptr310 + (T0.stride[1] * (i17 + nvfuser_zero))),b6852);
    }
    Ampere::cpAsyncCommit();
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  Ampere::cpAsyncPartialBarrier<3>();
  float T1[2];
  T1[0]
     = T4[0];
  #pragma unroll 1
  for(nvfuser_index_t i19 = 0; i19 < T0.size[0]; ++i19) {
    float* ptr2292;
    ptr2292 = ptr2237 + (T0.stride[0] * i19);
    int64_t i2765;
    i2765 = 4 + i19;
    unsigned i2768;
    i2768 = (toSmem(T4)) + (12 * (i2765 % 5));
    int64_t i3667;
    i3667 = 1 + (3 * (i19 % 5));
    int64_t i5370;
    i5370 = 3 * i19;
    bool b7647;
    b7647 = i2765 < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      Ampere::cpAsyncCa<float, 1>((i2768 + (4 * i17)),(ptr2292 + (T0.stride[1] * (i17 + nvfuser_zero))),b7647);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    Ampere::cpAsyncCommit();
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 2; ++i22) {
      T1[((1 + i22) % 2)]
         = T4[(i3667 + i22)];
      float T2[1];
      T2[0]
         = T1[(i22 % 2)];
      T3[(i5370 + (i22 + nvfuser_zero))]
         = T2[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T2[1];
    T2[0]
       = T1[0];
    T3[(2 + i5370)]
       = T2[0];
    NVFUSER_UPDATE_MAGIC_ZERO
    Ampere::cpAsyncPartialBarrier<3>();
    T1[0]
       = T4[(3 * ((1 + i19) % 5))];
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}
} // namespace nvfuser

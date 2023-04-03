// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <inlining.h>
#include <ops/arith.h>
#include <scheduler/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

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
    int64_t i160;
    i160 = T0.stride[0] * i21;
    int64_t i516;
    i516 = 3 * i21;
    float T1[1];
    float T2[1];
    T1[0] = 0;
    T1[0]
       = T0[i160];
    T2[0]
       = T1[0];
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      int64_t i651;
      i651 = (1 + i22) + nvfuser_zero;
      float T3[1];
      T3[0]
         = T2[0];
      T4[(i516 + (i22 + nvfuser_zero))]
         = T3[0];
      T1[0] = 0;
      if ((i651 < 3)) {
        T1[0]
           = T0[(i160 + (T0.stride[1] * i651))];
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
  bool b918;
  b918 = 0 < T0.size[0];
  float T1[3];
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    if (b918) {
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
    int64_t i540;
    i540 = 3 * i24;
    int64_t i681;
    i681 = T0.stride[0] + (T0.stride[0] * i24);
    bool b1323;
    b1323 = (1 + i24) < T0.size[0];
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
      T4[(i540 + (i25 + nvfuser_zero))]
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
      if (b1323) {
        T1[i21]
           = T0[(i681 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
  int64_t i1529;
  i1529 = T0.size[0] * T0.size[1];
  float T1[5];
  float T2[5];
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    T1[i36] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    int64_t i154;
    i154 = i36 + nvfuser_zero;
    if ((i154 < i1529)) {
      T1[i36]
         = T0[((T0.stride[0] * (i154 / T0.size[1])) + (T0.stride[1] * (i154 % T0.size[1])))];
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
    int64_t i636;
    i636 = 5 * i39;
    int64_t i1230;
    i1230 = 5 + i636;
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
      int64_t i637;
      i637 = i636 + (i40 + nvfuser_zero);
      if ((i637 < i1529)) {
        T4[i637]
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
      int64_t i1231;
      i1231 = i1230 + (i36 + nvfuser_zero);
      if ((i1231 < i1529)) {
        T1[i36]
           = T0[((T0.stride[0] * (i1231 / T0.size[1])) + (T0.stride[1] * (i1231 % T0.size[1])))];
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
  int64_t i577;
  i577 = T0.stride[0] * 4;
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i24 = 0; i24 < 4; ++i24) {
    int64_t i152;
    i152 = 3 * i24;
    int64_t i223;
    i223 = T0.stride[0] * i24;
    bool b1187;
    b1187 = (i24 + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i152 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b1187) {
        T1[(i152 + i21)]
           = T0[(i223 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
    int64_t i493;
    i493 = 4 + i25;
    int64_t i495;
    i495 = 3 * (i493 % 5);
    int64_t i579;
    i579 = i577 + (T0.stride[0] * i25);
    int64_t i946;
    i946 = 3 * i25;
    int64_t i1067;
    i1067 = 3 * ((1 + i25) % 5);
    bool b1448;
    b1448 = i493 < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i495 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b1448) {
        T1[(i495 + i21)]
           = T0[(i579 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
      T4[(i946 + (i27 + nvfuser_zero))]
         = T3[i27];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i1067 + i22)];
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
  int64_t i545;
  i545 = 4 * T0.stride[0];
  int64_t i1150;
  i1150 = T0.stride[0] * 5;
  bool b1536;
  b1536 = 0 < T0.size[0];
  bool b1843;
  b1843 = 4 < T0.size[0];
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    if (b1536) {
      T1[i21]
         = T0[(T0.stride[1] * (i21 + nvfuser_zero))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i24 = 0; i24 < 4; ++i24) {
    int64_t i285;
    i285 = 3 + (3 * i24);
    int64_t i368;
    i368 = T0.stride[0] + (T0.stride[0] * i24);
    bool b1781;
    b1781 = ((1 + i24) + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i285 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b1781) {
        T1[(i285 + i21)]
           = T0[(i368 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
    if (b1843) {
      T1[(12 + i21)]
         = T0[(i545 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
    int64_t i919;
    i919 = 3 * i25;
    int64_t i1060;
    i1060 = 3 * (i25 % 5);
    int64_t i1152;
    i1152 = i1150 + (T0.stride[0] * i25);
    int64_t i1416;
    i1416 = 3 * ((1 + i25) % 5);
    bool b2264;
    b2264 = (5 + i25) < T0.size[0];
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i27 = 0; i27 < 3; ++i27) {
      T4[(i919 + (i27 + nvfuser_zero))]
         = T3[i27];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i1060 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b2264) {
        T1[(i1060 + i21)]
           = T0[(i1152 + (T0.stride[1] * (i21 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i1416 + i22)];
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
  float* ptr98;
  ptr98 = T0.data;
  float* ptr382;
  ptr382 = (T0.stride[0] * 4) + ptr98;
  smem_offset = alignBufferSize(smem_offset, 16);
  float* T4 = reinterpret_cast<float*>(array + smem_offset);
  smem_offset += (15 * sizeof(float));
  #pragma unroll
  for(nvfuser_index_t i18 = 0; i18 < 4; ++i18) {
    float* ptr165;
    ptr165 = ptr98 + (T0.stride[0] * i18);
    unsigned i249;
    i249 = (toSmem(T4)) + (12 * i18);
    bool b1334;
    b1334 = (i18 + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      Ampere::cpAsyncCa<float, 1>((i249 + (4 * i17)),(ptr165 + (T0.stride[1] * (i17 + nvfuser_zero))),b1334);
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
    float* ptr383;
    ptr383 = ptr382 + (T0.stride[0] * i19);
    int64_t i568;
    i568 = 4 + i19;
    unsigned i571;
    i571 = (toSmem(T4)) + (12 * (i568 % 5));
    int64_t i657;
    i657 = 1 + (3 * (i19 % 5));
    int64_t i1029;
    i1029 = 3 * i19;
    bool b1517;
    b1517 = i568 < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      Ampere::cpAsyncCa<float, 1>((i571 + (4 * i17)),(ptr383 + (T0.stride[1] * (i17 + nvfuser_zero))),b1517);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    Ampere::cpAsyncCommit();
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 2; ++i22) {
      T1[((1 + i22) % 2)]
         = T4[(i657 + i22)];
      float T2[1];
      T2[0]
         = T1[(i22 % 2)];
      T3[(i1029 + (i22 + nvfuser_zero))]
         = T2[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T2[1];
    T2[0]
       = T1[0];
    T3[(2 + i1029)]
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

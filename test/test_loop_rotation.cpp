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

  // TODO: b71 is trivially true, we should eliminate it
  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i21 = 0; i21 < T0.size[0]; ++i21) {
    int64_t i30;
    i30 = T0.stride[0] * i21;
    int64_t i44;
    i44 = 3 * i21;
    bool b71;
    b71 = i21 < T0.size[0];
    float T1[1];
    float T2[1];
    T1[0] = 0;
    if (b71) {
      T1[0]
         = T0[i30];
    }
    T2[0]
       = T1[0];
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      int64_t i37;
      i37 = i22 + nvfuser_zero;
      int64_t i61;
      i61 = (1 + i22) + nvfuser_zero;
      float T3[1];
      T3[0]
         = T2[0];
      if ((b71 && (i37 < 3))) {
        T4[(i44 + i37)]
           = T3[0];
      }
      T1[0] = 0;
      if ((b71 && (i61 < 3))) {
        T1[0]
           = T0[(i30 + (T0.stride[1] * i61))];
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
  bool b80;
  b80 = 0 < T0.size[0];
  float T1[3];
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    int64_t i29;
    i29 = i21 + nvfuser_zero;
    if ((b80 && (i29 < 3))) {
      T1[i21]
         = T0[(T0.stride[1] * i29)];
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
    int64_t i48;
    i48 = 3 * i24;
    int64_t i69;
    i69 = T0.stride[0] + (T0.stride[0] * i24);
    bool b104;
    b104 = i24 < T0.size[0];
    bool b153;
    b153 = (1 + i24) < T0.size[0];
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
      int64_t i41;
      i41 = i25 + nvfuser_zero;
      if ((b104 && (i41 < 3))) {
        T4[(i48 + i41)]
           = T3[i25];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[i21] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i52;
      i52 = i21 + nvfuser_zero;
      if ((b153 && (i52 < 3))) {
        T1[i21]
           = T0[(i69 + (T0.stride[1] * i52))];
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
  int64_t i279;
  i279 = T0.size[0] * T0.size[1];
  float T1[5];
  float T2[5];
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    T1[i36] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    int64_t i44;
    i44 = i36 + nvfuser_zero;
    if ((i44 < i279)) {
      T1[i36]
         = T0[((T0.stride[0] * (i44 / T0.size[1])) + (T0.stride[1] * (i44 % T0.size[1])))];
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
    int64_t i98;
    i98 = 5 * i39;
    int64_t i246;
    i246 = 5 + i98;
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
      int64_t i99;
      i99 = i98 + (i40 + nvfuser_zero);
      if ((i99 < i279)) {
        T4[i99]
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
      int64_t i247;
      i247 = i246 + (i36 + nvfuser_zero);
      if ((i247 < i279)) {
        T1[i36]
           = T0[((T0.stride[0] * (i247 / T0.size[1])) + (T0.stride[1] * (i247 % T0.size[1])))];
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
  int64_t i111;
  i111 = T0.stride[0] * 4;
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i24 = 0; i24 < 4; ++i24) {
    int64_t i40;
    i40 = 3 * i24;
    int64_t i51;
    i51 = T0.stride[0] * i24;
    bool b180;
    b180 = (i24 + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i43 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i42;
      i42 = i21 + nvfuser_zero;
      if ((b180 && (i42 < 3))) {
        T1[(i40 + i21)]
           = T0[(i51 + (T0.stride[1] * i42))];
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
    int64_t i89;
    i89 = 4 + i25;
    int64_t i91;
    i91 = 3 * (i89 % 5);
    int64_t i113;
    i113 = i111 + (T0.stride[0] * i25);
    int64_t i150;
    i150 = 3 * i25;
    int64_t i173;
    i173 = 3 * ((1 + i25) % 5);
    bool b231;
    b231 = i89 < T0.size[0];
    bool b238;
    b238 = i25 < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i94 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i93;
      i93 = i21 + nvfuser_zero;
      if ((b231 && (i93 < 3))) {
        T1[(i91 + i21)]
           = T0[(i113 + (T0.stride[1] * i93))];
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
      int64_t i143;
      i143 = i27 + nvfuser_zero;
      if ((b238 && (i143 < 3))) {
        T4[(i150 + i143)]
           = T3[i27];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i176 + i22)];
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
  int64_t i119;
  i119 = 4 * T0.stride[0];
  int64_t i220;
  i220 = T0.stride[0] * 5;
  bool b296;
  b296 = 0 < T0.size[0];
  bool b360;
  b360 = 4 < T0.size[0];
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    int64_t i35;
    i35 = i21 + nvfuser_zero;
    if ((b296 && (i35 < 3))) {
      T1[i21]
         = T0[(T0.stride[1] * i38)];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i24 = 0; i24 < 4; ++i24) {
    int64_t i57;
    i57 = 3 + (3 * i24);
    int64_t i78;
    i78 = T0.stride[0] + (T0.stride[0] * i24);
    bool b349;
    b349 = ((1 + i24) + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i60 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i61;
      i61 = i21 + nvfuser_zero;
      if ((b349 && (i61 < 3))) {
        T1[(i57 + i21)]
           = T0[(i78 + (T0.stride[1] * i61))];
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
    int64_t i109;
    i109 = i21 + nvfuser_zero;
    if ((b360 && (i109 < 3))) {
      T1[(12 + i21)]
         = T0[(i122 + (T0.stride[1] * i112))];
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
    int64_t i151;
    i151 = 3 * i25;
    int64_t i192;
    i192 = 3 * (i25 % 5);
    int64_t i222;
    i222 = i220 + (T0.stride[0] * i25);
    int64_t i288;
    i288 = 3 * ((1 + i25) % 5);
    bool b384;
    b384 = i25 < T0.size[0];
    bool b449;
    b449 = (5 + i25) < T0.size[0];
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i27 = 0; i27 < 3; ++i27) {
      int64_t i144;
      i144 = i27 + nvfuser_zero;
      if ((b384 && (i144 < 3))) {
        T4[(i151 + i144)]
           = T3[i27];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i195 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i196;
      i196 = i21 + nvfuser_zero;
      if ((b449 && (i196 < 3))) {
        T1[(i192 + i21)]
           = T0[(i222 + (T0.stride[1] * i196))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i291 + i22)];
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
  float* ptr44;
  ptr44 = T0.data;
  float* ptr114;
  ptr114 = ptr44 + (T0.stride[0] * 4);
  smem_offset = alignBufferSize(smem_offset, 16);
  float* T4 = reinterpret_cast<float*>(array + smem_offset);
  smem_offset += (15 * sizeof(float));
  #pragma unroll
  for(nvfuser_index_t i18 = 0; i18 < 4; ++i18) {
    float* ptr51;
    ptr51 = ptr44 + (T0.stride[0] * i18);
    unsigned i77;
    i77 = (toSmem(T4)) + (12 * i18);
    bool b261;
    b261 = (i18 + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      int64_t i38;
      i38 = i17 + nvfuser_zero;
      Ampere::cpAsyncCa<float, 1>((i77 + (4 * i17)),(ptr51 + (T0.stride[1] * i38)),(b261 && (i38 < 3)));
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
    float* ptr115;
    ptr115 = ptr114 + (T0.stride[0] * i19);
    int64_t i158;
    i158 = 4 + i19;
    unsigned i161;
    i161 = (toSmem(T4)) + (12 * (i158 % 5));
    int64_t i181;
    i181 = 1 + (3 * (i19 % 5));
    int64_t i217;
    i217 = 3 * i19;
    bool b312;
    b312 = i158 < T0.size[0];
    bool b321;
    b321 = i19 < T0.size[0];
    Ampere::cpAsyncPartialBarrier<3>();
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      int64_t i87;
      i87 = i17 + nvfuser_zero;
      Ampere::cpAsyncCa<float, 1>((i161 + (4 * i17)),(ptr115 + (T0.stride[1] * i87)),(b312 && (i87 < 3)));
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    Ampere::cpAsyncCommit();
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 2; ++i22) {
      int64_t i209;
      i209 = i22 + nvfuser_zero;
      T1[((1 + i22) % 2)]
         = T4[(i181 + i22)];
      float T2[1];
      T2[0]
         = T1[(i22 % 2)];
      if ((b321 && (i209 < 3))) {
        T3[(i217 + i209)]
           = T2[0];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T2[1];
    T2[0]
       = T1[0];
    if (b321) {
      T3[(2 + i217)]
         = T2[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
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

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
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i22 = 0; i22 < T0.size[0]; ++i22) {
    int64_t i52;
    i52 = T0.stride[0] * i22;
    int64_t i84;
    i84 = 3 * i22;
    float T1[1];
    float T2[1];
    T1[0] = 0;
    T1[0]
       = T0[i52];
    T2[0]
       = T1[0];
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i111;
      i111 = (1 + i21) + nvfuser_zero;
      float T3[1];
      T3[0]
         = T2[0];
      T4[(i84 + (i21 + nvfuser_zero))]
         = T3[0];
      T1[0] = 0;
      if ((i111 < 3)) {
        T1[0]
           = T0[(i52 + (T0.stride[1] * i111))];
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  bool b144;
  b144 = 0 < T0.size[0];
  float T1[3];
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    if (b144) {
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
  for(nvfuser_index_t i25 = 0; i25 < T0.size[0]; ++i25) {
    int64_t i90;
    i90 = 3 * i25;
    int64_t i123;
    i123 = T0.stride[0] + (T0.stride[0] * i25);
    bool b225;
    b225 = (1 + i25) < T0.size[0];
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i24 = 0; i24 < 3; ++i24) {
      T4[(i90 + (i24 + nvfuser_zero))]
         = T3[i24];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[i21] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b225) {
        T1[i21]
           = T0[(i123 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i1117;
  i1117 = T0.size[0] * T0.size[1];
  float T1[5];
  float T2[5];
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    T1[i36] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    int64_t i68;
    i68 = i36 + nvfuser_zero;
    if ((i68 < i1117)) {
      T1[i36]
         = T0[((T0.stride[0] * (i68 / T0.size[1])) + (T0.stride[1] * (i68 % T0.size[1])))];
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
  for(nvfuser_index_t i40 = 0; i40 < (ceilDiv((T0.size[0] * T0.size[1]), 5)); ++i40) {
    int64_t i238;
    i238 = 5 * i40;
    int64_t i936;
    i936 = 5 + i238;
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i38 = 0; i38 < 5; ++i38) {
      T3[i38]
         = T2[i38];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i39 = 0; i39 < 5; ++i39) {
      int64_t i239;
      i239 = i238 + (i39 + nvfuser_zero);
      if ((i239 < i1117)) {
        T4[i239]
           = T3[i39];
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
      int64_t i945;
      i945 = i936 + (i36 + nvfuser_zero);
      if ((i945 < i1117)) {
        T1[i36]
           = T0[((T0.stride[0] * (i945 / T0.size[1])) + (T0.stride[1] * (i945 % T0.size[1])))];
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i199;
  i199 = T0.stride[0] * 4;
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i25 = 0; i25 < 4; ++i25) {
    int64_t i62;
    i62 = 3 * i25;
    int64_t i79;
    i79 = T0.stride[0] * i25;
    bool b377;
    b377 = (i25 + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i62 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b377) {
        T1[(i62 + i21)]
           = T0[(i79 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
  for(nvfuser_index_t i26 = 0; i26 < T0.size[0]; ++i26) {
    int64_t i169;
    i169 = 4 + i26;
    int64_t i171;
    i171 = 3 * (i169 % 5);
    int64_t i201;
    i201 = i199 + (T0.stride[0] * i26);
    int64_t i298;
    i298 = 3 * i26;
    int64_t i365;
    i365 = 3 * ((1 + i26) % 5);
    bool b440;
    b440 = i169 < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i171 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b440) {
        T1[(i171 + i21)]
           = T0[(i201 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
    for(nvfuser_index_t i24 = 0; i24 < 3; ++i24) {
      T4[(i298 + (i24 + nvfuser_zero))]
         = T3[i24];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i365 + i22)];
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i167;
  i167 = 4 * T0.stride[0];
  int64_t i340;
  i340 = T0.stride[0] * 5;
  bool b510;
  b510 = 0 < T0.size[0];
  bool b583;
  b583 = 4 < T0.size[0];
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    if (b510) {
      T1[i21]
         = T0[(T0.stride[1] * (i21 + nvfuser_zero))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i25 = 0; i25 < 4; ++i25) {
    int64_t i87;
    i87 = 3 + (3 * i25);
    int64_t i116;
    i116 = T0.stride[0] + (T0.stride[0] * i25);
    bool b575;
    b575 = ((1 + i25) + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i87 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b575) {
        T1[(i87 + i21)]
           = T0[(i116 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
    if (b583) {
      T1[(12 + i21)]
         = T0[(i167 + (T0.stride[1] * (i21 + nvfuser_zero)))];
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
  for(nvfuser_index_t i26 = 0; i26 < T0.size[0]; ++i26) {
    int64_t i217;
    i217 = 3 * i26;
    int64_t i304;
    i304 = 3 * (i26 % 5);
    int64_t i342;
    i342 = i340 + (T0.stride[0] * i26);
    int64_t i498;
    i498 = 3 * ((1 + i26) % 5);
    bool b680;
    b680 = (5 + i26) < T0.size[0];
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i24 = 0; i24 < 3; ++i24) {
      T4[(i217 + (i24 + nvfuser_zero))]
         = T3[i24];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i304 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      if (b680) {
        T1[(i304 + i21)]
           = T0[(i342 + (T0.stride[1] * (i21 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i498 + i22)];
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
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T3) {
  alignas(16) extern __shared__ char array[];
  unsigned smem_offset = 0;
  NVFUSER_DEFINE_MAGIC_ZERO
  float* ptr63;
  ptr63 = T0.data;
  float* ptr149;
  ptr149 = (T0.stride[0] * 4) + ptr63;
  smem_offset = alignBufferSize(smem_offset, 16);
  float* T4 = reinterpret_cast<float*>(array + smem_offset);
  smem_offset += (15 * sizeof(float));
  #pragma unroll
  for(nvfuser_index_t i18 = 0; i18 < 4; ++i18) {
    float* ptr76;
    ptr76 = ptr63 + (T0.stride[0] * i18);
    unsigned i106;
    i106 = (toSmem(T4)) + (12 * i18);
    bool b552;
    b552 = (i18 + nvfuser_zero) < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      Ampere::cpAsyncCa<float, 1>((i106 + (4 * i17)),(ptr76 + (T0.stride[1] * (i17 + nvfuser_zero))),b552);
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
    float* ptr150;
    ptr150 = ptr149 + (T0.stride[0] * i19);
    int64_t i281;
    i281 = 4 + i19;
    unsigned i284;
    i284 = (toSmem(T4)) + (12 * (i281 % 5));
    int64_t i319;
    i319 = 1 + (3 * (i19 % 5));
    int64_t i456;
    i456 = 3 * i19;
    bool b609;
    b609 = i281 < T0.size[0];
    #pragma unroll
    for(nvfuser_index_t i17 = 0; i17 < 3; ++i17) {
      Ampere::cpAsyncCa<float, 1>((i284 + (4 * i17)),(ptr150 + (T0.stride[1] * (i17 + nvfuser_zero))),b609);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    Ampere::cpAsyncCommit();
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 2; ++i22) {
      T1[((1 + i22) % 2)]
         = T4[(i319 + i22)];
      float T2[1];
      T2[0]
         = T1[(i22 % 2)];
      T3[(i456 + (i22 + nvfuser_zero))]
         = T2[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T2[1];
    T2[0]
       = T1[0];
    T3[(2 + i456)]
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

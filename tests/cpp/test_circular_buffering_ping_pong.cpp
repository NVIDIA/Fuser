// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <string.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <exception>
#include <utility>

namespace nvfuser {

// The name ping-pong comes from Warp-Specialized Persistent Ping-Pong kernels
// for GEMM in CUTLASS. Here it is generalized to represent any warp specialized
// kernels with multiple computation warp groups, includes GEMM and non-GEMM
// kernels.
//
// For non-GEMM kernel there is no ping-pong switching between tensor cores and
// cuda cores, but multiple warp groups work on different tiles and are
// scheduled and synchronized separately on the same SM.
using PingPongCircularBufferingParams = std::tuple<int>;
using PingPongCircularBuffering =
    NVFuserFixtureParamTest<PingPongCircularBufferingParams>;
TEST_P(PingPongCircularBuffering, StageSlicePositionComputeAt) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  int64_t rows_per_stage = 4;
  int64_t compute_warp_groups = 2;
  int64_t circular_loop = 12;
  int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int dim0 =
      rows_per_stage * compute_warp_groups * sm_count * circular_loop;
  const int dim1 = 128;

  int stages = 6;

  TensorView* tv0 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  fusion.addInput(tv0);

  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  fusion.addOutput(tv3);

  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  tv1->setMemoryType(MemoryType::Shared);

  Fusion fusion_copy = fusion;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(0, rows_per_stage);
    tv->split(0, sm_count);
    tv->split(0, compute_warp_groups);

    // [I, 2, 132, 4]
    tv->axis(0)->parallelize(ParallelType::Serial);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    tv->axis(3)->parallelize(ParallelType::Unroll);
  }

  tv1->axis(1)->parallelize(ParallelType::Serial);
  tv1->axis(4)->parallelize(ParallelType::Bulk);

  for (auto tv : {tv2, tv3}) {
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(4)->parallelize(ParallelType::TIDx);
  }

  // [I, 2, 132, 4] --> [132, I, 2, 4]
  std::unordered_map<int64_t, int64_t> reorder_map = {{0, 1}, {1, 2}, {2, 0}};
  for (auto tv : {tv1, tv2, tv3}) {
    tv->reorder(reorder_map);
  }

  inlineSelectedAt({tv1}, tv2, /*reference_pos=*/2);
  inlineSelectedAt({tv2}, tv2, /*reference_pos=*/3);

  auto [stage_slice_position] = GetParam();
  tv1->circularBuffer(
      stages,
      stages - 1,
      WarpSpecialized(ParallelType::TIDy, stage_slice_position));

  KernelExecutor ke;

  try {
    ke.compile(&fusion, {t0});
  } catch (const std::exception& e) {
    if (stage_slice_position == -1) {
      const char* error_msg = R"(Slice position must be non-negative integer)";
      const char* str_match_pointer = strstr(e.what(), error_msg);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    } else if (stage_slice_position < 2) {
      const char* error_msg =
          R"(Expected outer_most_circular_buffer_position <= inner_most_circular_buffer_position)";
      const char* str_match_pointer = strstr(e.what(), error_msg);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    } else if (stage_slice_position == 5) {
      const char* error_msg =
          R"(Detected an iterDomain with ParallelType::Bulk to the left of stage slice position.)";
      const char* str_match_pointer = strstr(e.what(), error_msg);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    } else if (stage_slice_position == 6) {
      const char* error_msg =
          R"(Slice position must be inside TensorView nDims.)";
      const char* str_match_pointer = strstr(e.what(), error_msg);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    }

    throw;
  }

  auto cg_outputs = ke.run({t0});
  testValidate(&fusion_copy, cg_outputs, {t0}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    ,
    PingPongCircularBuffering,
    ::testing::Combine(::testing::Range(-1, 7)),
    [](const testing::TestParamInfo<PingPongCircularBufferingParams>& info) {
      std::stringstream ss;
      ss << "stage_slice_position_" << std::get<0>(info.param);
      return sanitizeTestName(ss.str());
    });

using PingPongTest = NVFuserTest;
TEST_F(PingPongTest, TwoTmaLoads) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  int rows_per_stage = 4, compute_warp_groups = 2, circular_loop = 12;
  int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int dim0 =
      rows_per_stage * compute_warp_groups * sm_count * circular_loop;
  const int dim1 = 128;
  int stages = 6;
  auto tv0 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  auto tv1 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = set(tv0);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  tv2->setMemoryType(MemoryType::Shared);
  auto tv3 = set(tv1);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  tv3->setMemoryType(MemoryType::Shared);
  auto tv4 = set(tv2);
  auto tv5 = set(tv3);
  auto tv6 = add(tv4, tv5);
  auto tv7 = set(tv6);
  fusion.addOutput(tv7);

  Fusion fusion_copy = fusion;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  at::Tensor t1 = at::randn({dim0, dim1}, options);

  for (auto tv : {tv2, tv3, tv4, tv5, tv6, tv7}) {
    tv->split(0, rows_per_stage);
    tv->split(0, sm_count);
    tv->split(0, compute_warp_groups);
    // [I, 2, 132, 4]
    tv->axis(0)->parallelize(ParallelType::Serial);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::BIDy);
    tv->axis(3)->parallelize(ParallelType::Unroll);
    tv->axis(4)->parallelize(ParallelType::TIDx);
    if (tv == tv2 || tv == tv3) {
      tv->axis(1)->parallelize(ParallelType::Serial);
      tv->axis(4)->parallelize(ParallelType::Bulk);
    }
  }
  // [I, 2, 132, 4] --> [132, I, 2, 4]
  std::unordered_map<int64_t, int64_t> reorder_map;
  reorder_map[0] = 1;
  reorder_map[1] = 2;
  reorder_map[2] = 0;
  for (auto tv : {tv2, tv3, tv4, tv5, tv6, tv7}) {
    tv->reorder(reorder_map);
  }
  for (auto tv : {tv2, tv3}) {
    inlineSelectedAt({tv}, tv, 2);
  }
  for (auto tv : {tv4, tv5, tv6, tv7}) {
    inlineSelectedAt({tv}, tv, 3);
  }
  int64_t stage_slice_position = 3;
  for (auto tv : {tv2, tv3}) {
    tv->circularBuffer(
        stages,
        stages - 1,
        WarpSpecialized(ParallelType::TIDy, stage_slice_position));
  }

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});
  testValidate(&fusion_copy, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}
} // namespace nvfuser

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

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t rows_per_stage = 4;
  constexpr int64_t compute_warp_groups = 2;
  constexpr int64_t circular_loop = 12;
  int64_t sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t dim0 =
      rows_per_stage * compute_warp_groups * sm_count * circular_loop;
  constexpr int64_t dim1 = 128;
  constexpr int64_t stages = 6;

  TensorView* tv0 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  fusion->addInput(tv0);

  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  fusion->addOutput(tv3);

  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  tv1->setMemoryType(MemoryType::Shared);

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

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  KernelExecutor ke;

  try {
    ke.compile(fusion.get(), {t0});
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

  auto out_ref = t0 + t0;
  auto cg_outputs = ke.run({t0});
  testValidate(fusion.get(), cg_outputs, {t0}, {out_ref}, __LINE__, __FILE__);
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

TEST_F(PingPongCircularBuffering, ProducerWarpSpecializedError) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  Fusion fusion;
  FusionGuard fg(&fusion);

  int rows_per_stage = 8;
  int compute_warp_groups = 2;
  int circular_loop = 12;
  int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int dim0 =
      rows_per_stage * compute_warp_groups * sm_count * circular_loop;
  const int dim1 = 128;
  int stages = 6;

  auto tv0 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, tv2);
  fusion.addOutput(tv3);

  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  tv1->setMemoryType(MemoryType::Shared);

  tv1->split(0, rows_per_stage);
  tv1->split(0, sm_count);
  tv1->split(0, compute_warp_groups);

  // [I, 2, 132, 8, Bulk]
  tv1->axis(0)->parallelize(ParallelType::Serial);
  // tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::Bulk);

  for (auto tv : {tv2, tv3}) {
    tv->split(0, rows_per_stage);
    tv->split(1, 2); // split rows_per_stage by 2
    tv->split(0, sm_count);
    tv->split(0, compute_warp_groups);

    // [I, 2, 132, 4, 2, TIDx]
    tv->axis(0)->parallelize(ParallelType::Serial);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  // [I, 2, 132, U, ...] --> [132, I, 2, U, ...]
  std::unordered_map<int64_t, int64_t> reorder_map = {{0, 1}, {1, 2}, {2, 0}};
  for (auto tv : {tv1, tv2, tv3}) {
    tv->reorder(reorder_map);
  }

  // InlineMost causes the producer's loop domain to use the WarpSpecialized
  // axis. The CUDA kernel cannot use this thread axis in the AsyncWarp.
  inlineMost();

  tv1->circularBuffer(stages, stages - 1, WarpSpecialized(ParallelType::TIDy));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  KernelExecutor ke;
  try {
    ke.compile(&fusion, {t0});
  } catch (const std::exception& e) {
    const char* error_msg =
        R"(The warp specialized thread axis cannot appear in the AsyncWarp TensorView)";
    const char* str_match_pointer = strstr(e.what(), error_msg);
    ASSERT_TRUE(str_match_pointer != nullptr);
    return;
  }
}

TEST_F(PingPongCircularBuffering, ProducerConsumerDifferentError) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  Fusion fusion;
  FusionGuard fg(&fusion);

  int rows_per_stage = 8;
  int compute_warp_groups = 2;
  int circular_loop = 12;
  int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int dim0 =
      rows_per_stage * compute_warp_groups * sm_count * circular_loop;
  const int dim1 = 128;
  int stages = 6;

  auto tv0 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, tv2);
  fusion.addOutput(tv3);

  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  tv1->setMemoryType(MemoryType::Shared);

  tv1->split(0, rows_per_stage);
  tv1->split(0, sm_count);
  tv1->split(0, compute_warp_groups);

  // [I, 2, 132, 8, Bulk]
  tv1->axis(0)->parallelize(ParallelType::Serial);
  tv1->axis(2)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::Bulk);

  for (auto tv : {tv2, tv3}) {
    tv->split(0, rows_per_stage);
    tv->split(1, 2); // split rows_per_stage by 2
    tv->split(0, sm_count);
    tv->split(0, compute_warp_groups);

    // [I, 2, 132, 4, 2, TIDx]
    tv->axis(0)->parallelize(ParallelType::Serial);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  // [I, 2, 132, U, ...] --> [132, I, 2, U, ...]
  std::unordered_map<int64_t, int64_t> reorder_map = {{0, 1}, {1, 2}, {2, 0}};
  for (auto tv : {tv1, tv2, tv3}) {
    tv->reorder(reorder_map);
  }

  inlineSelectedAt({tv1}, tv2, /*reference_pos=*/2);
  inlineSelectedAt({tv2}, tv2, /*reference_pos=*/3);

  constexpr int64_t stage_slice_position = 4;
  tv1->circularBuffer(
      stages,
      stages - 1,
      WarpSpecialized(ParallelType::TIDy, stage_slice_position));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  KernelExecutor ke;

  try {
    ke.compile(&fusion, {t0});
  } catch (const std::exception& e) {
    const char* error_msg =
        R"(All iterDomains of the producer and consumer TensorViews to the left of the stage_slice_position must be in the same Broadcast ValGroup.)";
    const char* str_match_pointer = strstr(e.what(), error_msg);
    ASSERT_TRUE(str_match_pointer != nullptr);
    return;
  }

  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser

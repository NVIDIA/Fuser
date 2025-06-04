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
  auto [stage_slice_position] = GetParam();

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t rows_per_stage = 4;
  constexpr int64_t compute_warp_groups = 2;
  constexpr int64_t circular_loop = 12;
  int64_t sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t dim0 = rows_per_stage;
  // Make dim0 non-divisible to test predicate
  if (stage_slice_position == 2) {
    // only not divisible by [circular_loop]
    dim0 *= (compute_warp_groups * sm_count * (circular_loop + 1));
  } else if (stage_slice_position == 3) {
    // may not divisible by [circular_loop], [sm_count], and
    // [compute_warp_groups]
    dim0 *= (compute_warp_groups * sm_count * circular_loop + 1);
  } else {
    dim0 *= (compute_warp_groups * sm_count * circular_loop);
  }

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
    // Axis 3 cannot use ParallelType::Unroll, so explicitly set it to
    // ParallelType::Serial here
    tv->axis(3)->parallelize(ParallelType::Serial);
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

  // Check shared memory size is allocated based on stage_slice_position
  int64_t size_per_row = dim1 * sizeof(float);
  int64_t rows_per_sync = 0;
  if (stage_slice_position == 2) {
    rows_per_sync = rows_per_stage * compute_warp_groups;
  } else if (stage_slice_position == 3) {
    rows_per_sync = rows_per_stage;
  } else if (stage_slice_position == 4) {
    rows_per_sync = 1;
  }
  int64_t smem_buffer_size = size_per_row * stages * rows_per_sync;
  int64_t smem_barrier_size = 128;
  EXPECT_EQ(ke.lastLaunchParams().smem(), smem_buffer_size + smem_barrier_size)
      << "Shared memory size error, expected "
      << smem_buffer_size + smem_barrier_size << ", got "
      << ke.lastLaunchParams().smem();
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

using PingPongSiblingLoadsParams = std::tuple<bool, int>;
using SiblingPingPongCircularBuffering =
    NVFuserFixtureParamTest<PingPongSiblingLoadsParams>;
TEST_P(SiblingPingPongCircularBuffering, TwoTmaLoads) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  auto [use_id_model, stage_slice_position] = GetParam();
  if (use_id_model) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    GTEST_SKIP() << "Fails without setting allocateIndexVariables via IdModel";
  }

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t rows_per_stage = 4;
  constexpr int64_t compute_warp_groups = 2;
  constexpr int64_t inner_split = 3;
  constexpr int64_t circular_loop = 12;
  int64_t sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t dim0 =
      rows_per_stage * compute_warp_groups * sm_count * circular_loop;
  constexpr int64_t dim1 = 128;
  constexpr int64_t stages = 6;

  TensorView* tv0 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  TensorView* tv1 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  TensorView* tv6 = add(tv0, tv1);
  fusion->addOutput(tv6);

  // Create Cache Tensors
  TensorView* tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  TensorView* tv3 = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  TensorView* tv4 = tv2->cacheAfter();
  TensorView* tv5 = tv2->cacheAfter();
  TensorView* tv7 = tv6->cacheBefore();

  // Move first level cache to shared memory
  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  for (TensorView* tv : {tv2, tv3, tv4, tv5, tv6, tv7}) {
    tv->split(0, rows_per_stage);
    tv->split(0, sm_count);
    tv->split(0, compute_warp_groups);
    tv->split(0, inner_split);
    // [I, 3, 2, 132, 4]
    tv->axis(0)->parallelize(ParallelType::Serial);
    tv->axis(2)->parallelize(ParallelType::TIDy);
    tv->axis(3)->parallelize(ParallelType::BIDy);
    tv->axis(4)->parallelize(ParallelType::Unroll);
    tv->axis(5)->parallelize(ParallelType::TIDx);
    if (tv == tv2 || tv == tv3) {
      tv->axis(2)->parallelize(ParallelType::Serial);
      tv->axis(5)->parallelize(ParallelType::Bulk);
    }
  }

  // Reorder Axes
  // [I, 3, 2, 132, 4] --> [132, I, 3, 2, 4]
  std::unordered_map<int64_t, int64_t> reorder_map;
  reorder_map[0] = 1;
  reorder_map[1] = 2;
  reorder_map[2] = 3;
  reorder_map[3] = 0;
  for (TensorView* tv : {tv2, tv3, tv4, tv5, tv6, tv7}) {
    tv->reorder(reorder_map);
  }

  // Inline Step
  for (TensorView* tv : {tv2, tv3}) {
    inlineSelectedAt({tv}, tv, 3);
  }

  for (TensorView* tv : {tv4, tv5, tv6, tv7}) {
    inlineSelectedAt({tv}, tv, 4);
  }

  for (auto tv : {tv2, tv3}) {
    tv->circularBuffer(
        stages,
        stages - 1,
        WarpSpecialized(ParallelType::TIDy, stage_slice_position));
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  at::Tensor t1 = at::randn({dim0, dim1}, options);
  at::Tensor t2 = t0 + t1;

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1});
  auto cg_outputs = ke.run({t0, t1});
  testValidate(fusion.get(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);

  // Validate loop mappings for sibling tma loads tv2 and tv3
  IdModel id_model(fusion.get(), /*build_graphs=*/true);
  const ValGraph& loop_graph = id_model.idGraph(IdMappingMode::LOOP);
  NVF_ERROR(tv2->nDims() == tv3->nDims());
  NVF_ERROR(
      std::ranges::all_of(
          std::ranges::iota_view{0, stage_slice_position},
          [&](int64_t pos) {
            return loop_graph.toGroup(tv2->axis(pos))->has(tv3->axis(pos));
          }),
      "Expected sibling iterDomains to the left of stage_slice_position to "
      "belong to the same ValGroup in LOOP map");
  NVF_ERROR(
      std::ranges::all_of(
          std::ranges::iota_view{stage_slice_position, tv2->nDims()},
          [&](int64_t pos) {
            return !loop_graph.toGroup(tv2->axis(pos))->has(tv3->axis(pos));
          }),
      "Expected sibling iterDomains to the right of and including "
      "stage_slice_position to belong to the same ValGroup in LOOP map");
}
// Stage_Split_Position 2 does not work currently with multiple TMA loads.
// TODO: Enable after supporting multi-role specialization.
INSTANTIATE_TEST_SUITE_P(
    ,
    SiblingPingPongCircularBuffering,
    ::testing::Combine(testing::Bool(), testing::Range(3, 5)),
    [](const testing::TestParamInfo<PingPongSiblingLoadsParams>& info) {
      std::stringstream ss;
      ss << "IdModel_" << std::get<0>(info.param);
      ss << "_stage_slice_position_" << std::get<1>(info.param);
      return sanitizeTestName(ss.str());
    });

} // namespace nvfuser

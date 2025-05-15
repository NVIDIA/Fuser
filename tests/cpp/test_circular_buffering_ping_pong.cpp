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
// For non-GEMM kernel there is no ping-pong switching between tensor cores and
// cuda cores, but multiple warp groups work on different tiles and are
// scheduled and synchronized separately on the same SM.
class PingPongCircularBufferingTest : public NVFuserTest {
 protected:
  void SetUp() override {
    if (cudaArchGuardShouldSkip(9, 0)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
  }
};

TEST_F(PingPongCircularBufferingTest, SimpleFusion) {
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
  fusion.printMath();

  inlineSelectedAt({tv1}, tv2, /*reference_pos=*/2);
  inlineSelectedAt({tv2}, tv2, /*reference_pos=*/3);

  tv1->circularBuffer(stages, stages - 1, WarpSpecialized(ParallelType::TIDy));

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion_copy, cg_outputs, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser

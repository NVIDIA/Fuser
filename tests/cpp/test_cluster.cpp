// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include "type.h"

namespace nvfuser {

class ClusterReductionTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  }
};

TEST_F(ClusterReductionTest, ManusalScheduledSimpleFusion) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());
  const int64_t vect = 8, bdimx = 128, persistent_batch = 2;
  const int64_t cluster_size = 8;
  const int64_t reduction_size = vect * bdimx * persistent_batch * cluster_size;
  DataType dtype = DataType::BFloat16;
  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = maybeCastOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv1, tv3);
  auto tv5 = maybeCastOp(DataType::BFloat16, tv4);
  fusion.addOutput(tv5);
  auto unscheduled_fusion_copy = fusion;

  // [I, R]
  tv2->split(1, 8);
  // [I, R/8, 8]
  tv2->split(1, 128);
  // [I, R/8/128, 128, 8]
  tv2->split(1, 2, false);
  // [I, 2, R/8/128/2, 128, 8]
  // [BIDy, Serial, BIDx(cluster), TIDx, Vectorize for IO]
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-3)->parallelize(ParallelType::BIDx);
  tv2->axis(-3)->setClusteredBlocks(true);
  tv2->axis(0)->parallelize(ParallelType::BIDy);

  auto reference = tv2->rFactor({-1, -4});

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv5->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({256, reduction_size}, options);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0});
  auto outputs = ke.run({t0});
  testValidate(&unscheduled_fusion_copy, outputs, {t0});
}

} // namespace nvfuser

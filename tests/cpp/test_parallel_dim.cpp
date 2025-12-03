// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/internal_nodes.h>
#include <ops/all_ops.h>
#include <parallel_dimension_map.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ParallelDimTest = NVFuserTest;

TEST_F(ParallelDimTest, Basic) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  fusion->getParallelDim(ParallelType::Serial);

  fusion->getParallelDim(ParallelType::BIDx);

  fusion->getParallelDim(ParallelType::ClusterIDy);
  fusion->getParallelDim(ParallelType::ClusterCtaIDy);

  // Test that we can only create "real" parallel dims. Derived dims must be
  // created with ops like dim->split()
  EXPECT_ANY_THROW(fusion->getParallelDim(ParallelType::Derived););
  EXPECT_ANY_THROW(fusion->getParallelDim(ParallelType::Count););
}

TEST_F(ParallelDimTest, Binding) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);

  auto tv1 = set(tv0);

  fusion->addOutput(tv1);

  tv1->split(2, 256); // [ i0, i1, i2/256, 256 ]
  tv1->split(3, 32); // [ i0, i1, i2/256, 256/32, 32 ]
  tv1->split(1, 2); // [ i0, i1/128, 128, i2/256, 256/32, 32 ]

  tv1->axis(0)->setParallelDim(fusion->getParallelDim(ParallelType::BIDx));
  tv1->axis(2)->parallelize(ParallelType::TIDy);

  auto [warp_id, lane_id] =
      fusion->getParallelDim(ParallelType::TIDx)->split("WarpID", "LaneID");

  tv1->axis(-1)->setParallelDim(lane_id);
  tv1->axis(-2)->setParallelDim(warp_id);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({192, 768, 384}, options);
  std::vector<c10::IValue> inputs{t0};

  KernelExecutor ke;
  ke.compile(fusion, inputs);

  const ParallelDimensionMap& pdm =
      ke.compiledKernel()->lowered()->info().parallelDimensionMap();

  EXPECT_TRUE(pdm.has(ParallelType::BIDx));
  EXPECT_TRUE(pdm.has(ParallelType::TIDx));
  EXPECT_TRUE(pdm.has(ParallelType::TIDy));

  const auto cg_outputs = ke.run(inputs);

  const LaunchParams lp = ke.lastLaunchParams();
  EXPECT_EQ(lp.gdimx(), t0.size(0));
  EXPECT_EQ(lp.bdimx(), 256);
  EXPECT_EQ(lp.bdimy(), 2);

  testValidate(fusion, cg_outputs, inputs, __LINE__, __FILE__);
}

} // namespace nvfuser

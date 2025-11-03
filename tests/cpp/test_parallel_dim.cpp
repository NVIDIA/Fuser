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
#include <tests/cpp/utils.h>

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

  std::cout << fusion->parallelDimGraphMermaid() << std::endl;

  EXPECT_ANY_THROW(fusion->getParallelDim(ParallelType::Count););
}

TEST_F(ParallelDimTest, Binding) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = set(tv0);

  fusion->addOutput(tv1);

  tv1->axis(0)->setParallelDim(fusion->getParallelDim(ParallelType::BIDx));
  tv1->split(1, 256);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  KernelExecutor ke;
  ke.compile(fusion);
}

} // namespace nvfuser

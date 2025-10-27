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
#include <tests/cpp/utils.h>

namespace nvfuser {

using ParallelDimTest = NVFuserTest;

TEST_F(ParallelDimTest, Basic) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  fusion->getParallelDim(ParallelType::Serial);

  fusion->getParallelDim(ParallelType::BIDx);

  EXPECT_ANY_THROW(fusion->getParallelDim(ParallelType::Count););

}

} // namespace nvfuser

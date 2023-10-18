// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <csrc/fusion.h>
#include <csrc/ops/alias.h>
#include <csrc/optimization/alias_analysis.h>
#include <test/utils.h>

namespace nvfuser {

using AliasAnalysisTest = NVFuserTest;
using testing::IsEmpty;
using testing::Pair;
using testing::UnorderedElementsAre;

TEST_F(AliasAnalysisTest, View_ContiguousAndSameAllocationOrder) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(fusion);
  EXPECT_THAT(alias_analysis, UnorderedElementsAre(Pair(out, in)));
}

TEST_F(AliasAnalysisTest, ChainOfViews) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> intermediate_shape({2, 12});
  const std::vector<int64_t> out_shape({24});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* intermediate = reshape(in, in_shape, intermediate_shape);
  TensorView* out = reshape(intermediate, intermediate_shape, out_shape);
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(fusion);
  EXPECT_THAT(
      alias_analysis,
      UnorderedElementsAre(Pair(out, intermediate), Pair(intermediate, in)));
}

TEST_F(AliasAnalysisTest, View_DifferentAllocationOrder) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);
  out->setAllocationDomain(
      {out->axis(1), out->axis(0)}, /*new_contiguity=*/true);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(fusion);
  EXPECT_THAT(alias_analysis, IsEmpty());
}

TEST_F(AliasAnalysisTest, View_NonContiguous) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);
  out->setAllocationDomain(
      {out->axis(0), out->axis(1)}, /*new_contiguity=*/{true, false});

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(fusion);
  EXPECT_THAT(alias_analysis, IsEmpty());
}

TEST_F(AliasAnalysisTest, Permute) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = permute(in, {1, 2, 0});
  fusion.addOutput(out);

  // We haven't handled `Set.Permute` yet.
  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(fusion);
  EXPECT_THAT(alias_analysis, IsEmpty());
}

} // namespace nvfuser

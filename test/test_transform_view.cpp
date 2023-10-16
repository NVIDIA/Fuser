// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <csrc/fusion.h>
#include <csrc/ops/alias.h>
#include <csrc/transform_view.h>
#include <ir/interface_nodes.h>
#include <test/utils.h>

namespace nvfuser {

using testing::ElementsAre;
using TransformViewTest = NVFuserTest;

MATCHER_P(isMerge, index, "") {
  const ViewTransform* transform = arg.get();
  if (const MergeTransform* merge =
          dynamic_cast<const MergeTransform*>(transform)) {
    return merge->index() == index;
  }
  return false;
}

MATCHER_P2(isSplit, index, factor, "") {
  const ViewTransform* transform = arg.get();
  if (const SplitTransform* split =
          dynamic_cast<const SplitTransform*>(transform)) {
    return split->index() == index && split->split_factor() == factor;
  }
  return false;
}

TEST_F(TransformViewTest, MergeSplit) {
  const std::vector<int64_t> in_shape({2, 1, 3, 4});
  const std::vector<int64_t> out_shape({6, 2, 2, 1});

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);

  AnalyzeViewResult view_analysis = analyzeView(in, in_shape, out_shape);
  // The last dimension of out_shape is broadcast.
  EXPECT_THAT(
      view_analysis.broadcast_axes, ElementsAre(false, false, false, true));
  // Dimension 1 is squeezed.
  EXPECT_THAT(
      view_analysis.squeeze_axes, ElementsAre(false, true, false, false));

  EXPECT_THAT(view_analysis.transforms, ElementsAre(isMerge(0), isSplit(1, 2)));
}

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// #include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/vectorize_helper.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fstream>

namespace nvfuser {

namespace {

void checkMappedVal(
    const std::unordered_map<TensorView*, Val*>& map,
    TensorView* tv_target,
    int64_t val) {
  auto iter = map.find(tv_target);
  EXPECT_TRUE(iter != map.end());
  if (iter != map.end()) {
    EXPECT_EQ(iter->second->evaluate(), val);
  }
}

} // namespace

using VectorizationAnalysisTest = NVFuserTest;

// Simple pad test
TEST_F(
    VectorizationAnalysisTest,
    ContigInnerDimsMapperResizeFastestDimensionP2C) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  auto tv0 = makeContigConcreteTensor({4, 8, 16});
  fusion.addInput(tv0);

  // positive resize (+2, +2)
  auto inner_pos =
      pad(tv0, {IrBuilder::create<Val>(2L), IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(inner_pos, 2));
  fusion.addOutput(inner_pos);

  // positive uneven resize (+4, +2)
  auto inner_pos_uneven =
      pad(tv0, {IrBuilder::create<Val>(4L), IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(inner_pos_uneven, 2));
  fusion.addOutput(inner_pos_uneven);

  // positive large resize (+32, +32)
  auto inner_pos_large =
      pad(tv0, {IrBuilder::create<Val>(32L), IrBuilder::create<Val>(32L)});
  // projected extent is 16
  expection_list.emplace_back(std::make_pair(inner_pos_large, 16));
  fusion.addOutput(inner_pos_large);

  // negative resize (-2, -2)
  auto inner_neg =
      pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-2L)});
  expection_list.emplace_back(std::make_pair(inner_neg, 2));
  fusion.addOutput(inner_neg);

  // negative uneven resize (-2, -4)
  auto inner_neg_uneven =
      pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-4L)});
  expection_list.emplace_back(std::make_pair(inner_neg_uneven, 2));
  fusion.addOutput(inner_neg_uneven);

  // negative large resize to zero (-8, -8)
  auto inner_neg_large =
      pad(tv0, {IrBuilder::create<Val>(-8L), IrBuilder::create<Val>(-8L)});
  // output id with extent 0 cannot be vectorized
  expection_list.emplace_back(std::make_pair(inner_neg_large, 0));
  fusion.addOutput(inner_neg_large);

  // uneven resize (-2, 4)
  auto inner_uneven =
      pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(4L)});
  expection_list.emplace_back(std::make_pair(inner_uneven, 2));
  fusion.addOutput(inner_uneven);

  // one side resize (0, 4)
  auto inner_one_size =
      pad(tv0, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(4L)});
  // resize extent of 0 wouldn't affect vectorization factor
  expection_list.emplace_back(std::make_pair(inner_one_size, 4));
  fusion.addOutput(inner_one_size);

  std::unordered_map<TensorView*, Val*> projected_extent_map =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv0, tv0->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();

  for (const auto& [tv, val] : expection_list) {
    checkMappedVal(projected_extent_map, tv, val);
  }
}

// Simple pad test
TEST_F(
    VectorizationAnalysisTest,
    ContigInnerDimsMapperResizeFastestDimensionC2P) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  auto tv0 = makeContigConcreteTensor({4, 8, 8});
  fusion.addInput(tv0);
  // positive resize (+24, +24)
  auto tv1 =
      pad(tv0, {IrBuilder::create<Val>(24L), IrBuilder::create<Val>(24L)});
  fusion.addOutput(tv1);

  // negative resize to zero (-4, -4)
  auto tv2 =
      pad(tv0, {IrBuilder::create<Val>(-4), IrBuilder::create<Val>(-4L)});
  fusion.addOutput(tv2);

  std::unordered_map<TensorView*, Val*> projected_extent_map_from_tv1 =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv1, tv1->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  checkMappedVal(projected_extent_map_from_tv1, tv0, 8);
  checkMappedVal(projected_extent_map_from_tv1, tv2, 0);

  // because tv2's fastest dimension is resized to 0
  std::unordered_map<TensorView*, Val*> projected_extent_map_from_tv2 =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv2, tv2->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  checkMappedVal(projected_extent_map_from_tv2, tv0, 0);
  checkMappedVal(projected_extent_map_from_tv2, tv1, 0);
}

TEST_F(VectorizationAnalysisTest, ContigInnerDimsMapperResizeMiddleDimension) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  auto tv0 = makeContigConcreteTensor({4, 8, 16});
  fusion.addInput(tv0);

  // positive resize (+2, +2)
  auto middle_pos =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(2L),
           IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(middle_pos, 2 * 16));
  fusion.addOutput(middle_pos);

  // negative resize (-2, -2)
  auto middle_neg =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(-2L),
           IrBuilder::create<Val>(-2L)});
  expection_list.emplace_back(std::make_pair(middle_neg, 2 * 16));
  fusion.addOutput(middle_neg);

  std::unordered_map<TensorView*, Val*> projected_extent_map =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv0, tv0->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  for (const auto& [tv, val] : expection_list) {
    checkMappedVal(projected_extent_map, tv, val);
  }
}

TEST_F(
    VectorizationAnalysisTest,
    ContigInnerDimsMapperResizeMultipleDimension) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4, 8, 32});
  fusion.addInput(tv0);

  // the inner most dimension of resize would participate in vectorization
  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(8L),
           IrBuilder::create<Val>(8L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(4L)});
  fusion.addOutput(tv1);

  std::unordered_map<TensorView*, Val*> projected_extent_map_from_producer =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv0, tv0->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  checkMappedVal(projected_extent_map_from_producer, tv1, 8);

  std::unordered_map<TensorView*, Val*> projected_extent_map_from_consumer =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv1, tv1->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  checkMappedVal(projected_extent_map_from_consumer, tv0, 8);
}

TEST_F(VectorizationAnalysisTest, ContigInnerDimsMapperResizeStacked) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  auto tv0 = makeContigConcreteTensor({4, 8, 36});
  fusion.addInput(tv0);
  // resize on different dimension
  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(-4L),
           IrBuilder::create<Val>(-4L)});
  auto tv2 =
      pad(tv1,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(-2L),
           IrBuilder::create<Val>(-2L)});
  // only the inner most resize is included in vectorization analysis
  expection_list.emplace_back(std::make_pair(tv2, 2 * 36));
  fusion.addOutput(tv2);

  // resize on the same dimension, squeeze size to zero
  auto tv3 =
      pad(tv0, {IrBuilder::create<Val>(-9L), IrBuilder::create<Val>(-9L)});
  auto tv4 =
      pad(tv3, {IrBuilder::create<Val>(-9L), IrBuilder::create<Val>(-9L)});
  // output id with extent 0 cannot be vectorized
  expection_list.emplace_back(std::make_pair(tv4, 0));
  fusion.addOutput(tv4);

  // resize on the same dimension
  auto tv5 =
      pad(tv0, {IrBuilder::create<Val>(-6L), IrBuilder::create<Val>(-6L)});
  auto tv6 = pad(tv5, {IrBuilder::create<Val>(9L), IrBuilder::create<Val>(9L)});
  // two resize operation would stack
  expection_list.emplace_back(std::make_pair(tv6, 3));
  fusion.addOutput(tv6);

  std::unordered_map<TensorView*, Val*> projected_extent_map =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv0, tv0->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  for (const auto& [tv, val] : expection_list) {
    checkMappedVal(projected_extent_map, tv, val);
  }
}

} // namespace nvfuser

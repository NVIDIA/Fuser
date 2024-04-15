// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <options.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

namespace {
using FeaturesTest = NVFuserTest;
} // namespace

TEST_F(FeaturesTest, DefaultFeatures) {
  FeatureSet feats;

  // IndexHoist is enabled by default (this should fail if
  // NVFUSER_DISABLE=index_hoist is given)
  ASSERT_TRUE(feats.has(Feature::IndexHoist));

  feats.set(Feature::IndexHoist, false);
  EXPECT_FALSE(feats.has(Feature::IndexHoist));

  // Test insert and erase
  feats.insert(Feature::IndexHoist);
  EXPECT_TRUE(feats.has(Feature::IndexHoist));
  feats.erase(Feature::IndexHoist);
  EXPECT_FALSE(feats.has(Feature::IndexHoist));

  // Test copy constructor
  FeatureSet feats_copy = feats;
  EXPECT_FALSE(feats.has(Feature::IndexHoist));

  // Test adding an argument, retrieving it. Test that it survives a copy.
  EXPECT_TRUE(feats.args(Feature::WarnRegisterSpill).empty());
  feats.args(Feature::WarnRegisterSpill).push_back("foo");
  EXPECT_EQ(feats.args(Feature::WarnRegisterSpill).size(), 1);
  FeatureSet feats_copy2 = feats;
  EXPECT_EQ(feats_copy2.args(Feature::WarnRegisterSpill).size(), 1);
}

} // namespace nvfuser

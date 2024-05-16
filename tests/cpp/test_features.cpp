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

#include <iostream>

namespace nvfuser {

namespace {
using FeaturesTest = NVFuserTest;
} // namespace

TEST_F(FeaturesTest, DefaultFeatures) {
  // Test looking up features by name
  ASSERT_FALSE(nameToFeature("foo").has_value());
  ASSERT_EQ(nameToFeature("index_hoist"), Feature::IndexHoist);

  FeatureSet feats;

  // IndexHoist is enabled by default (this should fail if the environment
  // variable NVFUSER_DISABLE=index_hoist is present)
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
  EXPECT_FALSE(feats.hasArgs(Feature::WarnRegisterSpill));
  feats.setArgs(Feature::WarnRegisterSpill, {"10"});
  EXPECT_TRUE(feats.hasArgs(Feature::WarnRegisterSpill));
  EXPECT_EQ(feats.getArgs(Feature::WarnRegisterSpill).size(), 1);
  FeatureSet feats_copy2 = feats;
  EXPECT_TRUE(feats_copy2.hasArgs(Feature::WarnRegisterSpill));
  EXPECT_EQ(feats_copy2.getArgs(Feature::WarnRegisterSpill).size(), 1);

  {
    FeatureSet f;
    EXPECT_EQ(f.toString(), "FeatureSet[]");
    f.insert(Feature::KernelDb);
    // Enabling a non-default feature
    EXPECT_EQ(f.toString(), "FeatureSet[+kernel_db]");
    // Disabling a default feature
    f.erase(Feature::ExprSimplify);
    EXPECT_EQ(f.toString(), "FeatureSet[-expr_simplify, +kernel_db]");
    std::stringstream ss;
    ss << f;
    EXPECT_EQ(f.toString(), ss.str());
    // kernel_db does not affect execution, so it gets reset to default
    EXPECT_EQ(
        resetNonCompilationFeatures(f).toString(),
        "FeatureSet[-expr_simplify]");
  }
}

} // namespace nvfuser

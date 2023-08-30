// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/utils.h>

#include <polymorphic_value.h>

namespace nvfuser {

class PolymorphicValueTest : public NVFuserTest {};

TEST_F(PolymorphicValueTest, OpaqueEquality) {
  Opaque a{DataType::Int}, b{DataType::Int};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(a, b);
  EXPECT_EQ(b, a);
}

} // namespace nvfuser

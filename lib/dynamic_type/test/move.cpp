// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "dynamic_type.h"

using namespace dynamic_type;

class DynamicTypeTest : public ::testing::Test {};

TEST_F(DynamicTypeTest, MoveCtor) {
  struct NonCopyable {
    NonCopyable() = default;
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable(NonCopyable&&) = default;
    NonCopyable& operator=(const NonCopyable&) = delete;
    NonCopyable& operator=(NonCopyable&&) = default;
  };
  using NonCopyableType = DynamicType<NoContainers, NonCopyable>;
  static_assert(std::is_move_constructible_v<NonCopyableType>);
  static_assert(std::is_move_assignable_v<NonCopyableType>);
  static_assert(std::is_nothrow_move_constructible_v<NonCopyableType>);
  static_assert(std::is_nothrow_move_assignable_v<NonCopyableType>);
  NonCopyable a;
  // This should not compile:
  // NonCopyableType bb(a);
  NonCopyableType b(std::move(a));
  // Suppress unused var warning
  (void)b;
}

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <iostream>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <exceptions.h>

#include <tests/cpp/utils.h>

namespace nvfuser {

using PerfHintTest = NVFuserTest;

TEST_F(PerfHintTest, Basic) {
  std::stringstream ss;
  DebugStreamGuard dsg(ss);

  {
    ss.str("");
    DisableOptionsGuard dog;
    // NVFUSER_DISABLE=
    DisableOptionsGuard::getCurOptions().unset(DisableOption::PerfHints);
    // Test with multiple args including non-string args that implement
    // operator<<(const std::ostream&)
    NVF_PERF_HINT("test_perf_bug", "Your kernel will be slow because ", 42);
    EXPECT_FALSE(ss.str().empty());
    EXPECT_EQ(
        ss.str(),
        "NVFUSER PERF HINT [test_perf_bug]: Your kernel will be slow because 42");
  }

  {
    ss.str("");
    DisableOptionsGuard dog;
    // NVFUSER_DISABLE=perf_hints(foo)
    DisableOptionsGuard::getCurOptions().set(DisableOption::PerfHints, {"foo"});
    NVF_PERF_HINT("test_perf_bug", "Your kernel will be slow because ", 42);
    EXPECT_FALSE(ss.str().empty());
  }

  {
    ss.str("");
    DisableOptionsGuard dog;
    // NVFUSER_DISABLE=perf_hints(-test_perf_bug)
    DisableOptionsGuard::getCurOptions().set(
        DisableOption::PerfHints, {"-test_perf_bug"});
    NVF_PERF_HINT("test_perf_bug", "Your kernel will be slow because ", 42);
    EXPECT_FALSE(ss.str().empty());
  }

  {
    ss.str("");
    DisableOptionsGuard dog;
    // NVFUSER_DISABLE=perf_hints(test_perf_bug)
    DisableOptionsGuard::getCurOptions().set(
        DisableOption::PerfHints, {"test_perf_bug"});
    NVF_PERF_HINT("test_perf_bug", "Your kernel will be slow because ", 42);
    EXPECT_TRUE(ss.str().empty());
  }

  {
    ss.str("");
    DisableOptionsGuard dog;
    // NVFUSER_DISABLE=perf_hints
    DisableOptionsGuard::getCurOptions().set(DisableOption::PerfHints);
    NVF_PERF_HINT("test_perf_bug", "Your kernel will be slow because ", 42);
    EXPECT_TRUE(ss.str().empty());
  }
}

} // namespace nvfuser

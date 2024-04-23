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

  NVF_PERF_HINT("test_perf_bug", "Your kernel will be slow because ", 42);

  std::cout << "Debug stream is \"" << ss.str() << "\"" << std::endl;
}

} // namespace nvfuser

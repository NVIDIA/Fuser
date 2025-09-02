// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// This is a refactor of the tests used for PyTorch macros --
// NVF_ERROR and NVF_CHECK.

#include <stdexcept>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <exceptions.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ExceptionTest = NVFuserTest;

using testing::HasSubstr;

namespace {

template <class Functor>
inline void expectThrows(Functor&& functor, const char* expectedMessage) {
  try {
    std::forward<Functor>(functor)();
  } catch (const nvfError& e) {
    EXPECT_THAT(e.what_without_backtrace(), HasSubstr(expectedMessage));
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \""
                << expectedMessage << "\" but didn't throw";
}
} // namespace

TEST_F(ExceptionTest, ErrorFormatting) {
  expectThrows(
      []() { NVF_CHECK(false, "This is invalid"); }, "This is invalid");
}

static int assertionArgumentCounter = 0;

namespace {
int getAssertionArgument() {
  return ++assertionArgumentCounter;
}

void failCheck() {
  NVF_CHECK(false, "message ", getAssertionArgument());
}

void failError() {
  NVF_THROW("message ", getAssertionArgument());
}
} // namespace

TEST_F(ExceptionTest, MultipleArgCalls) {
  assertionArgumentCounter = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failCheck());
  EXPECT_EQ(assertionArgumentCounter, 1) << "NVF_CHECK called argument twice";

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failError());
  EXPECT_EQ(assertionArgumentCounter, 2) << "NVF_ERROR called argument twice";
}

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ir/iostream.h>
#include <ops/alias.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using testing::HasSubstr;

using IostreamTest = NVFuserTest;

TEST_F(IostreamTest, Fusion) {
  Fusion fusion;
  {
    FusionGuard fg(&fusion);

    TensorView* in = makeSymbolicTensor(2);
    fusion.addInput(in);
    TensorView* out = flatten(in);
    fusion.addOutput(out);
  }

  captureStdout();
  std::cout << &fusion;
  std::string std_out = getCapturedStdout();

  EXPECT_THAT(std_out, HasSubstr("view("));
}

TEST_F(IostreamTest, NullFusion) {
  Fusion* fusion = nullptr;
  captureStdout();
  std::cout << fusion;
  std::string std_out = getCapturedStdout();
  EXPECT_THAT(std_out, HasSubstr("<null>"));
}

} // namespace nvfuser

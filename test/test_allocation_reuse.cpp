// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <device_lower/pass/interval_tree.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

class AllocationReuse : public NVFuserTest {};

// Simple exercises for CenteredIntervalTree
TEST_F(AllocationReuse, CenteredIntervalTree) {
  //CenteredIntervalTree cit;
}

} // namespace nvfuser

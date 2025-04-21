// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>

namespace nvfuser {

// A simple garbage collection mechanism to snapshot Exprs and Vals upon entry
// and restore them upon exit. This avoids creating too many garbage Exprs and
// Vals in the complete fusion, making cloning slow.
class StatementGuard {
 public:
  StatementGuard(Fusion* fusion);
  ~StatementGuard();

 private:
  Fusion* fusion_;
  const int64_t prev_num_exprs_;
  const int64_t prev_num_vals_;
};

} // namespace nvfuser

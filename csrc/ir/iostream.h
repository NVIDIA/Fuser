// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <iosfwd>

#include "base.h"

namespace nvfuser {

void checkInlineable(const Expr* expr);

static constexpr char const* kTab = "  ";

// Indent the generated code
inline std::ostream& indent(std::ostream& os, int64_t indent_size) {
  for (const auto _ : arange(indent_size)) {
    (void)_; // Suppress unused variable warning
    os << "  ";
  }
  return os;
}

} // namespace nvfuser

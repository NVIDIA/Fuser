// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <iosfwd>

#include <dispatch.h>
#include <exceptions.h>
#include <visibility.h>

namespace nvfuser {

class Fusion;
class Scope;
namespace kir {
class Kernel;
} // namespace kir

namespace hir {
class HostIrContainer;
} // namespace hir

void checkInlineable(const Expr* expr);
static constexpr char const* kTab = "  ";

// Indent the generated code
inline std::ostream& indent(std::ostream& os, int indent_size) {
  for (const auto _ : arange(indent_size)) {
    (void)_; // Suppress unused variable warning
    os << "  ";
  }
  return os;
}

NVF_API std::ostream& operator<<(std::ostream& os, const Statement& stmt);
NVF_API std::ostream& operator<<(std::ostream& os, const Statement* stmt);

NVF_API std::ostream& operator<<(std::ostream& os, const Fusion& f);
NVF_API std::ostream& operator<<(std::ostream& os, const Fusion* f);

} // namespace nvfuser

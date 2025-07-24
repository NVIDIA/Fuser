// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

#include <dispatch.h>

#include <iostream>

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

//! Define pretty printing functions for IR nodes
//!
//! This class is intended for debug printing, so it attempts
//! to handle invalid states as well.
//!
class IrPrinter {
 public:
  explicit IrPrinter(std::ostream& os, int indent_size = 0)
      : os_(os), indent_size_(indent_size) {}
  virtual ~IrPrinter() = default;

  void resetIndent() {
    indent_size_ = 0;
  }

  bool printInline() const {
    return print_inline_;
  }

  virtual void handle(Fusion* f);

  // handle calls some non const fusion ops,
  // eventhough fusion should remain unchanged.
  // Need to look into this.
  virtual void handle(const Fusion* f) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    handle(const_cast<Fusion*>(f));
  }

  virtual void handle(Fusion& f) {
    handle(&f);
  }

  virtual void handle(const kir::Kernel* kernel);
  virtual void handle(kir::Kernel& kernel);

  virtual void handle(const hir::HostIrContainer* host_ir_container);
  virtual void handle(hir::HostIrContainer& host_ir_container);

 protected:
  std::ostream& os() {
    return os_;
  }

 private:
  std::ostream& os_;
  bool print_inline_ = false;
  int indent_size_ = 0;
};

NVF_API std::ostream& operator<<(std::ostream& os, const Statement* stmt);

std::ostream& operator<<(std::ostream& os, Fusion* f);
NVF_API std::ostream& operator<<(std::ostream& os, Fusion& f);

} // namespace nvfuser

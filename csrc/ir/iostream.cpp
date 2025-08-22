// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/iostream.h>

#include <ostream>

#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/printer.h>
#include <kernel_ir.h>

namespace nvfuser {

// Make sure we can inline something, before we attempt to.
void checkInlineable(const Expr* expr) {
  for (Val* input : expr->inputs()) {
    NVF_CHECK(
        input->isScalar() || input->isA<kir::TensorIndex>() ||
            (expr->isA<UnaryOp>() &&
             expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Address),
        "Printing inline computations involving values other than scalars is "
        "not currently supported.");
  }
  NVF_CHECK(
      expr->outputs().size() == 1,
      "Cannot print inline computations if there's more than one output.");
  NVF_CHECK(
      expr->output(0)->isScalar() || expr->output(0)->isA<NamedScalar>(),
      "Printing inline computations involving values other than scalars is not "
      "currently supported.");
}

std::ostream& operator<<(std::ostream& os, const Statement& stmt) {
  return os << stmt.toString();
}

std::ostream& operator<<(std::ostream& os, const Fusion& f) {
  IrPrinter p(os);
  p.handle(&f);
  return os;
}

namespace {
template <typename T>
void print(std::ostream& os, const T* t) {
  if (t == nullptr) {
    os << "<null>";
  } else {
    os << *t;
  }
}
} // namespace

std::ostream& operator<<(std::ostream& os, const Statement* t) {
  print(os, t);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Fusion* t) {
  print(os, t);
  return os;
}

} // namespace nvfuser

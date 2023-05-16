// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/iostream.h>
#include <ir/printer.h>

#include <device_lower/utils.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <kernel.h>
#include <utils.h>

#include <c10/util/irange.h>

namespace nvfuser {

// Make sure we can inline something, before we attempt to.
void checkInlineable(const Expr* expr) {
  for (auto input : expr->inputs()) {
    TORCH_CHECK(
        input->isScalar() || input->isA<kir::TensorIndex>() ||
            (expr->isA<UnaryOp>() &&
             expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Address),
        "Printing inline computations involving values other than scalars is not currently supported.");
  }
  TORCH_CHECK(
      expr->outputs().size() == 1,
      "Cannot print inline computations if there's more than one output.");
  TORCH_CHECK(
      expr->output(0)->isScalar() || expr->output(0)->isA<NamedScalar>(),
      "Printing inline computations involving values other than scalars is not currently supported.");
}

void IrPrinter::handle(Fusion* fusion) {
  FUSER_PERF_SCOPE("IrPrinter");
  resetIndent();
  for (const Expr* expr : fusion->exprs()) {
    os_ << expr->toString();
  }
}

void IrPrinter::handle(const kir::Kernel* kernel) {
  TORCH_CHECK(kernel != nullptr);

  // kernel declaration
  os_ << "\nKERNEL (";
  for (auto in : kernel->inputs()) {
    os_ << in->toString();
    if (in != kernel->inputs().back()) {
      os_ << ", ";
    }
  }
  os_ << ") -> (";
  for (auto out : kernel->outputs()) {
    os_ << out->toString();
    if (out != kernel->outputs().back()) {
      os_ << ", ";
    }
  }
  os_ << ") :\n";

  // kernel body
  indent_size_++;
  for (auto expr : kernel->topLevelExprs()) {
    os_ << expr->toString();
  }
  indent_size_--;
  os_ << "END.\n\n";
}

void IrPrinter::handle(kir::Kernel& kernel) {
  handle(&kernel);
}

void IrTransformPrinter::handle(Fusion* f) {
  auto all_vals = f->usedMathVals();

  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    os() << tv->toString();
    os() << "\n";
    printTransforms(tv);
  }
}

void IrTransformPrinter::printTransforms(TensorView* tv) {
  auto root_domain = tv->getRootDomain();
  os() << " root domain : (" << toDelimitedString(root_domain) << ")\n";

  if (tv->hasRFactor()) {
    auto rfactor_domain = tv->getRFactorDomain();

    auto all_exp = DependencyCheck::getAllExprsBetween(
        {root_domain.begin(), root_domain.end()},
        {rfactor_domain.begin(), rfactor_domain.end()});

    for (auto exp : all_exp) {
      os() << "  " << exp->toString();
    }

    os() << " rfactor domain : (" << toDelimitedString(rfactor_domain) << ")\n";
  }

  os() << " contiguity: " << tv->domain()->getContiguityString() << "\n";

  const auto& from = tv->getMaybeRFactorDomain();
  const auto& leaf = tv->getLeafDomain();
  auto all_exp = DependencyCheck::getAllExprsBetween(
      {from.begin(), from.end()}, {leaf.begin(), leaf.end()});

  for (auto exp : all_exp) {
    os() << "  " << exp->toString();
  }
  os() << " leaf domain : (" << toDelimitedString(leaf) << ")\n";
}

std::ostream& operator<<(std::ostream& os, const Statement* stmt) {
  return os << stmt->toString();
}

std::ostream& operator<<(std::ostream& os, Fusion* f) {
  IrPrinter p(os);
  FusionGuard guard(f);
  p.handle(f);
  return os;
}

std::ostream& operator<<(std::ostream& os, Fusion& f) {
  return os << &f;
}

} // namespace nvfuser

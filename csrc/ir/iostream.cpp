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
#include <host_ir/container.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel.h>
#include <utils.h>

namespace nvfuser {

// Make sure we can inline something, before we attempt to.
void checkInlineable(const Expr* expr) {
  for (auto input : expr->inputs()) {
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

void IrPrinter::handle(const Fusion* fusion) {
  FUSER_PERF_SCOPE("IrPrinter");
  resetIndent();
  for (const Expr* expr : fusion->exprs()) {
    os_ << expr->toString();
  }
}

void IrPrinter::handle(const kir::Kernel* kernel) {
  NVF_CHECK(kernel != nullptr);

  // kernel declaration
  os_ << "\n%Kernel { (";
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
  os_ << "\n} // %Kernel.\n\n";
}

void IrPrinter::handle(const hir::HostIrContainer* host_ir_container) {
  NVF_CHECK(host_ir_container != nullptr);

  // host_ir_container declaration
  os() << "\n%HostIrContainer { (";
  for (auto in : host_ir_container->inputs()) {
    os() << in->toString(indent_size_);
    if (in != host_ir_container->inputs().back()) {
      os() << ", ";
    }
  }
  os() << ") -> (";
  for (auto out : host_ir_container->outputs()) {
    os() << out->toString(indent_size_);
    if (out != host_ir_container->outputs().back()) {
      os() << ", ";
    }
  }
  os() << ") :\n";

  // host_ir_container body
  indent_size_++;
  for (auto expr : host_ir_container->topLevelExprs()) {
    os() << expr->toString(indent_size_);
  }
  indent_size_--;
  for (auto* host_unit : ir_utils::filterByType<hir::HostUnit>(
           host_ir_container->unordered_exprs())) {
    os() << std::endl;
    os() << host_unit->toString(indent_size_);
  }
  os() << "} // %HostIrContainer\n\n";
}

void IrTransformPrinter::handle(const Fusion* f) {
  auto all_vals = f->usedMathVals();

  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    os() << tv->toString();
    os() << "\n";
    printTransforms(tv);
  }
}

void IrTransformPrinter::printTransforms(const TensorView* tv) {
  const auto& logical_domain = tv->getLogicalDomain();
  if (tv->hasRoot()) {
    const auto& root_domain = tv->getRootDomain();
    os() << " root domain : (" << toDelimitedString(root_domain) << ")\n";

    const auto all_exp = DependencyCheck::getAllExprsBetween(
        {root_domain.begin(), root_domain.end()},
        {logical_domain.begin(), logical_domain.end()});

    for (const auto exp : all_exp) {
      os() << "  " << exp->toString();
    }
  }

  os() << " logical domain : (" << toDelimitedString(logical_domain) << ")\n";

  if (tv->hasAllocation()) {
    const auto& alloc_domain = tv->getAllocationDomain();

    os() << " allocation domain : (" << toDelimitedString(alloc_domain)
         << ")\n";
  }

  os() << " contiguity: " << tv->domain()->getContiguityString() << "\n";

  for (const auto exp : tv->domain()->allExprs()) {
    os() << "  " << exp->toString();
  }
  os() << " loop domain : (" << toDelimitedString(tv->getLoopDomain()) << ")\n";
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

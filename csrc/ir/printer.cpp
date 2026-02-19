// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "ir/printer.h"

#include "fusion.h"
#include "host_ir/container.h"
#include "instrumentation.h"
#include "ir/iostream.h"
#include "ir/utils.h"
#include "kernel_ir.h"

namespace nvfuser {

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

  // Print the definitions of the indices that are used in the host_ir_container
  if (hasDebugDumpArgument(DebugDumpOption::HostIr, "indices")) {
    os() << "Index definitions:\n";
    indent_size_++;
    for (Val* val : host_ir_container->vals()) {
      if (val->isScalar() && val->definition() != nullptr &&
          val->dtype() == DataType::Index) {
        os() << val->definition()->toString(indent_size_);
      }
    }
    indent_size_--;
    os() << "\n";
  }
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
    indent(os(), 1) << "root domain: (" << toDelimitedString(root_domain) << ")"
                    << std::endl;

    const auto all_exp = DependencyCheck::getAllExprsBetween(
        {root_domain.begin(), root_domain.end()},
        {logical_domain.begin(), logical_domain.end()});

    for (const auto exp : all_exp) {
      indent(os(), 2) << exp->toString();
    }
  }

  indent(os(), 1) << "logical domain: (" << toDelimitedString(logical_domain)
                  << ")" << std::endl;

  if (tv->hasAllocation()) {
    const auto& alloc_domain = tv->getAllocationDomain();

    indent(os(), 1) << "allocation domain: (" << toDelimitedString(alloc_domain)
                    << ")" << std::endl;
  }

  indent(os(), 1) << "contiguity: " << tv->domain()->getContiguityString()
                  << std::endl;

  for (const auto exp : tv->domain()->allExprs()) {
    indent(os(), 2) << exp->toString();
  }
  indent(os(), 1) << "loop domain: (" << toDelimitedString(tv->getLoopDomain())
                  << ")" << std::endl;
  if (tv->getAlternateLoopDomain().has_value()) {
    indent(os(), 1) << "alternate loop domain: ("
                    << toDelimitedString(tv->getAlternateLoopDomain().value())
                    << ")" << std::endl;
  }
}

} // namespace nvfuser

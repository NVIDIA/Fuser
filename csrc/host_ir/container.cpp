// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/host_ir.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>

namespace nvfuser {

namespace hir {

HostIrContainer::HostIrContainer(int64_t num_kernel_executors)
    : kernel_executors_(num_kernel_executors) {}

HostIrContainer::~HostIrContainer() = default;

Stream* HostIrContainer::getDefaultStream() {
  if (default_stream_ == nullptr) {
    default_stream_ = IrBuilder::createInContainer<Stream>(this);
  }
  return default_stream_;
}

std::ostream& HostIrContainer::print(std::ostream& os) const {
  IrMathPrinter op_exprs(os);
  op_exprs.handle(this);
  if (alias_.size() > 0) {
    os << "Aliases:{";
    for (const auto& alias : alias_) {
      os << "\n  " << alias.first << " -> " << alias.second;
    }
    os << "\n}\n";
  }
  return os;
}

const std::vector<Expr*>& HostIrContainer::topLevelExprs() const {
  return top_level_exprs_;
}

void HostIrContainer::pushBackTopLevelExprs(Expr* expr) {
  assertInContainer(expr, "Cannot add expr, ");
  top_level_exprs_.push_back(expr);
}

void HostIrContainer::setKernelExecutor(
    int64_t index,
    std::unique_ptr<KernelExecutor> ke) {
  NVF_ERROR(kernel_executors_.at(index) == nullptr);
  kernel_executors_.at(index) = std::move(ke);
}

KernelExecutor* HostIrContainer::getKernelExecutor(int64_t index) const {
  return kernel_executors_.at(index).get();
}

} // namespace hir

} // namespace nvfuser

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

HostIrContainer::HostIrContainer(int64_t num_groups)
    : kernel_executors_(num_groups) {}

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
  return os;
}

const std::vector<Expr*>& HostIrContainer::topLevelExprs() const {
  return top_level_exprs_;
}

void HostIrContainer::insertExprAfter(int64_t index, Expr* expr) {
  top_level_exprs_.insert(top_level_exprs_.begin() + index + 1, expr);
}

void HostIrContainer::pushBackTopLevelExprs(Expr* expr) {
  assertInContainer(expr, "Cannot add expr, ");
  top_level_exprs_.push_back(expr);
}

void HostIrContainer::addKernelExecutor(std::unique_ptr<KernelExecutor> ke) {
  const int64_t group_id = ke->groupId();
  NVF_ERROR(
      kernel_executors_.at(group_id) == nullptr,
      "KernelExecutor with the same group ID (",
      group_id,
      " already exists. You may have forgotten to KernelExecutor::setGroupId "
      "before calling HostIrContainer::addKernelExecutor.");
  kernel_executors_.at(group_id) = std::move(ke);
}

KernelExecutor* HostIrContainer::getKernelExecutor(
    const int64_t group_id) const {
  return kernel_executors_.at(group_id).get();
}

} // namespace hir

} // namespace nvfuser

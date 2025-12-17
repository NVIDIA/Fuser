// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "host_ir/container.h"

#include "host_ir/ir.h"
#include "ir/builder.h"
#include "ir/cloner.h"
#include "ir/printer.h"
#include "ir/utils.h"
#include "kernel_ir.h"
#include "ops/all_ops.h"
#include "runtime/executor.h"

namespace nvfuser {

namespace hir {

Stream* HostIrContainer::getDefaultStream() {
  if (default_stream_ == nullptr) {
    default_stream_ = IrBuilder::createInContainer<Stream>(this);
  }
  return default_stream_;
}

std::ostream& HostIrContainer::print(std::ostream& os) const {
  IrPrinter op_exprs(os);
  op_exprs.handle(this);
  return os;
}

void HostIrContainer::resetTopLevelExprs(std::list<Expr*> exprs) {
  top_level_.mutableExprs() = std::move(exprs);
}

void HostIrContainer::insertExprBefore(Scope::Iterator position, Expr* e) {
  top_level_.insert(position, e);
}

Scope::Iterator HostIrContainer::pushBackTopLevelExprs(Expr* e) {
  assertInContainer(e, "Cannot add expr, ");
  return top_level_.push_back(e);
}

bool HostIrContainer::hasKernelExecutor(int64_t group_id) const {
  return group_id < std::ssize(kernel_executors_) &&
      kernel_executors_.at(group_id) != nullptr;
}

void HostIrContainer::addKernelExecutor(std::unique_ptr<KernelExecutor> ke) {
  const int64_t group_id = ke->groupId();
  if (group_id >= std::ssize(kernel_executors_)) {
    kernel_executors_.resize(group_id + 1);
  }
  NVF_ERROR(
      kernel_executors_.at(group_id) == nullptr,
      "KernelExecutor with the same group ID (",
      group_id,
      " already exists. You may have forgotten to KernelExecutor::setGroupId "
      "before calling HostIrContainer::addKernelExecutor.");
  kernel_executors_.at(group_id) = std::move(ke);
}

KernelExecutor& HostIrContainer::getKernelExecutor(
    const int64_t group_id) const {
  NVF_CHECK(
      hasKernelExecutor(group_id),
      "KernelExecutor with group ID ",
      group_id,
      " not found.");
  return *kernel_executors_.at(group_id);
}

} // namespace hir

} // namespace nvfuser

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

Stream* HostIrContainer::getDefaultStream() {
  if (!default_stream_) {
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

void HostIrContainer::pushBackTopLevelExprs(Expr* expr) {
  assertInContainer(expr, "Cannot add expr, ");
  return top_level_exprs_.push_back(expr);
}

void HostIrContainer::pushBackKernelExecutor(
    std::unique_ptr<KernelExecutor> ke) {
  return kernel_executors_.push_back(std::move(ke));
}

KernelExecutor* HostIrContainer::getKernelExecutor(int64_t index) const {
  return kernel_executors_.at(index).get();
}

} // namespace hir

} // namespace nvfuser

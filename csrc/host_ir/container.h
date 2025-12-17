// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <list>

#include "fusion.h"
#include "host_ir/ir.h"
#include "ir/internal_nodes.h"
#include "runtime/executor.h"

namespace nvfuser::hir {

// HostIrContainer is used to represent a host program.
// 1) It inherits from Fusion, so that (Host) IRs can be registered to it.
// 2) It holds a list of Host Expressions `top_level_` that represent
// the host program.
class HostIrContainer final : public Fusion {
 public:
  HostIrContainer() = default;
  HostIrContainer(const HostIrContainer&) = delete;
  HostIrContainer& operator=(const HostIrContainer&) = delete;

  // Print to an output stream
  std::ostream& print(std::ostream& os) const;

  const Scope& topLevel() const {
    return top_level_;
  }
  Scope& topLevel() {
    return top_level_;
  }
  const Scope::ExprList& topLevelExprs() const {
    return topLevel().exprs();
  }

  // Appends `expr` and returns the iterator pointing to `expr`.
  Scope::Iterator pushBackTopLevelExprs(Expr* expr);
  void insertExprBefore(Scope::Iterator position, Expr* expr);
  // Only used for MultiDeviceExecutor. While convenient, it should generally
  // be avoided because it implicitly modifies `top_level_`, making the
  // code harder to reason about.
  void resetTopLevelExprs(std::list<Expr*> exprs);

  void addKernelExecutor(std::unique_ptr<KernelExecutor> ke);
  bool hasKernelExecutor(int64_t group_id) const;
  KernelExecutor& getKernelExecutor(int64_t group_id) const;

  Stream* getDefaultStream();

 private:
  Scope top_level_{/*owner=*/nullptr};

  // Indexed by group ID. This way, parallel compilation can write to disjoint
  // locations without having to precompute a global index.
  std::vector<std::unique_ptr<KernelExecutor>> kernel_executors_;

  Stream* default_stream_ = nullptr;
};

} // namespace nvfuser::hir

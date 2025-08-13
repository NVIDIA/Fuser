// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <host_ir/host_ir.h>
#include <runtime/executor.h>

namespace nvfuser {

namespace hir {

// HostIrContainer is used to represent a host program.
// 1) It inherits from Fusion, so that (Host) IRs can be resgistered to it.
// 2) It holds a vector of Host Expressions `top_level_exprs_` that represent
// the host program. For now, this vector is manually managed. Moreover, because
// we use a vector as data structure, top_level_exprs_ can only represent linear
// Host programs. Later, we it should support non-linear program having a DAG
// structure.
class HostIrContainer final : public Fusion {
 public:
  HostIrContainer() = default;
  HostIrContainer(const HostIrContainer&) = delete;
  HostIrContainer& operator=(const HostIrContainer&) = delete;

  //! Print to an output stream
  std::ostream& print(std::ostream& os) const;

  void resetTopLevelExprs(std::vector<Expr*> exprs) {
    top_level_exprs_ = std::move(exprs);
  }
  const std::vector<Expr*>& topLevelExprs() const;
  void pushBackTopLevelExprs(Expr* expr);
  void insertExprAfter(int64_t index, Expr* expr);

  void addKernelExecutor(std::unique_ptr<KernelExecutor> ke);
  bool hasKernelExecutor(int64_t group_id) const;
  KernelExecutor& getKernelExecutor(int64_t group_id) const;

  Stream* getDefaultStream();

 private:
  std::vector<Expr*> top_level_exprs_;
  // Indexed by group ID. This way, parallel compilation can write to disjoint
  // locations without having to precompute a global index.
  std::vector<std::unique_ptr<KernelExecutor>> kernel_executors_;
  Stream* default_stream_ = nullptr;
};

} // namespace hir

} // namespace nvfuser

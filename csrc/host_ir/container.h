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
  // num_kernel_executors is only needed when the container has LaunchKernel
  // instructions.
  explicit HostIrContainer(int64_t num_kernel_executors = 0);
  HostIrContainer(const HostIrContainer&) = delete;
  HostIrContainer& operator=(const HostIrContainer&) = delete;

  // Do not have a definition here as it requires the definition of
  // KernelExecutor due to kernel_executors_.
  // NOLINTNEXTLINE (modernize-use-equals-default)
  ~HostIrContainer() override;

  //! Print to an output stream
  std::ostream& print(std::ostream& os) const;

  void resetTopLevelExprs(std::vector<Expr*> exprs) {
    top_level_exprs_ = std::move(exprs);
  }

  const std::vector<Expr*>& topLevelExprs() const;

  void pushBackTopLevelExprs(Expr* expr);

  void insertExprAfter(int64_t index, Expr* expr);

  void setKernelExecutor(int64_t index, std::unique_ptr<KernelExecutor> ke);

  bool hasKernelExecutor(int64_t index) const {
    return kernel_executors_.at(index) != nullptr;
  }

  KernelExecutor* getKernelExecutor(int64_t index) const;

  Stream* getDefaultStream();

  void markAlias(TensorView* original, const TensorView* new_alias) {
    while (alias_.count(original)) {
      original = alias_[original]->as<TensorView>();
    }
    alias_[new_alias] = original;
  }

  const auto& alias() const {
    return alias_;
  }

 private:
  std::vector<Expr*> top_level_exprs_;
  std::vector<std::unique_ptr<KernelExecutor>> kernel_executors_;
  Stream* default_stream_ = nullptr;
  std::unordered_map<const Val*, Val*> alias_;
};

} // namespace hir

} // namespace nvfuser

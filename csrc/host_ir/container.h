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

namespace nvfuser {

class KernelExecutor;

namespace hir {

/*
HostIrContainer is used to represent a host program.
1) It inherits from Fusion, so that (Host) IRs can be resgistered to it.
2) It holds a vector of Host Expressions `top_level_exprs_` that represent the
host program. For now, this vector is manually managed. Moreover, because we use
a vector as data structure, top_level_exprs_ can only represent linear Host
programs. Later, we it should support non-linear program having a DAG structure.
*/

class HostIrContainer final : public Fusion {
 public:
  HostIrContainer() = default;
  HostIrContainer(const HostIrContainer&) = delete;
  HostIrContainer& operator=(const HostIrContainer&) = delete;

  // Do not have a definition here as it requires the definition of
  // KernelExecutor due to kernel_executors_.
  // NOLINTNEXTLINE (modernize-use-equals-default)
  ~HostIrContainer() override;

  //! Print to an output stream
  std::ostream& print(std::ostream& os) const;

  const std::vector<Expr*>& topLevelExprs() const;

  void pushBackTopLevelExprs(Expr* expr);

  void pushBackKernelExecutor(std::unique_ptr<KernelExecutor> ke);

  KernelExecutor* getKernelExecutor(int64_t index) const;

  Stream* getDefaultStream();

 private:
  std::vector<Expr*> top_level_exprs_;
  std::vector<std::unique_ptr<KernelExecutor>> kernel_executors_;
  Stream* default_stream_ = nullptr;
};

} // namespace hir

} // namespace nvfuser

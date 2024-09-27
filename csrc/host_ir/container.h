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

  //! Print to an output stream
  std::ostream& print(std::ostream& os) const;

  const auto& topLevelExprs() const {
    return top_level_exprs_;
  }

  void pushBackTopLevelExprs(Expr* expr) {
    assertInContainer(expr, "Cannot add expr, ");
    return top_level_exprs_.push_back(expr);
  }

  Stream* getDefaultStream();

 private:
  std::vector<Expr*> top_level_exprs_;
  Stream* default_stream_ = nullptr;
};

} // namespace hir

} // namespace nvfuser

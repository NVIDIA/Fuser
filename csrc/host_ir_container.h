// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>

namespace nvfuser {

namespace hir {

class HostIrContainer final : public Fusion {
 public:
  HostIrContainer() = default;
  HostIrContainer(const HostIrContainer&) = delete;
  HostIrContainer& operator=(const HostIrContainer&) = delete;

  //! Print to an output stream
  std::ostream& print(
      std::ostream& os,
      bool include_tensor_transforms = false,
      int indent_size = 0) const;

  const auto& topLevelExprs() const {
    return top_level_exprs;
  }

  std::vector<Expr*> top_level_exprs;
};

} // namespace hir

} // namespace nvfuser

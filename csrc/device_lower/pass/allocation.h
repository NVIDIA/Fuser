// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <ir/all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

//! Buffer allocation information to store in GPU lower to avoid
//!  logic duplication
struct LocalAllocationInfo {
  kir::Allocate* alloc_expr = nullptr;
  std::vector<IterDomain*> alloc_domains;
  bool has_halo = false;
};

using LocalAllocationInfoMap = std::
    unordered_map<const kir::Allocate*, std::unique_ptr<LocalAllocationInfo>>;

//! Insert buffer allocations
std::vector<Expr*> insertAllocations(const std::vector<Expr*>& exprs);

} // namespace nvfuser

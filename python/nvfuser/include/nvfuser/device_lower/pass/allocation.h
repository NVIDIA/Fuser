// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>

#include <ir/all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

struct AllocationDomainInfo {
  std::vector<IterDomain*> ids;
  std::vector<Val*> strides;
  std::vector<bool> contiguity;
};

//! Buffer allocation information to store in GPU lower to avoid
//!  logic duplication
struct LocalAllocationInfo {
  kir::Allocate* alloc_expr = nullptr;
  std::vector<IterDomain*> alloc_domains;
};

using LocalAllocationInfoMap = std::
    unordered_map<const kir::Allocate*, std::unique_ptr<LocalAllocationInfo>>;

//! Insert buffer allocations
std::vector<Expr*> insertAllocations(const std::vector<Expr*>& exprs);

} // namespace nvfuser

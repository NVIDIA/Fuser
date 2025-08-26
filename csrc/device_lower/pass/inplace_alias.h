// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

namespace nvfuser {

class Expr;

// This lowering pass creates aliasing relationships between
// kir::Allocate nodes. Currently, this is only designed for the
// inplace scatter op.
//
// More specifically, it first gathers all the information by scanning
// the given Kernel, including mappings of tensors to their kir::Allocate
// nodes, grouping of tensors that alias each other. It is used then
// to mutate Kernel such that all tensors in the same alias group
// points to the same actual allocation. This is done by updating
// kir::Allocate's alias attribute.
//
// The selection of the actual allocation for each tensor group is
// done by just picking the first tensor, except when the group has a
// fusion output, in which case the output is selected.
std::vector<Expr*> setInplaceAlias(const std::vector<Expr*>& exprs);

} // namespace nvfuser

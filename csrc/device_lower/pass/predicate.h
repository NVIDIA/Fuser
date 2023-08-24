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

//! Update predicates with valid bool conditionals
//!
std::vector<Expr*> generateConditionalFromPredicate(
    const std::vector<Expr*>& exprs);

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <ir/all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

//! Detect ReductionOps that have serialGridReductionRequested() == true. When
//! found, confirm that no conflicting operations exist, then place sync nodes
//! before and after outer-most non-parallelized loop.
std::vector<Expr*> insertGridSerializationSyncs(
    const std::vector<Expr*>& exprs);

} // namespace nvfuser

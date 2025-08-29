// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

class GpuLower;

//! Pass to convert ReductionOp to ClusterReductionOp after allocation
//! This ensures mbarriers are available during ClusterReductionOp creation
std::vector<Expr*> convertToClusterReduction(const std::vector<Expr*>& exprs);

} // namespace nvfuser

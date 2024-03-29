// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dispatch.h>
#include <fusion.h>
#include <ir/all_nodes.h>

#include <vector>

namespace nvfuser {

// Transpose, Shift, Gather, and View Ops with LoadStoreOps
std::vector<Expr*> loadStoreOpInserter(const std::vector<Expr*>& exprs);

} // namespace nvfuser

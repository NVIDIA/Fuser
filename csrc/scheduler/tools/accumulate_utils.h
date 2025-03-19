// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <val_graph.h>

namespace nvfuser {

class Expr;
class TensorView;

namespace scheduler_tools {

bool isAccumulationBasedOp(Expr* expr);

bool hasAccumulationBasedOps(Fusion* fusion);

std::vector<Expr*> getAccumulationBasedOps(Fusion* fusion);

} // namespace scheduler_tools
} // namespace nvfuser

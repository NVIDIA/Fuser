// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

namespace nvfuser {

class Expr;
class Fusion;

namespace scheduler_tools {

void propagateSqueezedSliceToOutputs(Fusion* fusion);

void propagateResizeTensorOpToInputs(Expr* resize_op);

} // namespace scheduler_tools
} // namespace nvfuser

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

// If a size-one slice ID is squeezed, add a broadcast loop ID to all
// tensors that would have a corresponding ID if not squeezed. This
// simplifies scheduling by making the fusion more uniform.
void propagateSqueezedSliceToOutputs(Fusion* fusion);

void propagateResizeTensorOpToInputs(Expr* resize_op);

} // namespace scheduler_tools
} // namespace nvfuser

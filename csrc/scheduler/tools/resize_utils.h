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
class IterDomain;

namespace scheduler_tools {

std::vector<IterDomain*> getSqueezedSlices(Fusion* fusion);

// If a size-one slice ID is squeezed, add a broadcast loop ID to all
// tensors that would have a corresponding ID if not squeezed. This
// simplifies scheduling by making the fusion more uniform.
void propagateSqueezedSliceToOutputs(Fusion* fusion);

void propagateResizeToInputs(Expr* resize_op);

} // namespace scheduler_tools
} // namespace nvfuser

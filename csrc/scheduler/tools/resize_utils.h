// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class Expr;

namespace scheduler_tools {

// For a given resize-based op such as slice and pad, make the loop
// domain of each depedent producer tensor exact-mapped by propagating
// the iter-domain ops of the output tensor of the given op. Note that
// fusion inputs are skipped as their loop domains don't matter.
void propagateResizeToInputs(Expr* resize_op);

} // namespace scheduler_tools
} // namespace nvfuser

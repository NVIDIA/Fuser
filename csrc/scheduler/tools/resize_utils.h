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

class CatOp;
class Fusion;
class SliceOp;

namespace scheduler_tools {

void propagateCatToInputs(CatOp* cat_op);
bool propagateCatToInputs(Fusion* fusion);

void propagateSliceToInputs(SliceOp* slice_op);
bool propagateSliceToInputs(Fusion* fusion);

bool propagateSliceToOutputs(Fusion* fusion);

bool propagateSqueezedSliceToOutputs(Fusion* fusion);

} // namespace scheduler_tools
} // namespace nvfuser

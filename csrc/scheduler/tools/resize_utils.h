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

namespace scheduler_tools {

std::vector<CatOp*> getRepresentativeCatOps(Fusion* fusion);

bool propagateResizeToCatInputs(CatOp* cat_op);

} // namespace scheduler_tools
} // namespace nvfuser

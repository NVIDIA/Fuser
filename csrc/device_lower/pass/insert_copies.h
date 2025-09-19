// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

namespace nvfuser {

class Fusion;
class Expr;

void copyInputsAndOutputsOfExternFuncs(Fusion* fusion);

std::vector<Expr*> prepareInputsForGroupedFuncs(const std::vector<Expr*>& exprs);

} // namespace nvfuser

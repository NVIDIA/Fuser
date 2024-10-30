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

// Change
//   if (cond1) {
//     expr1;
//   }
//   if (cond1) {
//     expr2;
//   }
// to
//   if (cond1) {
//     expr1;
//     expr2;
//   }
std::vector<Expr*> mergeIfThenElse(const std::vector<Expr*>& exprs);
    
} // namespace nvfuser

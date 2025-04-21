// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>

namespace nvfuser {

std::vector<Expr*> lowerToInlinePtx(const std::vector<Expr*>& exprs);

} // namespace nvfuser

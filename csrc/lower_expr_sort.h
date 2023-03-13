/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ir_base_nodes.h>

namespace nvfuser {

std::vector<Expr*> reorderExprsForComputeAt();

} // namespace nvfuser

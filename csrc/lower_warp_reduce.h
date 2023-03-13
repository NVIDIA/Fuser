// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <kernel_ir.h>

namespace nvfuser {

struct WarpPaddedParallelInfo {
  bool is_tidx_padded = false;
  bool is_tidx_single_warp = false;
  bool has_warp_reduction = false;
};

std::vector<Expr*> fuseWarpReduce(const std::vector<Expr*> exprs);

} // namespace nvfuser

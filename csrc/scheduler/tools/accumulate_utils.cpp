// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/tools/accumulate_utils.h>

namespace nvfuser::scheduler_tools {

bool isResizeBasedOp(Expr* expr) {
  return expr->isOneOf<IndexPutAccumulateOp>();
}

bool hasResizeBasedOps(Fusion* fusion) {
  return ir_utils::hasOpsOfType<IndexPutAccumulateOp>(fusion);
}

std::vector<Expr*> getResizeBasedOps(Fusion* fusion) {
  return ir_utils::getOpsOfType<IndexPutAccumulateOp>(fusion);
}

} // namespace nvfuser::scheduler_tools

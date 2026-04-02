// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/transpose_tma.h"

namespace nvfuser {
namespace transpose {
namespace tma {

std::unique_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  // TMA transpose scheduling is not yet implemented.
  return nullptr;
}

void scheduleTranspose(Fusion* fusion, const TransposeParams* tparams) {
  NVF_THROW("TMA transpose scheduling is not yet implemented.");
}

} // namespace tma
} // namespace transpose
} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/analysis/tensor_memory.h>
#include <fusion.h>
#include <ir/all_nodes.h>

namespace nvfuser {

TensorMemoryInfo computeTMemInfo(Fusion* fusion) {
  bool found = false;
  for (auto tv : fusion->allTvs()) {
    if (tv->getMemoryType() == MemoryType::Tensor) {
      NVF_ERROR(!found, "Only one tensor on TMem is supported");
      found = true;
    }
  }
  return {};
}

} // namespace nvfuser

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
#include <type.h>

namespace nvfuser {

TensorMemoryInfo computeTMemInfo(Fusion* fusion) {
  bool found = false;
  for (auto tv : fusion->allTvs()) {
    if (tv->getMemoryType() == MemoryType::Tensor) {
      NVF_ERROR(!found, "Only one tensor on TMem is supported");
      found = true;
    }
  }

  if (found) {
    // tcgen05.alloc stores the allocated address in shared memory. So we use a
    // TensorView with MemoryType::Shared to store this address.
    auto allocation_address = TensorViewBuilder()
                                  .shape(std::vector<Val*>{})
                                  .dtype(DataType::UInt32)
                                  .build();
    allocation_address->setMemoryType(MemoryType::Shared);
    return {allocation_address};
  }

  return {nullptr};
}

} // namespace nvfuser

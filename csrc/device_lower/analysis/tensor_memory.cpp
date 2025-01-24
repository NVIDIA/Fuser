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
  TensorMemoryInfo result;

  // Compute the allocation information for tensor memory. Currently, we use a
  // very simple heuristic that assign a separate region for each TensorView.
  // See note [Tensor Memory Allocation] for the overall design.
  auto& regions = result.allocation.regions;
  for (auto tv : fusion->allTvs()) {
    if (tv->getMemoryType() != MemoryType::Tensor) {
      continue;
    }
    regions.emplace_back();
    auto& region = regions.back();

    region.address = TensorViewBuilder()
                         .shape(std::vector<Val*>{})
                         .dtype(DataType::UInt32)
                         .build();
    region.address->setMemoryType(MemoryType::Shared);

    // TODO: right now we hardcode the number of columns to be 32. This is
    // definitely not correct.
    region.num_columns = IrBuilder::create<Val>(32, DataType::UInt32);

    region.covered_tensors.emplace_back();
    auto& covered_tensor = region->covered_tensors.back();
    covered_tensor.tensor = tv;
    covered_tensor.lane_offset = tv->fusion()->zeroVal(DataType::UInt16);
    covered_tensor.column_offset = tv->fusion()->zeroVal(DataType::UInt16);
  }

  return result;
}

} // namespace nvfuser

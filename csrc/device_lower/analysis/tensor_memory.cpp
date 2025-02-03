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

// See note [Tensor Memory Allocation] for the overall design.
TensorMemoryInfo computeTMemInfo(Fusion* fusion) {
  TensorMemoryInfo result;

  // Step 1: partition the tensors. Each partition of tensors will become a
  // region, so we use the term partition and region interchangeably. The user
  // may have provided full or partial partitioning information. For the
  // TensorViews that the user has already specified which region they belong
  // to, we will use that information. For the rest of the tensors, we will
  // assign each of them to a separate region.
  using Partition = std::vector<std::vector<TensorView*>>;
  Partition partitions;
  if (fusion->hasManaged("tmem_regions")) {
    partitions = fusion->getManaged<Partition>("tmem_regions");
  } else {
    partitions = {};
  }

  // Verify that there is no overlap between user specified partitions
  std::unordered_set<TensorView*> tensors;
  for (auto& partition : partitions) {
    NVF_ERROR(!partition.empty(), "Empty partition");
    for (auto tv : partition) {
      NVF_ERROR(
          tv->getMemoryType() == MemoryType::Tensor, "Invalid memory type");
      NVF_ERROR(
          tensors.insert(tv).second, "Tensors cannot be in multiple regions");
    }
  }

  // For all TensorViews whose partition is not specified, assign them to a
  // separate region.
  for (auto tv : fusion->allTvs()) {
    if (tv->getMemoryType() != MemoryType::Tensor) {
      continue;
    }
    if (tensors.count(tv) == 0) {
      partitions.push_back({tv});
    }
  }

  // Step 2: Compute the allocation information for tensor memory. That is, for
  // each partition, we create a Region object and fill in the necessary
  // information.
  using Region = TMemAlllocationInfo::Region;
  std::vector<Region>& regions = result.allocation.regions;
  for (const auto& partition : partitions) {
    regions.emplace_back();
    auto& region = regions.back();

    // tcgen05.alloc stores the allocated address in shared memory. So we use a
    // TensorView with MemoryType::Shared to store this address.
    region.address = TensorViewBuilder()
                         .shape(std::vector<Val*>{})
                         .dtype(DataType::UInt32)
                         .build();
    region.address->setMemoryType(MemoryType::Shared);

    // Assign each tensor in the region a whole 128 lanes and N columns.
    region.num_columns = region.address->fusion()->zeroVal(DataType::UInt16);
    for (auto tv : partition) {
      // TODO: right now we hardcode the number of columns of each tensor to
      // be 32. This is definitely not correct.
      Val* num_columns = IrBuilder::create<Val>(32, DataType::UInt16);
      region.covered_tensors.emplace_back();
      auto& covered_tensor = region.covered_tensors.back();
      covered_tensor.tensor = tv;
      covered_tensor.lane_offset = tv->fusion()->zeroVal(DataType::UInt16);
      covered_tensor.column_offset = region.num_columns;
      region.num_columns =
          SimplifyingIrBuilder::addExpr(region.num_columns, num_columns);
    }
    region.num_columns =
        IrBuilder::maybeCastExpr(DataType::UInt32, region.num_columns);
  }

  return result;
}

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/analysis/tensor_memory.h>
#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <type.h>

#include <utility>
#include <vector>

namespace nvfuser {

// Returns the lane and column allocation domain that is actually allocated.
std::pair<std::vector<IterDomain*>, std::vector<IterDomain*>> getTMemAllocation(
    TensorView* tv) {
  NVF_ERROR(tv->getMemoryType() == MemoryType::Tensor);
  // Get all compute-at IDs
  const auto& loop_domain = tv->getLoopDomain();
  std::unordered_set<IterDomain*> ca_ids(
      loop_domain.begin(), loop_domain.begin() + tv->getComputeAtPosition());
  // Filter allocation domain
  std::vector<IterDomain*> lane;
  std::vector<IterDomain*> column;
  const auto& raw_allocation_domain = tv->getMaybeAllocationDomain();
  const int64_t dimsep = tv->getTMemDimSepPos();
  for (int64_t i : c10::irange((int64_t)raw_allocation_domain.size())) {
    std::vector<IterDomain*>& target = i < dimsep ? lane : column;
    IterDomain* id = raw_allocation_domain[i];
    ParallelType p_type = id->getParallelType();
    if (ir_utils::isMemorySharedAcross(MemoryType::Tensor, p_type)) {
      target.push_back(id);
      continue;
    }
    if (id->isBroadcast() || id->isReduction() || id->extent()->isOneInt() ||
        ir_utils::isMemoryPartitionedAcross(MemoryType::Tensor, p_type) ||
        ca_ids.count(id)) {
      continue;
    }
    target.push_back(id);
  }
  return {lane, column};
}

Val* productOfExtents(const std::vector<IterDomain*>& domain) {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (domain.empty()) {
    return fusion->oneVal();
  }
  Val* product = fusion->oneVal();
  for (IterDomain* id : domain) {
    product = SimplifyingIrBuilder::mulExpr(product, id->extent());
  }
  return product;
}

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
  Val* total_num_columns = fusion->zeroVal();
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
    region.num_columns = fusion->zeroVal();
    for (auto tv : partition) {
      region.covered_tensors.emplace_back();
      auto& covered_tensor = region.covered_tensors.back();
      covered_tensor.tensor = tv;
      std::tie(
          covered_tensor.lane_allocation, covered_tensor.column_allocation) =
          getTMemAllocation(tv);
      Val* num_columns = productOfExtents(covered_tensor.column_allocation);
      covered_tensor.lane_offset = tv->fusion()->zeroVal(DataType::UInt16);
      covered_tensor.column_offset =
          IrBuilder::maybeCastExpr(DataType::UInt16, region.num_columns);
      region.num_columns =
          SimplifyingIrBuilder::addExpr(region.num_columns, num_columns);

      // Validate lane allocation
      Val* num_lanes = productOfExtents(covered_tensor.lane_allocation);
      constexpr int64_t max_lanes = 128;
      Val* max_lanes_val = IrBuilder::create<Val>(max_lanes);
      GpuLower::current()->validate(
          SimplifyingIrBuilder::leExpr(num_lanes, max_lanes_val),
          "Not enough tensor memory lanes: tried to allocate ",
          num_lanes->toInlineString(),
          ", but only ",
          max_lanes,
          " available.");
    }
    constexpr int64_t unit_of_allocation = 512;
    Val* unit_of_allocation_val = IrBuilder::create<Val>(unit_of_allocation);
    region.num_columns = SimplifyingIrBuilder::maxExpr(
        unit_of_allocation_val, region.num_columns);
    total_num_columns =
        SimplifyingIrBuilder::addExpr(total_num_columns, region.num_columns);
    region.num_columns =
        IrBuilder::maybeCastExpr(DataType::UInt32, region.num_columns);
  }
  constexpr int64_t max_columns = 512;
  Val* max_columns_val = IrBuilder::create<Val>(max_columns);
  GpuLower::current()->validate(
      SimplifyingIrBuilder::leExpr(total_num_columns, max_columns_val),
      "Not enough tensor memory columns: tried to allocate ",
      total_num_columns->toInlineString(),
      ", but only ",
      max_columns,
      " available.");

  return result;
}

} // namespace nvfuser

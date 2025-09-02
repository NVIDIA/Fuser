// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/analysis/tensor_memory.h>
#include <device_lower/lower2device.h>
#include <expr_simplifier.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <options.h>
#include <scheduler/tools/abstract_tensor.h>
#include <type.h>
#include <utils.h>

#include <ranges>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nvfuser {

const TMemAlllocationInfo::Region::TVInfo& TMemAlllocationInfo::getTVInfo(
    TensorView* tv) const {
  for (const auto& region : regions) {
    for (const auto& tv_info : region.covered_tensors) {
      if (tv_info.tensor == tv) {
        return tv_info;
      }
    }
  }
  NVF_ERROR(false, "TensorView not found in TMemAlllocationInfo");
}

namespace {

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
  for (int64_t i : arange((int64_t)raw_allocation_domain.size())) {
    std::vector<IterDomain*>& target = i < dimsep ? lane : column;
    IterDomain* id = raw_allocation_domain[i];
    ParallelType p_type = id->getParallelType();
    if (id->isBroadcast() || id->isReduction() || id->extent()->isOneInt()) {
      continue;
    }
    if (ir_utils::isMemorySharedAcross(MemoryType::Tensor, p_type)) {
      target.push_back(id);
      continue;
    }
    if (ir_utils::isMemoryPartitionedAcross(MemoryType::Tensor, p_type) ||
        ca_ids.count(id)) {
      continue;
    }
    target.push_back(id);
  }
  return {std::move(lane), std::move(column)};
}

Val* productOfExtents(const std::vector<IterDomain*>& domain) {
  Fusion* fusion = FusionGuard::getCurFusion();
  Val* product = fusion->oneVal();
  for (IterDomain* id : domain) {
    product = SimplifyingIrBuilder::mulExpr(product, id->extent());
  }
  return product;
}

// See note [Tensor Memory Allocation] for the overall design.
TMemAlllocationInfo computeTMemAlllocationInfo(Fusion* fusion) {
  TMemAlllocationInfo result;

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

  // Validate the number of columns. There is at most 512 columns.
  auto validate_columns = [](Val* num_columns) {
    constexpr int64_t max_columns = 512;
    Val* max_columns_val = IrBuilder::create<Val>(max_columns);
    NVFUSER_LOWER_VALIDATE(
        SimplifyingIrBuilder::leExpr(num_columns, max_columns_val),
        "Not enough tensor memory columns: tried to allocate ",
        num_columns->toInlineString(),
        ", but only ",
        max_columns,
        " available.");
  };

  Val* total_num_columns = fusion->zeroVal();
  using Region = TMemAlllocationInfo::Region;
  std::vector<Region>& regions = result.regions;
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
      // Each column is 4 bytes.
      Val* num_columns = SimplifyingIrBuilder::ceilDivExpr(
          SimplifyingIrBuilder::mulExpr(
              productOfExtents(covered_tensor.column_allocation),
              IrBuilder::create<Val>(dataTypeSizeByte(tv->dtype()))),
          IrBuilder::create<Val>(4));
      covered_tensor.lane_offset = tv->fusion()->zeroVal(DataType::UInt16);
      covered_tensor.column_offset =
          IrBuilder::maybeCastExpr(DataType::UInt16, region.num_columns);
      region.num_columns =
          SimplifyingIrBuilder::addExpr(region.num_columns, num_columns);

      // Validate lane allocation
      Val* num_lanes = productOfExtents(covered_tensor.lane_allocation);
      constexpr int64_t max_lanes = 128;
      Val* max_lanes_val = IrBuilder::create<Val>(max_lanes);
      NVFUSER_LOWER_VALIDATE(
          SimplifyingIrBuilder::leExpr(num_lanes, max_lanes_val),
          "Not enough tensor memory lanes: tried to allocate ",
          num_lanes->toInlineString(),
          ", but only ",
          max_lanes,
          " available.");
    }

    // Validate region.num_columns before rounding up for better error message
    validate_columns(region.num_columns);

    // Number of columns must be a power of 2 with a minimum of 32.
    constexpr int64_t unit_of_allocation = 32;
    Val* unit_of_allocation_val =
        IrBuilder::create<Val>(unit_of_allocation, DataType::UInt32);
    region.num_columns =
        IrBuilder::maybeCastExpr(DataType::UInt32, region.num_columns);
    region.num_columns = SimplifyingIrBuilder::maxExpr(
        unit_of_allocation_val, IrBuilder::bitCeilExpr(region.num_columns));
    total_num_columns =
        SimplifyingIrBuilder::addExpr(total_num_columns, region.num_columns);
  }
  validate_columns(total_num_columns);
  return result;
}

// Get the TID Parallel types that are not trivial. We are not interested
// in the parallel types that are not used in the kernel, and the ones that have
// size 1. The order of the returned parallel types is from z to x.
std::vector<ParallelType> getNonTrivialActiveThreadParallelTypes(
    Fusion* fusion) {
  const auto& pdim_map = GpuLower::current()->info().parallelDimensionMap();
  std::vector<ParallelType> nontrivial_tid_ptypes;
  for (auto pt : std::views::reverse(kParallelTypeTIDs)) {
    Val* size = pdim_map.getRaw(pt);
    if (size == nullptr) {
      continue;
    }
    Val* size_is_one =
        simplifyExpr(SimplifyingIrBuilder::eqExpr(size, fusion->oneVal()));
    if (size_is_one->isTrue()) {
      continue;
    }
    nontrivial_tid_ptypes.push_back(pt);
  }

  NVF_CHECK(
      !nontrivial_tid_ptypes.empty(),
      "Invalid data access pattern in TMem load/store: ",
      "TMem load/store must be warp-collective, but CTA size is one.");
  return nontrivial_tid_ptypes;
}

// Get the [TIDz, TIDy, TIDx] projected to the given expression as ValGroups,
// and merge them by contiguity. If any of the TIDz, TIDy, TIDx is not
// interested (see above), we just ignore it. Return the merged ValGroups as an
// AbstractTensor, and the strides of these ValGroups.
//
// Why do we need this function?
//
// In the CUDA programming model, each CTA has TIDx, TIDy, and TIDz.
// Unfortunately, the mapping of these TIDs to hardware concepts like warp, warp
// group, are not clear and depend on the kernel launch configuration. Here, we
// try to not assume anything like "TIDx must be a multiple of 32", but still,
// we must be able to validate and pattern match the data access of the tensor
// memory load/store.
//
// We need to construct a ValGroup that represents "warp" for an expression from
// its consumer's loop domain. Naively speaking, it is just:
//   split(Iz * Iy * Ix, 32).inner
// where Iz, Iy, Ix are the IterDomains in the loop domain that are parallelized
// on TIDz, TIDy and TIDx. But unfortunately, in reality, it is not that simple.
// NVFuser allows parallelizating IterDomains in an inexact way, for example, if
// the kernel's parallel dimension size for TIDx is 8, then the IterDomain being
// parallelized with TIDx (Ix) does not have to be exactly 8. This inexactness
// is especially common in warp-specialized kernels. If, for example, the TIDx
// parallelized IterDomain (Ix) in the loop domain is not exact (for example,
// have extent 7), then
//   split(Iz * Iy * Ix, 32).inner
// may not be the warp. To handle this, we need to create a new concept
// "contiguity of thread parallelized IterDomains in the loop domain". We can
// represent warp as
//   split(Iz * Iy * Ix, 32).inner
// if and only if Iz and Iy are contiguous. If Iz is not contiguous but Iy is,
// then warp would be:
//   split(Iy * Ix, 32).inner
// If neither Iz nor Iy is contiguous, then warp would be:
//   split(Ix, 32).inner
// Another way to think about the contiguity as described above is to consider
// all threads in the CTA as a 3D lattice, and we are drawing a 3d box from the
// origin to some point in the lattice. The length of the edges of the box are
// Ix.extent, Iy.extent, and Iz.extent. The contiguity of Ix, Iy, Iz are just
// like considering the lattice as a 3D tensor, and the box as a slice of the
// tensor.
//
// The strides returned are the stride of each edge of the box in that 3D CTA
// lattice. For example, if the parallel dimension sizes of the kernel are:
//   TIDz: 32, TIDy: 8, TIDx: 8
// and the loop domain is:
//   I0: TIDz, extent 32
//   I1: TIDy, extent 7
//   I2: TIDx, extent 8
// then the strides are [8*8, 8, 1].
std::pair<AbstractTensor, std::vector<Val*>>
getThreadParallelTypesMergedByContiguity(const Expr* expr) {
  auto& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
  auto nontrivial_tid_ptypes =
      getNonTrivialActiveThreadParallelTypes(expr->fusion());
  const auto& loop_domain = ir_utils::getTvOutput(expr)->getLoopDomain();
  const auto& pdim_map = GpuLower::current()->info().parallelDimensionMap();
  // Get the contiguity of nontrivial_tid_ptypes in the loop domain as described
  // above. The contiguity of each item in nontrivial_tid_ptypes can be computed
  // as follows:
  // - The innermost parallel type of nontrivial_tid_ptypes is always
  // contiguous.
  // - The item at index i is contiguous if the item at index i+1 is
  //   exact (its extent in the loop domain is the same as parallel
  //   dimension size of the kernel).
  // For example, if the parallel dimension sizes of the kernel are:
  //   TIDz: 32, TIDy: 8, TIDx: 8
  // and the loop domain is:
  //   I0: TIDz, extent 32
  //   I1: TIDy, extent 7
  //   I2: TIDx, extent 8
  // then the contiguity of TIDz, TIDy, TIDx in the loop domain is:
  //   TIDz: false, TIDy: true, TIDx: true
  // A true contiguity of I1 with respect to I2 means that I1 and I2 can be
  // merged
  std::vector<bool> contiguity;
  contiguity.reserve(nontrivial_tid_ptypes.size());
  bool prev_exact = true;
  for (ParallelType pt : std::views::reverse(nontrivial_tid_ptypes)) {
    contiguity.push_back(prev_exact);
    // Update prev_exact
    if (pdim_map.isExact(pt)) {
      // If the parallel dimension map says exact, then all IDs with this
      // parallel type have the same extent, so we can skip the equality check
      // below.
      prev_exact = true;
      continue;
    }
    // If the parallel dimension map does not say exact, then pt could still
    // be exact in this loop domain if the corresponding ID's extent is the
    // same as the parallel dimension size of the kernel.
    Val* pt_extent = pdim_map.getRaw(pt);
    auto pt_in_loop_domain_it = std::find_if(
        loop_domain.begin(), loop_domain.end(), [pt](IterDomain* id) {
          return id->getParallelType() == pt;
        });
    if (pt_in_loop_domain_it == loop_domain.end()) {
      prev_exact = false;
      continue;
    }
    IterDomain* pt_in_loop_domain = *pt_in_loop_domain_it;
    Val* extent_in_loop_domain = pt_in_loop_domain->extent();
    // If we can not symbolically prove that the extents are the same, then
    // we assume that they are not the same.
    prev_exact = simplifyExpr(SimplifyingIrBuilder::eqExpr(
                                  pt_extent, extent_in_loop_domain))
                     ->isTrue();
  }
  std::reverse(contiguity.begin(), contiguity.end());

  // Grab ValGroups for each parallel type from loop domain and store it in
  // AbstractTensor
  struct ContiguityAndStride {
    bool contiguity;
    Val* stride;
    static ContiguityAndStride merge(
        ContiguityAndStride x,
        ContiguityAndStride y) {
      NVF_ERROR(x.contiguity);
      return {y.contiguity, y.stride};
    }
    static std::pair<ContiguityAndStride, ContiguityAndStride> split(
        ContiguityAndStride x) {
      NVF_THROW("Should not reach here");
    }
    static std::pair<ContiguityAndStride, ContiguityAndStride> swizzle(
        ContiguityAndStride x,
        ContiguityAndStride y) {
      NVF_THROW("Should not reach here");
    }
  };
  AbstractTensorWithInfo<ContiguityAndStride> pdims;
  Val* stride = expr->fusion()->oneVal();
  for (auto [i, pt] : enumerate(nontrivial_tid_ptypes) | std::views::reverse) {
    Val* pdim_size = pdim_map.getRaw(pt);
    auto id_it = std::find_if(
        loop_domain.begin(), loop_domain.end(), [pt](IterDomain* id) {
          // NOLINTNEXTLINE
          return id->getParallelType() == pt;
        });
    if (id_it == loop_domain.end()) {
      stride = SimplifyingIrBuilder::mulExpr(stride, pdim_size);
      continue;
    }
    IterDomain* id = *id_it;
    const ValGroup& val_group = id_graph.toGroup(id);
    pdims.pushBack(
        ValGroupAndItsGraph{val_group, &id_graph},
        ContiguityAndStride{contiguity[i], stride});
    stride = SimplifyingIrBuilder::mulExpr(stride, pdim_size);
  }
  pdims.reverse();
  // Merge contiguous parallel types
  for (int64_t index = 0; index < (int64_t)pdims.size() - 1;) {
    if (pdims.info(index).contiguity) {
      pdims.merge(index);
    } else {
      index++;
    }
  }

  std::vector<Val*> strides;
  strides.reserve(pdims.size());
  for (auto pdim : pdims.domainAndInfo()) {
    strides.push_back(pdim.second.stride);
  }

  return {pdims.dropInfo(), strides};
}

// Infer the data path of TMem load/store operations from the loop domain of
// the consumer and the allocation domain of the TMem tensor. Based on the
// parallelization of the loop domain and the IterDomain transformations between
// the loop domain and the lane-allocation domain, we can tell which thread
// accesses which part of the TMem tensor. This information is used to check if
// the data access falls into one of the supported patterns, and if so, which
// pattern it is.
std::pair<
    std::unordered_map<TensorView*, TMemRegisterDataPath>,
    std::unordered_map<TensorView*, TMemRegisterDataPath>>
computeTMemLdStDataPath(Fusion* fusion, const TMemAlllocationInfo& allocation) {
  // This function uses simplifyExpr extensively. If we have disable expression
  // simplification in order to help inspect generated kernels then we will get
  // incorrect results here. Instead, we ensure it is enabled using this guard.
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().unset(DisableOption::ExprSimplify);

  // For all expressions in the fusion, find the data path
  using DPMap = std::unordered_map<TensorView*, TMemRegisterDataPath>;
  DPMap load_data_path;
  DPMap store_data_path;
  for (auto expr : fusion->exprs()) {
    auto ldst = dynamic_cast<LoadStoreOp*>(expr);
    if (ldst == nullptr) {
      continue;
    }
    TensorView* tmem_tv = nullptr;
    DPMap* target = nullptr;
    if (ldst->opType() == LoadStoreOpType::LdTMem) {
      tmem_tv = ir_utils::getTvInput(ldst);
      target = &load_data_path;
    } else if (ldst->opType() == LoadStoreOpType::StTMem) {
      tmem_tv = ir_utils::getTvOutput(ldst);
      target = &store_data_path;
    } else {
      continue;
    }
    const auto& tmem_tv_info = allocation.getTVInfo(tmem_tv);
    auto& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
    ValGroups lane_allocation_valgroups =
        id_graph.toGroups(tmem_tv_info.lane_allocation);

    // Get the merged parallel types of the interested TID parallel types
    // projected to expr.
    auto [pdims, strides] = getThreadParallelTypesMergedByContiguity(expr);

    // The innermost merged parallel type must be a multiple of 32, otherwise
    // the expr won't be warp-collective.
    Val* inner_extent = pdims.back()
                            .as<ValGroupAndItsGraph>()
                            .group->front()
                            ->as<IterDomain>()
                            ->extent();
    Val* inner_extent_is_multiple_of_32 = SimplifyingIrBuilder::eqExpr(
        SimplifyingIrBuilder::modExpr(
            inner_extent, IrBuilder::create<Val>(32, DataType::Index)),
        fusion->zeroVal());
    NVFUSER_LOWER_VALIDATE(
        inner_extent_is_multiple_of_32,
        "Invalid data access pattern in TMem load/store: ",
        "TMem load/store must be warp-collective, but the innermost extent is "
        "not a multiple of 32.");

    // For each outer parallel type that has extent > 1, its stride must be a
    // multiple of 32.
    for (auto [pdim, stride] :
         zip(pdims, strides) | std::views::take(strides.size() - 1)) {
      Val* pdim_extent = pdim.as<ValGroupAndItsGraph>()
                             .group->front()
                             ->as<IterDomain>()
                             ->extent();
      Val* pdim_extent_is_one =
          SimplifyingIrBuilder::eqExpr(pdim_extent, fusion->oneVal());
      Val* stride_is_multiple_of_32 = SimplifyingIrBuilder::eqExpr(
          SimplifyingIrBuilder::modExpr(
              stride, IrBuilder::create<Val>(32, DataType::Index)),
          fusion->zeroVal());
      NVFUSER_LOWER_VALIDATE(
          SimplifyingIrBuilder::logicalOrExpr(
              pdim_extent_is_one, stride_is_multiple_of_32),
          "Invalid data access pattern in TMem load/store: ",
          "Outer parallel types' strides must be a multiple of 32.");
    }

    // Start pattern matching:
    // fail_reasons will be used to store the reasons why the pattern does
    // not match for each pattern.
    std::vector<std::string> fail_reasons;
    bool matched = false;
    // Pattern match 32x32b
    if (!matched) {
      std::string reason_32x32b = "";
      AbstractTensor t = pdims;
      t.split(-1, 32);
      const ValGroup& warp = t.back().as<ValGroupAndItsGraph>().group;
      Val* stride = lower_utils::proveLinearAndGetStride(
          id_graph, warp, lane_allocation_valgroups);
      if (stride == nullptr) {
        reason_32x32b =
            "Not 32x32b because warps are not linearly accessing the lane "
            "allocation.";
        fail_reasons.push_back(std::move(reason_32x32b));
      } else {
        NVFUSER_LOWER_VALIDATE(
            SimplifyingIrBuilder::eqExpr(stride, fusion->oneVal()),
            "Invalid data access pattern in TMem load/store: ",
            "Warp linearly accessing lanes, but not with stride 1.");
        matched = true;
        (*target)[tmem_tv] = TMemRegisterDataPath::Path32x32b;
      }
    }
    // TODO: Pattern match 16x64b
    if (!matched) {
      std::string reason_16x64b =
          "Not 16x64b because it is not implemented in NVFuser yet.";
      fail_reasons.push_back(std::move(reason_16x64b));
    }
    // TODO: Pattern match 16x128b
    if (!matched) {
      std::string reason_16x128b =
          "Not 16x128b because it is not implemented in NVFuser yet.";
      fail_reasons.push_back(std::move(reason_16x128b));
    }
    // TODO: Pattern match 16x256b
    if (!matched) {
      std::string reason_16x256b =
          "Not 16x256b because it is not implemented in NVFuser yet.";
      fail_reasons.push_back(std::move(reason_16x256b));
    }
    // TODO: Pattern match 16x32bx2
    if (!matched) {
      std::string reason_16x32bx2 =
          "Not 16x32bx2 because it is not implemented in NVFuser yet.";
      fail_reasons.push_back(std::move(reason_16x32bx2));
    }
    // If none of the patterns match, throw an error.
    if (!matched) {
      std::stringstream error;
      error << "Invalid data access pattern in TMem load/store:";
      NVF_ERROR(fail_reasons.size() == 5);
      for (const std::string& reason : fail_reasons) {
        error << "\n  " << reason;
      }
      NVF_THROW(error.str());
    }
    // Validate that warps are accessing the correct sub-partition
    // Warp i can only access the sub-partition i % 4
    AbstractTensor t = pdims;
    t.split(-1, 32);
    t.split(-2, 4);
    Val* warp_group_stride = lower_utils::proveLinearAndGetStride(
        id_graph,
        t[-2].as<ValGroupAndItsGraph>().group,
        lane_allocation_valgroups);
    NVF_ERROR(
        warp_group_stride != nullptr,
        "Invalid data access pattern in TMem load/store: ",
        "Warps are not accessing the correct sub-partition.");
    // The stride must be either 0 or 32, 32 is the most common case.
    // 0 is a special value indicating that there is only one warp.
    NVFUSER_LOWER_VALIDATE(
        SimplifyingIrBuilder::logicalOrExpr(
            SimplifyingIrBuilder::eqExpr(
                warp_group_stride, IrBuilder::create<Val>(32)),
            SimplifyingIrBuilder::eqExpr(
                warp_group_stride, IrBuilder::create<Val>(0))),
        "Invalid data access pattern in TMem load/store: ",
        "Warps are not accessing the correct sub-partition.");
  }
  return {std::move(load_data_path), std::move(store_data_path)};
}

} // namespace

TensorMemoryInfo computeTMemInfo(Fusion* fusion) {
  TensorMemoryInfo result;
  result.allocation = computeTMemAlllocationInfo(fusion);
  std::tie(result.load_data_path, result.store_data_path) =
      computeTMemLdStDataPath(fusion, result.allocation);
  return result;
}

} // namespace nvfuser

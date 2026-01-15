// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <parallel_dimension_map.h>

#include <ATen/cuda/CUDAContext.h>
#include <device_lower/analysis/fusion_info.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <expr_simplifier.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <scheduler/utils.h>
#include <utils.h>

#include <sstream>
#include <string>
#include <utility>

using PAndID = std::pair<nvfuser::ParallelType, nvfuser::IterDomain*>;

namespace std {

template <>
struct hash<PAndID> {
  std::size_t operator()(const PAndID& data) const noexcept {
    size_t ptype = static_cast<size_t>(data.first);
    size_t address = reinterpret_cast<size_t>(data.second);
    size_t combined = (address << 8) | ptype;
    return std::hash<size_t>()(combined);
  }
};

} // namespace std

namespace nvfuser {

ParallelDimensionMap::ParallelDimensionMap(Fusion* fusion) {
  VectorOfUniqueEntries<PAndID> all_concrete_ids;
  auto all_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    if (tv->isCircularBuffered() &&
        std::holds_alternative<WarpSpecialized>(
            tv->circularBufferOptions().type)) {
      const auto& warp_specialized =
          std::get<WarpSpecialized>(tv->circularBufferOptions().type);
      NVF_ERROR(
          !warp_specialized_parallel_type_.has_value() ||
              warp_specialized_parallel_type_.value() == warp_specialized.on,
          "Multiple warp specialized axis detected.");
      warp_specialized_parallel_type_ = warp_specialized.on;
    }
    for (auto id : tv->domain()->allIDs()) {
      auto ptype = id->getParallelType();
      if (!isParallelTypeThread(ptype)) {
        continue;
      }
      auto concrete_id =
          FusionInfoGuard::current()->caMap().getConcreteMappedID(
              id, IdMappingMode::EXACT);
      if (concrete_id->isBroadcast()) {
        // Broadcasted concrete id's don't specify anything about shape
        continue;
      }
      all_concrete_ids.pushBack(std::make_pair(ptype, concrete_id));
    }
  }

  // Scan all TVs to build dim_map_
  for (auto [ptype, concrete_id] : all_concrete_ids) {
    exact_types_.insert(ptype); // insert now and cleanup later
    if (dim_map_.count(ptype) == 0) {
      dim_map_[ptype] = concrete_id->extent();
    } else {
      dim_map_.at(ptype) = SimplifyingIrBuilder::maxExpr(
          dim_map_.at(ptype), concrete_id->extent());
    }
  }

  // Simplify dim_map_
  for (auto& [k, v] : dim_map_) {
    v = simplifyExpr(v);
  }

  // Compute exact_types_
  for (auto [ptype, concrete_id] : all_concrete_ids) {
    auto expr_val = simplifyExpr(SimplifyingIrBuilder::eqExpr(
                                     dim_map_.at(ptype), concrete_id->extent()))
                        ->value();
    if (!expr_val.hasValue() || !expr_val.as<bool>()) {
      exact_types_.erase(ptype);
    }
  }

  adjustMappingsForWarpPadding();
  adjustMappingsForWarpSpecialization();
}

void ParallelDimensionMap::adjustMappingsForWarpPadding() {
  // If TIDx is padded to a multiple of the warp size, mark it as
  // non-exact.
  NVF_ERROR(
      FusionInfoGuard::hasCurrent() &&
      FusionInfoGuard::current()->hasPaddedParallelDimensions());
  const auto& warp_info =
      FusionInfoGuard::current()->paddedParallelDimensions();
  // TIDx isn't really padded if there isn't a warp reduction (this could
  // change)
  if (!(warp_info.is_tidx_padded && warp_info.has_warp_reduction)) {
    return;
  }

  const auto tidx_pt = ParallelType::TIDx;
  auto warp_size_val = IrBuilder::create<Val>(32L, DataType::Index);
  auto tidx_dim = getRaw(tidx_pt);

  NVF_ERROR(tidx_dim != nullptr);

  // If tidx is strictly defined as blockDim.x then it must be set to a
  // multiple of the warp, there is nothing to do
  if (tidx_dim->sameAs(NamedScalar::getParallelDim(tidx_pt))) {
    return;
  }

  auto expr_val =
      simplifyExpr(SimplifyingIrBuilder::eqExpr(
                       SimplifyingIrBuilder::modExpr(tidx_dim, warp_size_val),
                       tidx_dim->container()->zeroVal()))
          ->value();

  // If already multiple of warp, nothing to do
  if (expr_val.is<bool>() && expr_val.as<bool>()) {
    return;
  }

  // TIDx is padded to a multiple of warp. If it's known to be a
  // single warp, use the constant warp size as the dimension of
  // TIDx. Otherwise, just use blockDim.x.
  if (warp_info.is_tidx_single_warp) {
    dim_map_.at(ParallelType::TIDx) = warp_size_val;
  } else {
    dim_map_.at(ParallelType::TIDx) =
        simplifyExpr(SimplifyingIrBuilder::mulExpr(
            SimplifyingIrBuilder::ceilDivExpr(tidx_dim, warp_size_val),
            warp_size_val));
  }

  // TIDx is no longer exact
  exact_types_.erase(ParallelType::TIDx);
}

int64_t ParallelDimensionMap::getThreadCountInDim(ParallelType pt) {
  if (!dim_map_.contains(pt)) {
    return 1;
  }
  if (dim_map_.at(pt)->isConstScalar()) {
    return dim_map_.at(pt)->value().as<int64_t>();
  }
  // If dimension is dynamic but we have compile-time CTA shape available,
  // use the actual compile parameter value
  NVF_ERROR(GpuLower::hasCurrent());
  const auto& cparams = GpuLower::current()->compileParams();
  if (pt == ParallelType::TIDx && cparams.bdimx.has_value()) {
    return cparams.bdimx.value();
  } else if (pt == ParallelType::TIDy && cparams.bdimy.has_value()) {
    return cparams.bdimy.value();
  } else if (pt == ParallelType::TIDz && cparams.bdimz.has_value()) {
    return cparams.bdimz.value();
  }
  // Return -1 for dynamic dimensions when compile-time CTA shape is not known,
  // this disables register sharing on dynamic dimensions since we can't
  // guarantee the number of threads is divisible by 128.
  return -1;
}

void ParallelDimensionMap::adjustMappingsForWarpSpecialization() {
  // shortcut for case without warp specialization
  if (!warp_specialized_parallel_type_.has_value()) {
    return;
  }
  auto ws_pt = warp_specialized_parallel_type_.value();

  // Must use static CTA shape for warp specialization
  // If loop domain is symbolic, derive corresponding threads from compile
  // params
  NVF_ERROR(GpuLower::hasCurrent());
  const auto& cparams = GpuLower::current()->compileParams();
  for (auto pt : kParallelTypeTIDs) {
    if (pt == ws_pt) {
      continue;
    }
    if (dim_map_.contains(pt) && !dim_map_.at(pt)->isConstScalar()) {
      NVF_ERROR(cparams.getDim(pt).has_value());
      dim_map_[pt] =
          IrBuilder::create<Val>(cparams.getDim(pt).value(), DataType::Index);
    }
  }

  for (auto [pt, dim] : dim_map_) {
    std::cout << pt << ": " << dim->toInlineString() << std::endl;
  }

  // validate the CTA shape, ensure preceding dimensions have 128x threads
  if (ws_pt == ParallelType::TIDx || ws_pt == ParallelType::TIDy) {
    NVF_ERROR(dim_map_.at(ParallelType::TIDx)->isConstScalar());
    const int64_t bdimx =
        dim_map_.at(ParallelType::TIDx)->evaluate().as<int64_t>();
    NVF_ERROR(bdimx % kWarpSpecializationPaddedThreads == 0);
  } else if (ws_pt == ParallelType::TIDz) {
    NVF_ERROR(dim_map_.at(ParallelType::TIDx)->isConstScalar());
    NVF_ERROR(dim_map_.at(ParallelType::TIDy)->isConstScalar());
    const int64_t bdimx =
        dim_map_.at(ParallelType::TIDx)->evaluate().as<int64_t>();
    const int64_t bdimy =
        dim_map_.at(ParallelType::TIDy)->evaluate().as<int64_t>();
    NVF_ERROR(bdimx * bdimy % kWarpSpecializationPaddedThreads == 0);
  }

  // for warp specialized dimension, set the padded value or directly use
  // compile params
  if (!dim_map_.contains(ws_pt) || dim_map_.at(ws_pt)->isConstScalar()) {
    int64_t unpadded_val = dim_map_.contains(ws_pt)
        ? dim_map_.at(ws_pt)->evaluate().as<int64_t>()
        : 1L;
    auto padded_val = unpadded_val + getWarpSpecializationPaddedVal(ws_pt);
    dim_map_[ws_pt] = IrBuilder::create<Val>(padded_val, DataType::Index);
  } else {
    NVF_ERROR(cparams.getDim(ws_pt).has_value());
    dim_map_[ws_pt] =
        IrBuilder::create<Val>(cparams.getDim(ws_pt).value(), DataType::Index);
  }
  exact_types_.erase(ws_pt);
}

Val* ParallelDimensionMap::getRaw(ParallelType pt) const {
  NVF_ERROR(isParallelTypeThread(pt), "Invalid ParallelType: ", pt);
  auto it = dim_map_.find(pt);
  if (it == dim_map_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

Val* ParallelDimensionMap::get(ParallelType pt) const {
  auto raw = getRaw(pt);
  if (raw != nullptr && !raw->isConstInt()) {
    return NamedScalar::getParallelDim(pt);
  }
  return raw;
}

bool ParallelDimensionMap::isExact(ParallelType pt) const {
  return exact_types_.find(pt) != exact_types_.end();
}

Val* ParallelDimensionMap::getRawCompute(ParallelType pt) const {
  Val* raw = getRaw(pt);
  if (isWarpSpecialized(pt)) {
    int64_t padded_val = getWarpSpecializationPaddedVal(pt);
    return SimplifyingIrBuilder::addExpr(raw, -padded_val);
  }
  return raw;
}

Val* ParallelDimensionMap::getRawAsync(ParallelType pt) const {
  if (isWarpSpecialized(pt)) {
    return IrBuilder::create<Val>(
        getWarpSpecializationPaddedVal(pt), DataType::Index);
  }
  return getRaw(pt);
}

Val* ParallelDimensionMap::getNumComputeThreadsEachBlock() const {
  Val* num_threads = FusionGuard::getCurFusion()->oneVal();
  for (auto pt : kParallelTypeTIDs) {
    // Skip warp specialized ParallelType if the are computation warp groups
    // are independent.
    if (isWarpSpecialized(pt) &&
        GpuLower::current()
            ->circularBufferInfo()
            .hasIndependentComputeWarpGroups()) {
      continue;
    }
    auto dim = getRawCompute(pt);
    if (dim == nullptr) {
      continue;
    }
    num_threads = SimplifyingIrBuilder::mulExpr(num_threads, dim);
  }
  return num_threads;
}

// For warp-specialization, the CTA is padded so the AsyncWarp contains 128
// threads. This function maps the AsyncWarp CTA to a linear index from
// [0, 128). It is used to divide AsyncWarp into four independent warps.
Val* ParallelDimensionMap::getLinearThreadIndexAsync() const {
  Val* index = GpuLower::current()->kernel()->zeroVal();
  Val* extent = GpuLower::current()->kernel()->oneVal();

  for (auto pt : kParallelTypeTIDs) {
    // For warp-specialization, an axis is padded so the AsyncWarp contains
    // 128 threads.
    Val* extent_for_pdim = getRawAsync(pt);
    // short-circuit: extent_for_pdim is not used in kernel.
    if (extent_for_pdim == nullptr) {
      continue;
    }
    // short-circuit: extent_for_pdim is trivial.
    if (extent_for_pdim->isConstScalar() &&
        extent_for_pdim->evaluate().as<int64_t>() == 1) {
      continue;
    }
    Val* pt_index = NamedScalar::getParallelIndex(pt);
    // Map the padded parallel index to [0, padded_value] range, so the linear
    // index will be in range of [0, 128).
    if (isWarpSpecialized(pt)) {
      pt_index = SimplifyingIrBuilder::subExpr(pt_index, getRawCompute(pt));
    }
    index = SimplifyingIrBuilder::addExpr(
        index, SimplifyingIrBuilder::mulExpr(pt_index, extent));
    extent = SimplifyingIrBuilder::mulExpr(extent, extent_for_pdim);
  }
  return index;
}

int64_t ParallelDimensionMap::getWarpSpecializationPaddedVal(
    ParallelType pt) const {
  NVF_ERROR(isWarpSpecialized(pt), "Can't find ParallelType: ", pt);
  switch (pt) {
    case ParallelType::TIDx:
      return kWarpSpecializationPaddedThreads; // 128
    case ParallelType::TIDy:
      return 1;
    case ParallelType::TIDz:
      return 1;
    default:
      NVF_ERROR("Not supported warp specialization on ", pt);
      return 0;
  }
}

bool ParallelDimensionMap::canUseElectSyncInAsyncWarp() const {
  // short-circuit: skip if warp specialization is not enabled
  if (!hasWarpSpecialization()) {
    return true;
  }
  // Currently only support one warp specialized axis
  NVF_ERROR(warp_specialized_parallel_type_.has_value());
  ParallelType ws_pt = warp_specialized_parallel_type_.value();

  // Check that BlockDim.x >= 32 active threads in AsyncWarp
  if (ws_pt != ParallelType::TIDx) {
    return true;
  }

  if (getWarpSpecializationPaddedVal(ws_pt) >= 32) {
    return true;
  }

  return false;
}

std::string ParallelDimensionMap::toString() const {
  std::stringstream ss;
  for (auto pt : kParallelTypeThreads) {
    ss << pt << ": ";
    auto dim = getRaw(pt);
    if (dim != nullptr) {
      ss << dim->toInlineString();
      if (isExact(pt)) {
        ss << ", exact";
      } else {
        ss << ", non-exact";
      }
    } else {
      ss << "unused";
    }
    ss << "\n";
  }

  return ss.str();
}

} // namespace nvfuser

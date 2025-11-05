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
#include <ir/builder.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <scheduler/utils.h>

#include <functional>
#include <sstream>
#include <string>
#include <utility>

using PAndID = std::pair<nvfuser::ParallelType, nvfuser::IterDomain*>;
using PDAndID = std::pair<nvfuser::ParallelDim*, nvfuser::IterDomain*>;

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

template <>
struct hash<PDAndID> {
  std::size_t operator()(const PDAndID& data) const noexcept {
    size_t h = std::hash<size_t>()(reinterpret_cast<size_t>(data.first));
    nvfuser::hashCombine(
        h, std::hash<size_t>()(reinterpret_cast<size_t>(data.second)));
    return h;
  }
};

} // namespace std

namespace nvfuser {

ParallelDimensionMap::ParallelDimensionMap(Fusion* fusion) {
  inferEvalExtents(fusion);
  inferCodegenExtents(fusion);
  inferIndices(fusion);

  for (auto [pdim, extent] : dim_eval_extent_map_) {
    if (pdim->parallelType() != ParallelType::Derived) {
      dim_map_[pdim->parallelType()] = extent;
    }
  }

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

void ParallelDimensionMap::inferEvalExtents(Fusion* fusion) {
  // TODO: I think we still need something like exact_types_ but for
  // ParallelDim. isExact() is currently only used by
  // getThreadParallelTypesMergedByContiguity in tmem analysis, so we should
  // reference that use case when designing exactness around ParallelDim*
  // instead of ParallelType*
  std::cout << "inferEvalExtents\n";

  VectorOfUniqueEntries<PDAndID> all_concrete_ids;
  auto all_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    for (auto id : tv->domain()->allIDs()) {
      std::cout << "  " << id->toString();
      ParallelDim* pdim = id->getParallelDim();
      ParallelType ptype = id->getParallelType();
      if (pdim == nullptr) {
        pdim = fusion->getParallelDim(ptype);
      }
      NVF_ERROR_EQ(pdim->parallelType(), ptype);

      if (ptype != ParallelType::Derived && !isParallelTypeThread(ptype)) {
        continue;
      }
      NVF_ERROR(FusionInfoGuard::hasCurrent());
      IterDomain* concrete_id =
          FusionInfoGuard::current()->caMap().getConcreteMappedID(
              id, IdMappingMode::EXACT);
      // NOTE: Broadcasted concrete id's can inform the shape, especially if
      // they are broadcast
      all_concrete_ids.pushBack(std::make_pair(pdim, concrete_id));
    }
  }

  // Scan all TVs to build dim_eval_extent_map_
  for (auto [pdim, concrete_id] : all_concrete_ids) {
    auto it = dim_eval_extent_map_.find(pdim);
    if (it == dim_eval_extent_map_.end()) {
      dim_eval_extent_map_[pdim] = concrete_id->getMaybeExpandedExtent();
    } else {
      it->second =
          SimplifyingIrBuilder::maxExpr(it->second, concrete_id->extent());
    }
  }

  // Simplify dim_map_
  for (auto& [pdim, extent] : dim_eval_extent_map_) {
    extent = simplifyExpr(extent);
    if (extent->isConstInt()) {
      pdim->value() = extent->evaluate().as<int64_t>();
    }
  }

  // At this point, the extents are specified for each _bound_ ParallelDim.
  // However, we might have only bound the dimensions for derived ParallelDims
  // but we will also need to evaluate the ones corresponding to ParallelTypes.
  // Consider a case like this:
  //
  //       TIDx
  //       /  \
  //   p2(8)  p3(32)
  //
  // Here the TIDx dim is split into p2 and p3 and they are both bound. In this
  // case we should infer that TIDx has extent 8*32=256.
  //
  // Now what if we had this:
  //
  //              BIDx{64}
  //              /      \
  //    ClusterIDx{16}   ClusterCtaIDx
  //
  // In this case, although we know that blockDim.x should be 64 we still need
  // to infer the cluster X dimension (64/16 = 4).
  //
  // These examples show that we must propagate in both directions from bound
  // extents to fill in as much as we can. We should also validate that the
  // bound values are consistent with the structure of the parallel graph so
  // that for example we could not have legally bound an ID with extent 15 to
  // ClusterIDx because that is not divisible by 64.

  // We process ParallelDim Exprs repeatedly until we have inferred all input
  // and output extents. We also track the number of changes made so far at the
  // time each item is pushed to the queue. That number allows us to prevent
  // infinite loops.
  std::queue<std::pair<Expr*, int64_t>> queue;
  int64_t num_updates = -1;

  // Initialize queue with definitions and uses of known parallel types
  for (auto ptype_num : arange(toUnderlying(ParallelType::Count))) {
    ParallelType ptype{ptype_num};
    if (ptype == ParallelType::Derived || ptype == ParallelType::Count) {
      continue;
    }
    if (fusion->hasParallelDim(ptype)) {
      ParallelDim* pdim = fusion->getParallelDim(ptype);
      NVF_ERROR(pdim != nullptr);
      if (pdim->definition() != nullptr) {
        queue.emplace(pdim->definition(), num_updates);
      }
      NVF_ERROR_LE(pdim->uses().size(), 1);
      for (Expr* use : pdim->uses()) {
        queue.emplace(use, num_updates);
      }
    }
  }

  auto update = [&](ParallelDim* pdim, Val* extent) {
    if (extent->isConstInt()) {
      pdim->value() = extent->evaluate();
    }
    dim_eval_extent_map_.emplace(pdim, extent);
    num_updates++;
  };

  num_updates = 0;
  while (!queue.empty()) {
    NVF_ERROR_LT(
        queue.size(),
        1 << 20,
        "Maximum ParallelDim processing queue size exceeded");

    auto [expr, num_updates_when_pushed] = queue.front();
    queue.pop();

    if (num_updates_when_pushed == num_updates) {
      // If we have not done any updates since pushing this expression, then
      // nothing will have changed so to prevent an infinite loop, we bail
      break;
    }

    for (Val* out : expr->outputs()) {
      std::cout << out->toString() << ", ";
    }
    std::cout << " = ";
    std::cout << expr->toString() << std::endl;
    if (auto* split = dynamic_cast<ParallelDimSplit*>(expr)) {
      Val* in = mapOrDefault(
          dim_eval_extent_map_, split->in(), /*default=*/(Val*)nullptr);
      Val* outer = mapOrDefault(
          dim_eval_extent_map_, split->outer(), /*default=*/(Val*)nullptr);
      Val* inner = mapOrDefault(
          dim_eval_extent_map_, split->inner(), /*default=*/(Val*)nullptr);
      if (in && outer && inner) {
        // Nothing to infer
        continue;
      } else if (in && outer && !inner) {
        // We know that inner should be  in / outer
        // TODO: we should mark in % outer == 0 for later validation somehow
        for (Expr* use : split->inner()->uses()) {
          queue.emplace(use, num_updates);
        }
        update(split->inner(), SimplifyingIrBuilder::divExpr(in, outer));
      } else if (in && !outer && inner) {
        for (Expr* use : split->outer()->uses()) {
          queue.emplace(use, num_updates);
        }
        // TODO: we should mark in % inner == 0 for later validation somehow
        update(split->outer(), SimplifyingIrBuilder::divExpr(in, inner));
      } else if (!in && outer && inner) {
        if (split->in()->definition() != nullptr) {
          queue.emplace(split->in()->definition(), num_updates);
        }
        update(split->in(), SimplifyingIrBuilder::mulExpr(outer, inner));
      } else {
        // We can't infer anything about this Expr yet
        queue.emplace(expr, num_updates);
      }
    } else {
      NVF_THROW("Unhandled ParallelDim expression: ", expr->toString());
    }
  }
}

void ParallelDimensionMap::inferCodegenExtents(Fusion* fusion) {}
void ParallelDimensionMap::inferIndices(Fusion* fusion) {}

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
  // Return -1 for dynamic dimensions, this disables register sharing on
  // dynamic dimensions since we can't guarantee the number of threads is
  // divisible by 128. We may allow this in the future and delegate this
  // check to a point where the launch parameters are known.
  return -1;
}

void ParallelDimensionMap::adjustMappingsForWarpSpecialization() {
  // shortcut for case without register sharing
  if (!warp_specialized_parallel_type_.has_value()) {
    return;
  }

  // Warp specialization with register sharing on parallel type pt
  // index = TIDx + TIDy * bdimx + TIDz * bdimx * bdimy
  auto ws_pt = warp_specialized_parallel_type_.value();
  auto dim_it = dim_map_.find(ws_pt);

  int64_t other_active_pts_threads = 1;
  for (ParallelType pt : kParallelTypeTIDs) {
    if (pt == ws_pt) {
      continue;
    }
    int64_t thread_count_for_pt = getThreadCountInDim(pt);
    NVF_ERROR(
        thread_count_for_pt != -1,
        "Detected dynamic size for parallel type ",
        pt,
        " in warp specialization kernel.");
    other_active_pts_threads *= thread_count_for_pt;
  }
  NVF_ERROR(
      other_active_pts_threads <= 128,
      "The # active threads in other thread dimensions > 128 threads.");
  NVF_ERROR(
      128 % other_active_pts_threads == 0,
      "The # active threads in other thread dimensions is not evenly ",
      "divisible with 128 threads.");
  int64_t ws_num_threads_pad = 128 / other_active_pts_threads;
  int64_t after_pad = getThreadCountInDim(ws_pt) + ws_num_threads_pad;
  NVF_ERROR(
      (after_pad * other_active_pts_threads) % 128 == 0,
      "Illegal register sharing on ",
      ws_pt,
      " with padded size ",
      after_pad,
      " and remaining active cta threads ",
      other_active_pts_threads);

  // Apply the pad
  warp_specialized_padding_value_ = ws_num_threads_pad;
  auto offset = IrBuilder::create<Val>(ws_num_threads_pad, DataType::Index);
  auto current_val = dim_it == dim_map_.end()
      ? IrBuilder::create<Val>(1, DataType::Index)
      : dim_it->second;
  dim_map_[ws_pt] = IrBuilder::addExpr(current_val, offset);
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
  if (!warp_specialized_parallel_type_.has_value()) {
    return 1;
  }
  NVF_ERROR(
      warp_specialized_parallel_type_.value() == pt,
      "Can't find padded val for: ",
      pt);
  return warp_specialized_padding_value_.value();
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

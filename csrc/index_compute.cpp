// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <index_compute.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/irange.h>

#include <contiguity.h>
#include <device_lower/analysis/index_compute.h>
#include <device_lower/analysis/shift.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/double_buffer.h>
#include <device_lower/pass/magic_zero.h>
#include <device_lower/pass/unroll.h>
#include <device_lower/utils.h>
#include <device_lower/validation.h>
#include <expr_simplifier.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <root_domain_map.h>
#include <swizzle.h>
#include <transform_iter.h>
#include <transform_replay.h>

#include <memory>

namespace nvfuser {

namespace {

//! Offset of an index of a producer axis with respect to its
//! corresponding consumer index
//! TODO: this function assumes that we are indexing into rFactor domain, which
//! is no longer the case. Today, we are indexing into allocaiton domain, and if
//! the allocation domain is not a permutation of the rFactor domain, this
//! function will just return 0, because we do not have a case that uses both
//! allocation domain and halo yet.
int getProducerHaloOffset(
    const TensorView* producer_tv,
    size_t producer_axis,
    const TensorView* consumer_tv) {
  // For indexing, having same extents is not required for root
  // domains
  auto p2c = PairwiseRootDomainMap(producer_tv, consumer_tv)
                 .mapBroadcast(true)
                 .mapDifferentExtents(true)
                 .mapProducerToConsumer();

  auto producer_id = producer_tv->getMaybeAllocationDomain()[producer_axis];

  auto it = p2c.find(producer_id);
  // p2c should always have a mapping for producer_id. The only case
  // where no mapping exists for a producer axis is when it is a
  // reduction axis. Since this function is only used for indexing
  // producer tensors, where reduction axes are skipped, producer_id
  // should never be a reduction axis.
  if (it == p2c.end()) {
    return 0;
  }
  IterDomain* consumer_id = it->second;

  const auto& halo_map = GpuLower::current()->haloInfo();
  const auto p_pad = halo_map->getRootAxisInfo(producer_id).width(0);
  const auto c_pad = halo_map->getRootAxisInfo(consumer_id).width(0);

  auto offset = p_pad - c_pad;

  // If the consumer is a result of shifting the producer, adjust the
  // producer index per the offsets argument of the shift op.
  if (auto shift_op = dynamic_cast<const ShiftOp*>(consumer_tv->definition())) {
    offset -= shift_op->offset(producer_axis);
  }

  return offset;
}

//! Offset producer index when necessary
Val* getProducerIndexWithHalo(
    const TensorView* producer_tv,
    size_t producer_axis,
    Val* producer_index,
    const TensorView* consumer_tv,
    bool is_overriden_index) {
  const int64_t offset = is_overriden_index
      ? 0
      : getProducerHaloOffset(producer_tv, producer_axis, consumer_tv);

  if (offset == 0) {
    return producer_index;
  }

  producer_index = SimplifyingIrBuilder::addExpr(producer_index, offset);

  return producer_index;
}

//! Create a producer offset based off a consumer index
//!
//! \param consumer_root_axis Position of corresponding consumer axis
//! \param consumer_tv Consumer TensorView
//! \param index_map Mappings from consumer or reference to indices
//! \param use_reference_map True when index_map maps reference domains
//! \param concrete_to_ref_map Mappings from concrete to reference domains
Val* getProducerOffsetWithGather(
    int64_t consumer_root_axis,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& index_map,
    bool use_reference_map = false,
    const std::unordered_map<IterDomain*, IterDomain*>& concrete_to_ref_map =
        {}) {
  const auto gpu_lower = GpuLower::current();

  const auto gather_expr = dynamic_cast<GatherOp*>(consumer_tv->definition());

  if (gather_expr == nullptr) {
    return gpu_lower->kernel()->zeroVal();
  }

  // If the window extent is one, no specific offsetting
  // is necessary
  if (consumer_root_axis >= (int)gather_expr->windowShape().size() ||
      gather_expr->windowShape()[consumer_root_axis] == 1) {
    return gpu_lower->kernel()->zeroVal();
  }

  // Basically, the goal is to build an expression of producer_index +
  // window_index, so we first need to locate the index expression
  // that corresponds to the window axis of this producer axis.

  const auto window_axis = gather_expr->gatherAxis(consumer_root_axis);
  auto window_id = consumer_tv->getRootDomain().at(window_axis);

  // When index_map maps a reference tensor, find the corresponding
  // reference ID of window_id.
  if (use_reference_map) {
    auto concrete_window_id = gpu_lower->caMap()->getConcreteMappedID(
        window_id, IdMappingMode::EXACT);
    auto concrete_2_ref_it = concrete_to_ref_map.find(concrete_window_id);
    NVF_ERROR(concrete_2_ref_it != concrete_to_ref_map.end());
    window_id = concrete_2_ref_it->second;
  }

  auto window_idx = index_map.at(window_id);

  // Positive padding at offset zero means the indexing shifted to the
  // negative direction.
  auto pad_width = (int64_t)gather_expr->padWidth()[consumer_root_axis][0];

  // producer offset: window_index - padding
  auto producer_offset = SimplifyingIrBuilder::subExpr(
      window_idx,
      SimplifyingIrBuilder::create<Val>(pad_width, DataType::Index));
  return producer_offset;
}

//! Create a producer offset based off a consumer index
//!
//! \param consumer_root_axis Position of corresponding consumer axis
//! \param consumer_tv Consumer TensorView
//! \param index_map Mappings from consumer or reference to indices
//! \param use_reference_map True when index_map maps reference domains
//! \param concrete_to_ref_map Mappings from concrete to reference domains
Val* getConcreteProducerOffsetWithGather(
    int64_t consumer_root_axis,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& index_map,
    bool use_concrete_map = false) {
  const auto gpu_lower = GpuLower::current();

  const auto gather_expr = dynamic_cast<GatherOp*>(consumer_tv->definition());

  if (gather_expr == nullptr) {
    return gpu_lower->kernel()->zeroVal();
  }

  // If the window extent is one, no specific offsetting
  // is necessary
  if (consumer_root_axis >= (int64_t)gather_expr->windowShape().size() ||
      gather_expr->windowShape()[consumer_root_axis] == 1) {
    return gpu_lower->kernel()->zeroVal();
  }

  // Basically, the goal is to build an expression of producer_index +
  // window_index, so we first need to locate the index expression
  // that corresponds to the window axis of this producer axis.

  const auto window_axis = gather_expr->gatherAxis(consumer_root_axis);
  auto window_id = consumer_tv->getRootDomain().at(window_axis);

  Val* window_idx = nullptr;

  if (use_concrete_map) {
    window_idx = index_map.at(GpuLower::current()->caMap()->getConcreteMappedID(
        window_id, IdMappingMode::EXACT));
  } else {
    window_idx = index_map.at(window_id);
  }

  // Positive padding at offset zero means the indexing shifted to the
  // negative direction.
  auto pad_width = (int64_t)gather_expr->padWidth()[consumer_root_axis][0];

  // producer offset: window_index - padding
  auto producer_offset = SimplifyingIrBuilder::subExpr(
      window_idx,
      SimplifyingIrBuilder::create<Val>(pad_width, DataType::Index));
  return producer_offset;
}

//! Offset a producer index of a gather expression
//!
//! Given an index of a producer root axis, build a new index
//! expression that accesses a window position that the current loop
//! structure refers to. Use getGatherProducerOffset to create an
//! offset Val.
Val* getProducerIndexWithGather(
    Val* producer_index,
    size_t producer_root_axis,
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& concrete_index_map) {
  auto gather_op = dynamic_cast<const GatherOp*>(consumer_tv->definition());

  // Just return the producer index as is if this is not a gather
  if (gather_op == nullptr) {
    return producer_index;
  }

  // Consumer axis that corresponds to the producer axis
  int64_t consumer_axis = -1;
  for (const auto i : c10::irange(producer_root_axis + 1)) {
    if (producer_tv->getMaybeRFactorDomain()[i]->isReduction() ||
        producer_tv->getMaybeRFactorDomain()[i]->isStride()) {
      continue;
    }
    ++consumer_axis;
  }

  NVF_ERROR(
      consumer_axis >= 0 &&
          consumer_axis < (int)gather_op->windowShape().size(),
      "Invalid consumer axis",
      consumer_axis,
      ", producer_axis: ",
      producer_root_axis);

  auto offset = getConcreteProducerOffsetWithGather(
      consumer_axis, consumer_tv, concrete_index_map, true);
  return SimplifyingIrBuilder::addExpr(producer_index, offset);
}

// Adjusts a global consumer index when its root domain is partially
// split. Note that non-global consumer indices don't need any
// adjustment.
Val* getGlobalConsumerOffsetWithPartialSplit(IterDomain* root_id) {
  auto offset = GpuLower::current()->partialSplitMap().getStartOffset(root_id);
  if (offset == nullptr) {
    return GpuLower::current()->kernel()->zeroVal();
  } else {
    return offset;
  }
}

// Adjusts a global producer index when its root domain and
// corresponding consumer root domain have non-matching split
// offsets. Specifically, since producer_index is calcualted based on
// the consumer, if the consumer has a non-zero offset,
// it needs to be added to the index. Also, when the producer itself
// also has a non-zero split offset, that needs to be subtracted from
// the index.
Val* getProducerIndexWithPartialSplit(
    Val* producer_index,
    IterDomain* producer_root_id,
    const TensorView* producer_tv,
    const TensorView* consumer_tv) {
  const auto gpu_lower = GpuLower::current();

  auto p2c =
      PairwiseRootDomainMap(producer_tv, consumer_tv).mapProducerToConsumer();

  auto it = p2c.find(producer_root_id);
  if (it == p2c.end()) {
    return producer_index;
  }

  auto consumer_root_id = it->second;

  auto consumer_offset =
      gpu_lower->partialSplitMap().getStartOffset(consumer_root_id);
  consumer_offset = consumer_offset == nullptr ? gpu_lower->kernel()->zeroVal()
                                               : consumer_offset;

  auto producer_offset =
      gpu_lower->partialSplitMap().getStartOffset(producer_root_id);
  producer_offset = producer_offset == nullptr ? gpu_lower->kernel()->zeroVal()
                                               : producer_offset;

  // If the producer is on global memory, it's always allocated
  // without trimming the out-of-bounds region, so the consumer offset
  // should be added to the index.
  if (producer_tv->getMemoryType() == MemoryType::Global) {
    if (consumer_offset->isZeroInt()) {
      return producer_index;
    } else {
      return SimplifyingIrBuilder::addExpr(producer_index, consumer_offset);
    }
  }

  // Non-global case. Difference of the split offsets must be
  // accounted.

  auto diff = SimplifyingIrBuilder::subExpr(consumer_offset, producer_offset);
  // We currently only allow constant offsetting
  NVF_ERROR(
      diff->isConstScalar(),
      "Invalid partial split, must be a constant value.");

  if (diff->evaluate() == 0) {
    return producer_index;
  }

  return SimplifyingIrBuilder::addExpr(
      producer_index,
      SimplifyingIrBuilder::create<Val>(diff->evaluate(), DataType::Index));
}

} // namespace

bool IndexCompute::hasUnswitchedDependentDomains(IterDomain* id) const {
  auto concrete_id = maybeGetExactMapConcreteID(id);
  auto it = unswitched_domain_map_.find(concrete_id);
  return it != unswitched_domain_map_.end() && !it->second.empty();
}

void IndexCompute::initializeUnswitchDomainMap() {
  NVF_ERROR(unswitched_domain_map_.empty());
  for (auto id : unswitched_leaf_domains_) {
    auto concrete_id = maybeGetExactMapConcreteID(id);
    unswitched_domain_map_.emplace(
        concrete_id,
        std::vector<std::deque<IterDomain*>>{
            std::deque<IterDomain*>{concrete_id}});
  }
}

void IndexCompute::updateUnswitchedDomains(Expr* expr) {
  if (auto split = dynamic_cast<Split*>(expr)) {
    auto split_in = maybeGetExactMapConcreteID(split->in());
    for (auto split_out : {split->inner(), split->outer()}) {
      auto concrete_id = maybeGetExactMapConcreteID(split_out);
      if (auto it = unswitched_domain_map_.find(concrete_id);
          it != unswitched_domain_map_.end()) {
        if (split_out == split->inner()) {
          // In the case of upward traversal from the inner output,
          // just copy the unswitched info
          unswitched_domain_map_[split_in] = it->second;
        } else {
          // In the case of upward traversal from the outer output,
          // prepend the inner domain to the lists
          for (auto unswitched_dep_ids : it->second) {
            unswitched_dep_ids.push_front(
                maybeGetExactMapConcreteID(split->inner()));
            unswitched_domain_map_[split_in].push_back(unswitched_dep_ids);
          }
        }
      }
    }
  } else {
    // Suppress a clang-tidy warning
    NVF_ERROR(expr != nullptr);
    // Propagate the unswitch info if any of outputs is
    // unswitched. Unlike the split case, the propagated info
    // is just reset as there's no obvious way to back-propagate the
    // info through, e.g., merge
    if (std::any_of(
            expr->outputs().begin(), expr->outputs().end(), [this](Val* out) {
              return out->isA<IterDomain>() &&
                  hasUnswitchedDependentDomains(out->as<IterDomain>());
            })) {
      for (auto inp : ir_utils::filterByType<IterDomain>(expr->inputs())) {
        auto inp_concrete = maybeGetExactMapConcreteID(inp);
        unswitched_domain_map_.emplace(
            inp_concrete,
            std::vector<std::deque<IterDomain*>>{
                std::deque<IterDomain*>{inp_concrete}});
      }
    }
  }
}

void IndexCompute::handle(Split* split) {
  auto in_id = maybeGetExactMapConcreteID(split->in()->as<IterDomain>());
  auto outer_id = maybeGetExactMapConcreteID(split->outer()->as<IterDomain>());
  auto inner_id = maybeGetExactMapConcreteID(split->inner()->as<IterDomain>());

  auto outer_it = index_map_.find(outer_id);
  auto inner_it = index_map_.find(inner_id);
  if (outer_it == index_map_.end() || inner_it == index_map_.end()) {
    return;
  }

  const auto outer_ind = outer_it->second;
  const auto inner_ind = inner_it->second;

  const bool outer_zero = isZero(outer_id);
  const bool inner_zero = isZero(inner_id);

  // We want to mark as zero merged in if we're working with shared or local
  // memory, and the dimension we're working with is not part of the allocation,
  // as we have special propagation rules for that scenario.

  // Maybe clear in_id as it could have been mapped over from another
  // IndexCompute. Uncertain if this is needed but seems to be safe.
  bool zero_merged_in = hasZeroMerged(in_id) || hasZeroMerged(inner_id) ||
      hasZeroMerged(outer_id);

  // If both are zero, the split input is also zero
  if (inner_zero && outer_zero) {
    zero_domains_.emplace(in_id);
  }

  if (zero_merged_in) {
    zero_merged_in_.emplace(in_id);
  }

  if (isZero(in_id)) {
    index_map_[in_id] = GpuLower::current()->kernel()->zeroVal();
    extent_map_[in_id] = GpuLower::current()->kernel()->zeroVal();
  } else if (zero_merged_in && outer_zero) {
    index_map_[in_id] = inner_ind;
    extent_map_[in_id] = getExtent(inner_id);
  } else if (zero_merged_in && inner_zero) {
    index_map_[in_id] = outer_ind;
    extent_map_[in_id] = getExtent(outer_id);
  } else {
    index_map_[in_id] = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(outer_ind, getExtent(inner_id)),
        inner_ind);
    // The extent should be updated only when its allocation is
    // partial, i.e., zero_merged_in is true. See PR #1270.
    if (zero_merged_in) {
      extent_map_[in_id] = SimplifyingIrBuilder::mulExpr(
          getExtent(outer_id), getExtent(inner_id));
    }
  }
}

bool IndexCompute::isModuloInvalidUnswitchedIndex(
    IterDomain* out_concrete_id,
    Val* out_ind,
    Val* inner_extent) const {
  // If not in the unswitched domain map, this domain has no dependent
  // unswitched domain
  auto unswitched_domain_map_it = unswitched_domain_map_.find(out_concrete_id);
  if (unswitched_domain_map_it == unswitched_domain_map_.end()) {
    return false;
  }

  for (const auto& unswitched_domain_list : unswitched_domain_map_it->second) {
    NVF_ERROR(!unswitched_domain_list.empty());

    // If the stride is a multiple of the inner extent, the leaf
    // unswitched index remains to be a valid maximum index as the
    // module by the inner extent will be just zero. More
    // specifically, the index for this unswitched domain would be (x
    // - 1) * extent_of_inner_domain_0 * extent_of_inner_domain_1
    // ...., so if the stride component, i.e., the multiplication of all
    // the inner extents is divisible by the merge inner extent, its
    // contribution propagated to the inner path will be zero. This
    // pattern is effectively the same as distributeDivisibleDivMod in
    // the expr simplifier.
    Val* stride = out_concrete_id->fusion()->oneVal();
    for (auto it = unswitched_domain_list.begin();
         it != unswitched_domain_list.end() - 1;
         ++it) {
      IterDomain* inner_id = *it;
      stride = IrBuilder::mulExpr(stride, getExtent(inner_id));
    }
    if (simplifyExpr(IrBuilder::modExpr(stride, inner_extent))->isZero()) {
      continue;
    }

    // Also, if the total extent including the inner domains is a
    // divisible factor of the inner extent, the contribution by the
    // unswitched domain is guaranteed to be still the maximum when
    // propagated to the inner path. This pattern is effectively the
    // same as distributeGcdRemainderDivMod in the expr simplifier.
    Val* total_extent =
        IrBuilder::mulExpr(stride, getExtent(unswitched_domain_list.back()));
    if (simplifyExpr(IrBuilder::modExpr(inner_extent, total_extent))
            ->isZero()) {
      continue;
    }

    // Not proven to be safe. This does not mean it's proven to be
    // invalid, but it's enough to make the generated code from the
    // existing C++ tests and benchmarks remain unchanged
    return true;
  }

  return false;
}

void IndexCompute::handle(Merge* merge) {
  auto out_id = maybeGetExactMapConcreteID(merge->out());
  auto outer_id = maybeGetExactMapConcreteID(merge->outer());
  auto inner_id = maybeGetExactMapConcreteID(merge->inner());

  auto out_it = index_map_.find(out_id);
  if (out_it == index_map_.end()) {
    return;
  }
  auto out_ind = out_it->second;

  auto zero = GpuLower::current()->kernel()->zeroVal();

  if (isZero(out_id)) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = zero;
    // TODO: Why do we set extent_map_ to zero? This has to be protected by zero
    // merged in, but seems logical to me the extent would still be one.
    extent_map_[outer_id] = zero;
    extent_map_[inner_id] = zero;
    zero_domains_.emplace(outer_id);
    zero_domains_.emplace(inner_id);
    return;
  }

  if (!hasZeroMerged(out_id) && contig_ids_.find(out_id) != contig_ids_.end()) {
    // Contiguous indexing path
    auto input_ids = ir_utils::iterDomainInputsOfOrderedAs(
        {merge->out()}, td_->maybeAllocation());

    // Shouldn't hit this, but don't want to segfault if somehow we do.
    NVF_ERROR(!input_ids.empty());

    // Try to find the last non broadcast entry to put the index in if it's a
    // contiguous merge. This isn't strictly necessary but there's implicit
    // assumptions in the indexing logic that assume broadcasted allocation
    // domains can be ignored. This logic is just to try and match that logic.
    // Initialize everything to zero.
    for (auto alloc_id : input_ids) {
      index_map_[alloc_id] = zero;
    }

    // If all are broadcast we can just send the index to the last entry.
    if (std::all_of(input_ids.begin(), input_ids.end(), [](IterDomain* id) {
          // I don't think reductions can be in here, but strictly matching the
          // logic in the indexing functions like
          // getNonGlobalConsumerStridedIndices
          return id->isBroadcast() || id->isReduction() || id->isStride();
        })) {
      index_map_[*(input_ids.end() - 1)] = out_ind;
    } else {
      for (auto id_it = input_ids.rbegin(); id_it != input_ids.rend();
           id_it++) {
        auto id = *id_it;
        if (id->isBroadcast() || id->isReduction() || id->isStride()) {
          continue;
        } else {
          index_map_[id] = out_ind;
          break;
        }
      }
    }

    return;
  }

  Val* inner_extent = getExtent(inner_id);

  // When the reference has halo extent for inner_id, that extent needs to
  // be used to un-merge
  if (halo_extent_map_.find(inner_id) != halo_extent_map_.end()) {
    inner_extent = halo_extent_map_[inner_id];
  }

  const auto outer_extent = getExtent(outer_id);

  if (inner_id->isBroadcast() && inner_extent->isOneInt()) {
    // Propagate away from broadcast dims
    index_map_[outer_id] = out_ind;
    index_map_[inner_id] = zero;

    extent_map_[outer_id] = getExtent(out_id);
    if (hasZeroMerged(out_id)) {
      zero_merged_in_.insert(outer_id);
    }
  } else if (outer_id->isBroadcast() && outer_extent->isOneInt()) {
    // Propagate away from broadcast dims
    index_map_[outer_id] = zero;
    index_map_[inner_id] = out_ind;

    extent_map_[inner_id] = getExtent(out_id);
    if (hasZeroMerged(out_id)) {
      zero_merged_in_.insert(inner_id);
    }
  } else if (hasZeroMerged(out_id)) {
    // Don't propagate to inner id if it's comprised of only broadcast
    // allocation domains, unless outer is also all broadcast domains. Index
    // shouldn't be anything but zero if both inner and outer are all broadcast
    // domains, but didn't add a hard check for this. See Indexing5 test.
    if (!inner_id->isBroadcast() && !outer_id->isBroadcast()) {
      // If neither dimension is a broadcast (should be true for reference
      // indexing) pick the preferred path or the inner path.
      if (preferred_paths_.find(outer_id) != preferred_paths_.end() &&
          preferred_paths_.find(inner_id) == preferred_paths_.end()) {
        // Marked that we should prop through outer, not inner.
        index_map_[outer_id] = out_ind;
        extent_map_[outer_id] = getExtent(out_id);
        index_map_[inner_id] = zero;
        extent_map_[inner_id] = zero;
        zero_domains_.emplace(inner_id);
      } else {
        // Prop through inner
        index_map_[inner_id] = out_ind;
        extent_map_[inner_id] = getExtent(out_id);
        index_map_[outer_id] = zero;
        extent_map_[outer_id] = zero;
        zero_domains_.emplace(outer_id);
      }
    } else if (inner_id->isBroadcast() && !outer_id->isBroadcast()) {
      // Inner is broadcast and outer isn't, prop through outer
      index_map_[outer_id] = out_ind;
      extent_map_[outer_id] = getExtent(out_id);
      index_map_[inner_id] = zero;
      extent_map_[inner_id] = zero;
      zero_domains_.emplace(inner_id);
    } else {
      // Default to propagating through inner
      index_map_[inner_id] = out_ind;
      extent_map_[inner_id] = getExtent(out_id);
      index_map_[outer_id] = zero;
      extent_map_[outer_id] = zero;
      zero_domains_.emplace(outer_id);
    }
    zero_merged_in_.emplace(inner_id);
    zero_merged_in_.emplace(outer_id);
  } else {
    index_map_[outer_id] = SimplifyingIrBuilder::divExpr(out_ind, inner_extent);
    // Take the absolute maximum if module could result in an invalid
    // index for an unswitched domain
    index_map_[inner_id] =
        isModuloInvalidUnswitchedIndex(out_id, out_ind, inner_extent)
        ? SimplifyingIrBuilder::subExpr(
              inner_extent, inner_extent->fusion()->oneVal())
        : SimplifyingIrBuilder::modExpr(out_ind, inner_extent);
  }
}

void IndexCompute::handle(Swizzle* swizzle) {
  auto out_x_id = maybeGetExactMapConcreteID(swizzle->outX());
  auto out_y_id = maybeGetExactMapConcreteID(swizzle->outY());
  auto in_x_id = maybeGetExactMapConcreteID(swizzle->inX());
  auto in_y_id = maybeGetExactMapConcreteID(swizzle->inY());

  auto out_x_it = index_map_.find(out_x_id);
  auto out_y_it = index_map_.find(out_y_id);

  if (out_x_it == index_map_.end() || out_y_it == index_map_.end()) {
    return;
  }

  const auto out_x_ind = out_x_it->second;
  const auto out_y_ind = out_y_it->second;

  std::pair<Val*, Val*> swizzled_index = dispatchSwizzle(
      swizzle->swizzleType(),
      out_x_ind,
      out_y_ind,
      getExtent(out_x_id),
      getExtent(out_y_id));
  index_map_[in_x_id] = swizzled_index.first;
  index_map_[in_y_id] = swizzled_index.second;
}

void IndexCompute::handle(Swizzle2D* swizzle_2d) {
  auto out_x_id = maybeGetExactMapConcreteID(swizzle_2d->outX());
  auto out_y_id = maybeGetExactMapConcreteID(swizzle_2d->outY());
  auto in_x_id = maybeGetExactMapConcreteID(swizzle_2d->inX());
  auto in_y_id = maybeGetExactMapConcreteID(swizzle_2d->inY());

  auto out_x_it = index_map_.find(out_x_id);
  auto out_y_it = index_map_.find(out_y_id);

  if (out_x_it == index_map_.end() || out_y_it == index_map_.end()) {
    return;
  }

  const auto out_x_ind = out_x_it->second;
  const auto out_y_ind = out_y_it->second;

  if (swizzle_mode_ == SwizzleMode::NoSwizzle ||
      swizzle_mode_ != swizzle_2d->swizzleMode()) {
    // Handle inactive swizzles by just passing through index
    //  and extend information.

    if (!index_map_.count(in_x_id)) {
      index_map_[in_x_id] = out_x_ind;
      extent_map_[in_x_id] = getExtent(out_x_id);
    }
    if (!index_map_.count(in_y_id)) {
      index_map_[in_y_id] = out_y_ind;
      extent_map_[in_y_id] = getExtent(out_y_id);
    }
  } else {
    // Generate integer swizzle math if the
    //  swizzle is activated. See also
    //  [Note on swizzle mode].
    std::pair<Val*, Val*> swizzled_index = dispatchSwizzle(
        swizzle_2d->swizzleType(),
        out_x_ind,
        out_y_ind,
        getExtent(out_x_id),
        getExtent(out_y_id));
    index_map_[in_x_id] = swizzled_index.first;
    index_map_[in_y_id] = swizzled_index.second;
  }
}

void IndexCompute::handle(Resize* resize) {
  auto out_id = maybeGetExactMapConcreteID(resize->out());
  auto in_id = maybeGetExactMapConcreteID(resize->in());

  auto out_it = index_map_.find(out_id);

  if (out_it == index_map_.end()) {
    return;
  }

  const auto out_ind = out_it->second;

  if (isZero(out_id) || hasZeroMerged(out_id)) {
    // When the out ID is (partially) zero, the in ID is not indexable. Don't
    // add any new mapping to the index and extent maps. This is fine since when
    // a resize shows up as part of rfactor transformations, the input to the
    // resize is not indexed as the indexing is done using the rfactor root
    // domain. This could be an issue when a resize is shows up outside of
    // rfactor transfomations, but currently that only can happen when a
    // producer tensor is transformed to look like a consumer. Since inlining is
    // not allowed with resize, the out ID should never be a zero domain in that
    // case.
    return;
  } else {
    index_map_[in_id] = sub(out_ind, resize->leftExpand());
    extent_map_[in_id] = sub(
        sub(getExtent(out_id), resize->leftExpand()), resize->rightExpand());
  }
}

void IndexCompute::dispatch(Expr* e) {
  auto is_expected_type =
      e->isOneOf<Split, Merge, Swizzle, Swizzle2D, Resize>();
  NVF_ERROR(
      is_expected_type, "Invalid expr type found in transform traversal.");
  updateUnswitchedDomains(e);
  BackwardVisitor::dispatch(e);
}

IndexCompute::IndexCompute(
    const TensorDomain* _td,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_map<IterDomain*, Val*> halo_extent_map)
    : IndexCompute(
          _td,
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_domains),
          std::move(zero_merged_in),
          ContigIDs::getNonContigIDs(),
          std::move(preferred_paths),
          std::move(halo_extent_map)) {}

IndexCompute::IndexCompute(
    const TensorDomain* _td,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in,
    const ContigIDs& contig_finder,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_map<IterDomain*, Val*> halo_extent_map,
    std::unordered_set<IterDomain*> unswitched_leaf_domains)
    : td_(_td),
      index_map_(std::move(initial_index_map)),
      extent_map_(std::move(extent_map)),
      zero_domains_(std::move(zero_domains)),
      zero_merged_in_(std::move(zero_merged_in)),
      contig_ids_{contig_finder.contigIDs()},
      preferred_paths_(std::move(preferred_paths)),
      halo_extent_map_(std::move(halo_extent_map)),
      unswitched_leaf_domains_(std::move(unswitched_leaf_domains)) {
  FUSER_PERF_SCOPE("GpuLower::Lower::IndexCompute::IndexCompute");
  // Make sure we recompute any indices we can that map to a contiguous access
  // in physical memory.
  const auto& within_contig = contig_finder.withinContigIDs();
  for (auto contig_id : contig_ids_) {
    if (index_map_.find(contig_id) != index_map_.end()) {
      NVF_ERROR(within_contig.find(contig_id) != within_contig.end());
      for (auto id : within_contig.at(contig_id)) {
        index_map_.erase(id);
      }
    }
  }

  initializeUnswitchDomainMap();
}

IndexCompute::IndexCompute(
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_map<IterDomain*, Val*> halo_extent_map,
    std::unordered_set<IterDomain*> unswitched_leaf_domains)
    : td_{nullptr},
      index_map_(std::move(initial_index_map)),
      zero_domains_(std::move(zero_domains)),
      preferred_paths_(std::move(preferred_paths)),
      halo_extent_map_(std::move(halo_extent_map)),
      concrete_id_pass_{true},
      swizzle_mode_{SwizzleMode::Loop},
      unswitched_leaf_domains_(std::move(unswitched_leaf_domains)) {
  FUSER_PERF_SCOPE("GpuLower::Lower::IndexCompute::IndexCompute");
  initializeUnswitchDomainMap();
}

void IndexCompute::run(const LoopIndexing& loop_indexing) {
  NVF_ERROR(concrete_id_pass_, "concrete pass only for this option");
  // Apply loop swizzles if there are any that outputs to
  //  the loop domains.
  // Currently only support loop swizzles that directly output
  //  to concrete loop domains and these are validated in
  //  validate swizzle pass.
  // TODO:
  //  will gradually enable replaying and mapping of loop
  // swizzles in the IR infrastructure and once that's piped
  // through this part of logic will be removed.
  std::unordered_set<Expr*> visited;
  for (auto loop_id : loop_indexing.loopDomains()) {
    auto loop_id_def = loop_id->definition();
    if (loop_id_def != nullptr && loop_id_def->isA<Swizzle2D>()) {
      if (visited.insert(loop_id_def).second) {
        dispatch(loop_id_def);
      }
    }
  }

  // Resolve the index vals that could be resolved with only
  //  the loops that consumer_tv doesn't share with any of its
  //  consumers, i.e. the not-inlined loops that define consumer_tv
  //  values.
  collectIndexIntoPermissiveMap(loop_indexing);

  // Run through the loop indexing expressions and generate
  //  the indexing integer math for the concrete ids.
  for (auto expr : loop_indexing.getBackwardExprList()) {
    // Resolve missing values from permissive map.
    updateIndexMapFromPermissiveMap(expr);

    dispatch(expr);
  }
}

void IndexCompute::collectIndexIntoPermissiveMap(
    const LoopIndexing& loop_indexing) {
  // Visit the expressions that only produces un-inlined iterdomains,
  //  in reverse topological order.
  for (auto expr : loop_indexing.getBackwardOutOfLineExprList()) {
    // Compute indexing vals for the expression inputs.
    //
    // This stage should run before any indexing computation so it could be
    //  made sure that all index values computed at this stage are
    //  the ones that can be resolved only with the not-inlined
    //  iterdomains.
    //
    auto id_outputs = ir_utils::filterByType<IterDomain>(expr->outputs());
    if (std::all_of(
            id_outputs.begin(), id_outputs.end(), [this](IterDomain* id) {
              return index_map_.count(
                  GpuLower::current()->caMap()->getConcreteMappedID(
                      id, IdMappingMode::EXACT));
            })) {
      // Visit this expression:
      // LoopIndexingAnalysis::traverseFromDomainVals made sure that each
      //  concrete index is bound exactly once so computing these expressions
      //  early should still be consistent.
      dispatch(expr);

      auto id_inputs = ir_utils::filterByType<IterDomain>(expr->inputs());
      for (auto id : id_inputs) {
        // Collect backward pass results from this expression if they are
        //  made available in by this expression.
        auto idx_it =
            index_map_.find(GpuLower::current()->caMap()->getConcreteMappedID(
                id, IdMappingMode::EXACT));

        if (idx_it != index_map_.end()) {
          permissive_index_map_
              [GpuLower::current()->caMap()->getConcreteMappedID(
                  id, IdMappingMode::PERMISSIVE)] = idx_it->second;
        }
      }
    }
  }
}

void IndexCompute::updateIndexMapFromPermissiveMap(const Expr* id_expr) {
  auto id_outputs = ir_utils::filterByType<IterDomain>(id_expr->outputs());
  for (auto id : id_outputs) {
    auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
        id, IdMappingMode::EXACT);
    // Only try to copy index val from permissive map when
    //  the index is missing.
    if (!index_map_.count(concrete_id)) {
      auto permissive_id = GpuLower::current()->caMap()->getConcreteMappedID(
          id, IdMappingMode::PERMISSIVE);
      // Write the permissive index val into index_map_ if the
      //  missing value is found here.
      auto permissive_it = permissive_index_map_.find(permissive_id);
      if (permissive_it != permissive_index_map_.end()) {
        index_map_[concrete_id] = permissive_it->second;
      }
    }
  }
}

void IndexCompute::run() {
  const std::vector<Val*> domain_vals(td_->leaf().begin(), td_->leaf().end());
  traverseTo(domain_vals, false);
}

IterDomain* IndexCompute::maybeGetExactMapConcreteID(IterDomain* id) const {
  if (concrete_id_pass_) {
    return GpuLower::current()->caMap()->getConcreteMappedID(
        id, IdMappingMode::EXACT);
  }
  return id;
}

Val* IndexCompute::getExtent(IterDomain* id) const {
  // Pick from extent_map_ if available. Previously parallel
  // dimensions were ued (e.g., blockDim.x), however, it would result
  // in out-of-bounds errors when the extent of IterDomain is smaller
  // than the threading dimension.
  if (extent_map_.find(id) != extent_map_.end()) {
    return extent_map_.at(id);
  } else {
    return id->extent();
  }
}

bool IndexCompute::hasZeroMerged(IterDomain* id) const {
  return zero_merged_in_.find(id) != zero_merged_in_.end() || isZero(id);
}

bool IndexCompute::isZero(IterDomain* id) const {
  return zero_domains_.find(id) != zero_domains_.end();
}

IndexCompute IndexCompute::updateIndexCompute(
    const TensorDomain* new_td,
    const std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>&
        id_map,
    const ContigIDs& contig_finder) const {
  FUSER_PERF_SCOPE("GpuLower::Lower::updateIndexCompute");

  std::unordered_map<IterDomain*, Val*> updated_index_map;
  std::unordered_map<IterDomain*, Val*> updated_extent_map;
  std::unordered_set<IterDomain*> updated_zero_domains;
  std::unordered_set<IterDomain*> updated_zero_merged_in;
  std::unordered_map<IterDomain*, Val*> updated_halo_extent_map;
  std::unordered_set<IterDomain*> updated_unswitched_domains;

  // Multile IDs can map to the same ID, so loop over the mappings in
  // a deterministic order to have deterministic indexing results
  for (auto prev_id : getSortedKeys(id_map, Statement::lessThan)) {
    const auto& new_ids = id_map.at(prev_id);
    for (auto new_id : new_ids.vector()) {
      if (index_map_.find(prev_id) != index_map_.end()) {
        updated_index_map[new_id] = index_map_.at(prev_id);
      }

      if (extent_map_.find(prev_id) != extent_map_.end()) {
        updated_extent_map[new_id] = getExtent(prev_id);
      }

      if (zero_domains_.find(prev_id) != zero_domains_.end()) {
        updated_zero_domains.emplace(new_id);
      }

      if (zero_merged_in_.find(prev_id) != zero_merged_in_.end()) {
        updated_zero_merged_in.emplace(new_id);
      }

      auto halo_extent_it = halo_extent_map_.find(prev_id);
      if (halo_extent_it != halo_extent_map_.end()) {
        updated_halo_extent_map[new_id] = halo_extent_it->second;
      }

      if (auto it = unswitched_leaf_domains_.find(prev_id);
          it != unswitched_leaf_domains_.end()) {
        updated_unswitched_domains.emplace(new_id);
      }
    }
  }

  IndexCompute updated_index_compute(
      new_td,
      updated_index_map,
      updated_extent_map,
      updated_zero_domains,
      updated_zero_merged_in,
      contig_finder,
      {},
      updated_halo_extent_map,
      updated_unswitched_domains);

  updated_index_compute.run();

  return updated_index_compute;
}

namespace {
// Map indices down to the leaf domains for applying swizzle
class UpdateLeafIndices : public IterVisitor {
 public:
  UpdateLeafIndices(
      const TensorDomain* td,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> extent_map)
      : td_(td),
        index_map_(std::move(initial_index_map)),
        extent_map_(std::move(extent_map)) {
    const std::vector<Val*> domain_vals(td_->leaf().begin(), td_->leaf().end());

    traverseTo(domain_vals, false);
  }

  const std::unordered_map<IterDomain*, Val*>& indexMap() const {
    return index_map_;
  }

  const std::unordered_map<IterDomain*, Val*>& extentMap() const {
    return extent_map_;
  }

 private:
  using IterVisitor::handle;

  void handle(Split* split) override {
    auto in_id = split->in();
    auto outer_id = split->outer();
    auto inner_id = split->inner();

    // Nothing need to be done when mappings for the output axes
    // already exist.
    if (index_map_.find(outer_id) != index_map_.end()) {
      NVF_ERROR(
          index_map_.find(inner_id) != index_map_.end(),
          "Outer exists but inner not found");
      return;
    }

    if (!index_map_.count(in_id)) {
      // Reduction axes on producer side could be visited on forward
      //  propagation pass and current implementation does not yet
      //  support reduction on swizzled iterdomains, so un-indexed
      //  reduction iterdomains are just ignored for now. It is the same
      //  for broadcast iterdomains.
      NVF_ERROR(
          in_id->isReduction() || in_id->isBroadcast(),
          "Undefined index for ",
          in_id->toString());
      return;
    }

    auto factor = split->factor();
    index_map_[inner_id] =
        SimplifyingIrBuilder::modExpr(index_map_[in_id], factor);
    extent_map_[inner_id] = factor;
    index_map_[outer_id] =
        SimplifyingIrBuilder::divExpr(index_map_[in_id], factor);
    extent_map_[outer_id] =
        SimplifyingIrBuilder::ceilDivExpr(getExtent(in_id), factor);
  }

  void handle(Merge* merge) override {
    auto out_id = merge->out();
    auto outer_id = merge->outer();
    auto inner_id = merge->inner();

    // Nothing need to be done when mappings for the output axes
    // already exist.
    if (index_map_.find(out_id) != index_map_.end()) {
      return;
    }

    if (outer_id->isBroadcast()) {
      if (!index_map_.count(inner_id)) {
        // Reduction axes on producer side could be visited on forward
        //  propagation pass and current implementation does not yet
        //  support reduciton on swizzled iterdomains, so un-indexed
        //  reduction iterdomains are just ignored for now. The same applies to
        //  BroadcastOp.
        NVF_ERROR(
            inner_id->isReduction() || inner_id->isBroadcast(),
            "Undefined index for ",
            inner_id->toString());
        return;
      }

      NVF_ERROR(
          index_map_.find(inner_id) != index_map_.end(), "Inner ID not found");

      index_map_[out_id] = index_map_[inner_id];
      extent_map_[out_id] = getExtent(inner_id);
      return;
    } else if (inner_id->isBroadcast()) {
      if (!index_map_.count(outer_id)) {
        // Reduction axes on producer side could be visited on forward
        //  propagation pass and current implementation does not yet
        //  support reduciton on swizzled iterdomains, so un-indexed
        //  reduction iterdomains are just ignored for now.
        NVF_ERROR(
            outer_id->isReduction() || outer_id->isBroadcast(),
            "Undefined index for ",
            outer_id->toString());
        return;
      }

      NVF_ERROR(
          index_map_.find(outer_id) != index_map_.end(), "Outer ID not found");

      index_map_[out_id] = index_map_[outer_id];
      extent_map_[out_id] = getExtent(outer_id);
      return;
    }

    if (!index_map_.count(outer_id) || !index_map_.count(inner_id)) {
      // Reduction axes on producer side could be visited on forward
      //  propagation pass and current implementation does not yet
      //  support reduciton on swizzled iterdomains, so un-indexed
      //  reduction iterdomains are just ignored for now.
      NVF_ERROR(
          (outer_id->isReduction() || outer_id->isBroadcast()) &&
              (inner_id->isReduction() || inner_id->isBroadcast()),
          "Undefined index for ",
          outer_id->toString(),
          " and ",
          inner_id->toString());
      return;
    }

    NVF_ERROR(
        index_map_.find(outer_id) != index_map_.end(), "Outer ID not found");
    NVF_ERROR(
        index_map_.find(inner_id) != index_map_.end(), "Inner ID not found");

    index_map_[out_id] = SimplifyingIrBuilder::addExpr(
        index_map_[inner_id],
        SimplifyingIrBuilder::mulExpr(
            index_map_[outer_id], getExtent(inner_id)));

    extent_map_[out_id] =
        SimplifyingIrBuilder::mulExpr(getExtent(outer_id), getExtent(inner_id));
  }

  void handle(Swizzle2D* swizzle_2d) override {
    auto in_x = swizzle_2d->inX();
    auto in_y = swizzle_2d->inY();
    auto out_x = swizzle_2d->outX();
    auto out_y = swizzle_2d->outY();

    // Forward propagation pass still just forward
    //  through the indices and the actual swizzle
    //  will be applied on the backward pass in
    //  IndexSwizzle class implementation.
    index_map_[out_x] = index_map_.at(in_x);
    extent_map_[out_x] = getExtent(in_x);
    index_map_[out_y] = index_map_.at(in_y);
    extent_map_[out_y] = getExtent(in_y);
  }

  // return extent_map_[id] if exists, else return id->extent()
  Val* getExtent(IterDomain* id) {
    if (extent_map_.find(id) != extent_map_.end()) {
      return extent_map_.at(id);
    } else {
      return id->extent();
    }
  }

 private:
  const TensorDomain* td_;
  std::unordered_map<IterDomain*, Val*> index_map_;
  std::unordered_map<IterDomain*, Val*> extent_map_;
};

// Returns halo-extended extent if id has halo. Otherwise, just
// returns id->extent.
Val* getHaloExtentOfRootAxis(IterDomain* id, Val* normal_extent = nullptr) {
  // If id is device dim, ignore the extent which holds the unsharded extent.
  if (id->isDeviceDim()) {
    normal_extent = GpuLower::current()->kernel()->oneVal();
  } else if (normal_extent == nullptr) {
    normal_extent = id->extent();
  }

  const auto& halo = GpuLower::current()->haloInfo()->getRootAxisInfo(id);
  if (halo.hasHalo()) {
    auto halo_extent = SimplifyingIrBuilder::addExpr(
        normal_extent,
        SimplifyingIrBuilder::create<Val>(
            (int64_t)halo.width(), DataType::Index));
    return halo_extent;
  } else {
    return normal_extent;
  }
}

} // namespace

IndexSwizzle::IndexSwizzle(
    const TensorView* tv,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in)
    : IndexCompute(
          tv->domain(),
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_domains),
          std::move(zero_merged_in)),
      tv_(tv) {}

IndexSwizzle::IndexSwizzle(
    const TensorView* tv,
    const TensorDomain* domain,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in)
    : IndexCompute(
          domain,
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_domains),
          std::move(zero_merged_in)),
      tv_(tv) {}

void IndexSwizzle::run() {
  if (tv_->hasSwizzleOp()) {
    // Propagate backward for the annotated swizzle path.
    // TODO:
    //  eventually will unify the two swizzling implementation
    //  code path in a follow up. Currently just focusing on
    //  getting the necessary implementation of the swizzle
    //  operator ready.
    //
    // At this intermediate state, the legacy swizzle implementation
    //  takes precedence, i.e. whenever swizzle_type_ is not NoSwizzle,
    //  the new swizzle op pass is disabled.
    UpdateLeafIndices update_leaves(td_, indexMap(), extentMap());
    index_map_ = update_leaves.indexMap();
    extent_map_ = update_leaves.extentMap();
    IndexCompute::swizzle_mode_ = SwizzleMode::Data;
    IndexCompute::run();
  }
}

void IndexSwizzle::dispatch(Expr* e) {
  auto out_ids = ir_utils::filterByType<IterDomain>(e->outputs());
  bool needs_update =
      std::any_of(
          out_ids.begin(),
          out_ids.end(),
          [this](IterDomain* id) {
            return swizzled_ids_.find(id) != swizzled_ids_.end();
          }) ||
      (e->isA<Swizzle2D>() &&
       e->as<Swizzle2D>()->swizzleType() != Swizzle2DType::NoSwizzle &&
       e->as<Swizzle2D>()->swizzleMode() == SwizzleMode::Data);
  if (!needs_update) {
    return;
  }

  IndexCompute::dispatch(e);
  for (auto input : ir_utils::filterByType<IterDomain>(e->inputs())) {
    swizzled_ids_.insert(input);
  }
}

void IndexSwizzle::handle(Swizzle2D* swizzle_2d) {
  auto out_x_id = swizzle_2d->outX();
  auto out_y_id = swizzle_2d->outY();

  auto out_x_it = index_map_.find(out_x_id);
  auto out_y_it = index_map_.find(out_y_id);

  NVF_ERROR(
      out_x_it != index_map_.end() && out_y_it != index_map_.end(),
      "Swizzle output indices were not propagated through");

  IndexCompute::handle(swizzle_2d);
}

namespace {

//! Check if the index of a parallel loop should be subsituted with
//! zero.
//!
//! Zero substitution only happens with the BID parallel types with
//! Local Or Shared tensors or the TID parallel types with Local
//! tensors.
//!
//! This check is straightforward for consumers, but for producers
//! the substitution is only done when the producer uses the same
//! parallel type as the loop parallel type.
//!
//! If there's a mapped producer IterDoamin and that ID is
//! parallelized, there are a couple of cases depending on the
//! parallel type and the producer memory type:
//!
//! Loop PT, producer PT, producer mem -> index
//! - BID, TID/Serial, Shared / Local -> use BID
//! - BID, BID, Shared / Local -> use zero when loop PT == producer PT
//! - BID, BID, Shared / Local -> invalid when loop PT != producer PT
//! - TID, Serial, Local -> use TID
//! - TID, TID, Local -> use zero when loop PT == producer PT
//! - TID, TID, Local -> invalid when loop PT != producer PT
//!
//! The invalid cases should not happen here as they should be already
//! detected as invalid parallelization. Thus, we just need to find if
//! the producer has a mapped IterDomain that has the same parallel
//! type as the loop IterDomain.
bool isParallelLoopIndexSubstitutedAsZero(
    const TensorView* tv,
    IterDomain* loop_id,
    bool as_consumer,
    bool within_mma_loops) {
  const auto ca_map = GpuLower::current()->caMap();

  // MMA operands are currently indexed in units of "fragments",
  //  so each mma tensor domain would be zero-ed and the tensor index
  //  calculated here would be the fragment index.
  // TODO: This is a quick WAR to enable iterating over a register array
  //  of MMA fragments, so we could generate unrolled mma loops.
  //  Eventually we still want IdGraph to be able to analyze the
  //  in-register layout of mma fragments for more unified indexing math
  //  as well as more flexibility in swizzling loops.
  if (loop_id->isMma() && !as_consumer) {
    return true;
  }

  const bool is_shared = tv->getMemoryType() == MemoryType::Shared;
  const bool is_local = tv->getMemoryType() == MemoryType::Local;

  if (!((loop_id->isBlockDim() && (is_shared || is_local)) ||
        (loop_id->isThread() && is_local))) {
    return false;
  }

  // If this is for a consumer, the above check is sufficient
  if (as_consumer) {
    return true;
  }

  // Note && TODO:
  // mma swizzled lane_id does not map naturally from producer
  // to consumer but they should still be detected as same
  // parallel type. In a follow up may want to extend
  // find_matching_parallel_domain to cover this case.
  if ((within_mma_loops || ir_utils::isLdMatrixOp(tv->definition())) &&
      loop_id->getParallelType() == ParallelType::TIDx) {
    return true;
  }

  // When indexing a producer, additional checks are required as
  // mentioned above
  auto producer_tv = tv;
  auto it = std::find_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      [&](IterDomain* tv_id) {
        // Matching is done using the index and loop maps. See
        // validateParallelize as well.
        return ca_map->areMapped(loop_id, tv_id, IdMappingMode::EXACT) ||
            ca_map->areMapped(loop_id, tv_id, IdMappingMode::PERMISSIVE);
      });

  // There's no mapped producer ID. Zero substitution shouldn't be
  // done.
  if (it == tv->getLeafDomain().end()) {
    return false;
  }

  // Producer ID that corresponds to the loop ID
  IterDomain* producer_id = *it;

  // If the loop PT and producer PT are the same, the producer ID can
  // be indexed as just zero. Otherwise, it must use the loop parallel
  // type as its index.

  // Sanity check when not substituted, i.e., when the producer ID
  // uses a different as the loop PT. Not necessary as these
  // conditions are already validated, but just double checking.

  if (loop_id->getParallelType() != producer_id->getParallelType()) {
    NVF_ERROR(
        (loop_id->isBlockDim() && !producer_id->isBlockDim()) ||
            (loop_id->isThreadDim() && !producer_id->isThread()),
        "Found invalid parallelization that should have been detected by the parallel validation: loop ID: ",
        loop_id->toString(),
        ", producer: ",
        producer_tv->toString());
  }

  return producer_id->getParallelType() == loop_id->getParallelType();
}

} // namespace

// Used for local and shared index mapping. Returns a map from loops
// to loop indices as well as a set of loops that do not contribute to
// indexing.
std::pair<
    std::unordered_map<kir::ForLoop*, Val*>,
    std::unordered_set<kir::ForLoop*>>
indexMapFromTV(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    kir::ForLoop* alloc_loop,
    bool as_consumer,
    kir::ForLoop* double_buffer_loop) {
  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  const bool is_global = tv->getMemoryType() == MemoryType::Global;
  const bool is_shared = tv->getMemoryType() == MemoryType::Shared;

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;

  // Check if the current op has an implicit loop implemented
  //  within an mma instruction.
  bool within_mma_loops =
      std::any_of(loops.begin(), loops.end(), [](kir::ForLoop* fl) {
        return fl->iter_domain()->isMma();
      });

  // Track domains that do not contibute to the resulting
  // index. Previously, index->isZeroInt() was used to detect such
  // domains, but that's not a reliable method as we may set an
  // initial index to zero for unswitch.
  std::unordered_set<kir::ForLoop*> zero_loops;

  for (auto loop : loops) {
    Val* idx = nullptr;
    // See also LoopNestGenerator::pushAlloc.
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (!within_alloc) {
      if ((loop->iter_domain()->isThreadDim() && is_shared) ||
          (loop->iter_domain()->isThread() && is_global)) {
        idx = loop->indexOrStartIfTrivial();
      } else {
        idx = GpuLower::current()->kernel()->zeroVal();
        zero_loops.insert(loop);
      }
    } else if (isParallelLoopIndexSubstitutedAsZero(
                   tv, loop->iter_domain(), as_consumer, within_mma_loops)) {
      idx = GpuLower::current()->kernel()->zeroVal();
      zero_loops.insert(loop);
    } else {
      idx = loop->indexOrStartIfTrivial();
    }

    if (rotated_loops.count(loop) > 0 && zero_loops.count(loop) == 0) {
      idx = SimplifyingIrBuilder::addExpr(idx, loop->step());
    }

    if (loop == double_buffer_loop) {
      const int64_t stage_depth =
          GpuLower::current()->doubleBufferInfo().getStageDepthFor(
              loop->iter_domain());
      idx = SimplifyingIrBuilder::addExpr(
          idx,
          SimplifyingIrBuilder::create<Val>(stage_depth - 1L, DataType::Index));
    }

    loop_to_ind_map[loop] = idx;

    if (!within_alloc && loop == alloc_loop) {
      within_alloc = true;
    }
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return {loop_to_ind_map, zero_loops};
}

//! Set "pragma unroll" required for loops that indexing of Local
//! tensors depends on.
//!
//! \param tv Indexed tensor
//! \param alloc_loop Allocation loop of tv
//! \param loops The current loop structure
//! \param id_map Producer-to-consumer map in case of indexing as producer
void ensureStaticIndexing(
    const TensorView* tv,
    kir::ForLoop* alloc_loop,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map) {
  if (tv->getMemoryType() != MemoryType::Local) {
    return;
  }

  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  for (auto loop : loops) {
    if (!within_alloc) {
      if (loop == alloc_loop) {
        within_alloc = true;
      }
      continue;
    }
    IterDomain* loop_id = loop->iter_domain();
    if (loop->vectorize() || loop_id->isThread()) {
      continue;
    }
    // Look for a domain that is mapped with the loop. If mapped in
    // the loop map, the loop index should be used for indexing of the
    // tensor, except for broadcast and reduction domains.
    auto it = std::find_if(
        tv->getLeafDomain().begin(),
        tv->getLeafDomain().end(),
        [loop_id, &id_map](IterDomain* id) {
          if (id->isBroadcast() || id->isReduction() || id->isStride()) {
            return false;
          }
          auto id_replacement = id_map.find(id);
          if (id_replacement != id_map.end()) {
            id = id_replacement->second;
          }
          return GpuLower::current()->caMap()->areMapped(
              loop_id, id, IdMappingMode::PERMISSIVE);
        });
    if (it != tv->getLeafDomain().end()) {
      loop->requireUnroll();
    }
  }
}

namespace {

std::unordered_map<IterDomain*, IterDomain*> invertOneToOneMap(
    const std::unordered_map<IterDomain*, IterDomain*>& map) {
  std::unordered_map<IterDomain*, IterDomain*> inverted;
  for (const auto& kv : map) {
    bool inserted = inverted.emplace(kv.second, kv.first).second;
    NVF_ERROR(
        inserted,
        "Multiple mappings to the same value detected: ",
        kv.second->toString());
  }
  return inverted;
}

} // namespace

std::vector<Val*> Index::getGlobalProducerStridedIndices(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getGlobalProducerIndex");

  auto alloc_indices = getProducerAllocationIndices(
      producer_tv, consumer_tv, loops, rotated_loops, override_index);

  const auto& alloc_dom = producer_tv->getMaybeAllocationDomain();

  // TODO: Abstract stride logic to reuse with consumer indexing
  std::vector<Val*> strides(alloc_dom.size(), nullptr);
  {
    int stride_i = 0;
    for (const auto i : c10::irange(alloc_dom.size())) {
      if (alloc_dom[i]->isReduction()) {
        strides[i] = GpuLower::current()->kernel()->oneVal();
        continue;
      }
      strides[i] = IrBuilder::getItemExpr(
          IrBuilder::getAttrExpr(
              IrBuilder::metadataExpr(producer_tv), "alloc_stride"),
          (int64_t)stride_i++);
    }
  }

  NVF_ERROR(alloc_dom.size() == producer_tv->domain()->contiguity().size());
  Val* cur_contig_stride = GpuLower::current()->kernel()->oneVal();
  for (const auto i : c10::irange(alloc_dom.size())) {
    auto dim = alloc_dom.size() - i - 1;
    if (alloc_dom[dim]->isReduction()) {
      continue;
    }

    auto producer_dim_contiguity = producer_tv->domain()->contiguity().at(dim);
    if (alloc_dom[dim]->isBroadcast()) {
      strides[dim] = cur_contig_stride->fusion()->zeroVal();
      NVF_ERROR(!producer_dim_contiguity.has_value());
    } else if (!producer_dim_contiguity.has_value()) {
      NVF_ERROR(false, "Expected value for dimension contiguity");
    } else if (producer_dim_contiguity.value()) {
      // If contig, used the stored stride which may be the previous
      // dimensions stride * previous dimensions size
      strides[dim] = cur_contig_stride;
      // Prepare for the next dimension which may also be contiguous, multiply
      // by extent of this dimension
      auto alloc_dim_extent = getHaloExtentOfRootAxis(alloc_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(cur_contig_stride, alloc_dim_extent);
    } else {
      // If non contiguous dimension, keep local stride information, set cur
      // stride to local stride * local raw extent
      auto alloc_dim_extent = getHaloExtentOfRootAxis(alloc_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(strides[dim], alloc_dim_extent);
    }
  }

  auto vectorize_shift =
      loops.empty() ? nullptr : loops.back()->vectorize_shift();

  // Global striding
  std::vector<Val*> strided_inds(
      alloc_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : c10::irange(alloc_dom.size())) {
    Val* alloc_ind = alloc_indices.at(i);

    if (alloc_ind->isZeroInt()) {
      continue;
    } else {
      auto strided_ind = SimplifyingIrBuilder::mulExpr(alloc_ind, strides[i]);
      if (i == alloc_dom.size() - 1 && vectorize_shift != nullptr) {
        strided_inds[i] =
            SimplifyingIrBuilder::addExpr(strided_ind, vectorize_shift);
      } else {
        strided_inds[i] = strided_ind;
      }
    }
  }

  return strided_inds;
}

namespace {

// Maps all producer domains to consumer with broadcast
// forwarding. Used to find the allocation position.
std::unordered_map<IterDomain*, IterDomain*> mapAllProducerDomainsToConsumer(
    TensorView* producer_tv,
    const TensorView* consumer_tv) {
  // This map has forwarded broadcast axes, it should only be used to compute
  // the allocation position of the producer
  std::unordered_map<IterDomain*, IterDomain*> p2c_alloc_map;

  //  We want to replay producer as consumer instead of the other way around
  //  since consumer may have some broadcasted axes producer doesn't have
  //  merged into loops producer may use. If we did consumer as producer we
  //  wouldn't have this information in the mapping.
  auto replay_PasC = BestEffortReplay::replayPasC(
      producer_tv,
      consumer_tv,
      -1,
      PairwiseRootDomainMap(producer_tv, consumer_tv));

  // Grab consumer domain entries and reverse replay map. TODO: Maybe
  // TransformReplay::replayPasC could return this map
  for (auto id : consumer_tv->getLeafDomain()) {
    const auto& c2p_map = replay_PasC.getReplay();
    auto c2p_it = c2p_map.find(id);
    if (c2p_it != c2p_map.end()) {
      auto c_id = c2p_it->first;
      auto p_id = c2p_it->second;
      p2c_alloc_map[p_id] = c_id;
    }
  }

  return p2c_alloc_map;
}

Val* sumVals(std::vector<Val*> vals) {
  Val* result_index = GpuLower::current()->kernel()->zeroVal();
  for (auto v : vals) {
    result_index = SimplifyingIrBuilder::addExpr(result_index, v);
  }
  return result_index;
}

} // namespace

// Producer index for either shared or local memory
std::vector<Val*> Index::getNonGlobalProducerStridedIndices(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  bool is_mma_input = consumer_tv->definition()->isA<MmaOp>();
  const auto gpu_lower = GpuLower::current();
  // Replay producer to look like consumer so we can index on producer since our
  // loop nests look like consumer
  auto pairwise_map = PairwiseRootDomainMap(producer_tv, consumer_tv);
  // Resize ops can be and should be replayed.
  auto producer_replayed_as_consumer =
      TransformReplay::replayPasC(
          producer_tv,
          consumer_tv,
          -1,
          pairwise_map,
          TransformReplayOptions().replayResize())
          .first;

  ir_utils::TVDomainGuard domain_guard(
      producer_tv, producer_replayed_as_consumer);
  const auto p2c_alloc_map =
      mapAllProducerDomainsToConsumer(producer_tv, consumer_tv);

  // Map everything we can from reference to producer using compute at index
  // map. All producer id's don't exist in the compute at map. The rfactor axes
  // all may be, but since I haven't proven that to be the case, going to do a
  // more conservative approach, which is to use the consumer as a proxy between
  // producer to reference.
  std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_producer;
  std::unordered_map<IterDomain*, IterDomain*> c2p_index_map;

  // Map sent to best effort replay needs to match the exact incantation for
  // compute_at_mode.cpp with MappingMode::Index
  auto c2p_root_map = PairwiseRootDomainMap(producer_tv, consumer_tv)
                          .mapBroadcast(false)
                          .mapConsumerToProducer();

  // This replay has to be consistent with compute at index map.
  BestEffortReplay replay_producer_as_consumer(
      producer_tv->getLeafDomain(), consumer_tv->getLeafDomain(), c2p_root_map);

  c2p_index_map = replay_producer_as_consumer.getReplay();

  const auto& producer_indexing_from_idgraph = getTensorIndexFromIdGraph(
      loops, rotated_loops, consumer_tv, producer_tv, false, c2p_index_map);

  const auto& producer_indexing = producer_indexing_from_idgraph.index;

  IndexSwizzle index_swizzle(
      producer_tv,
      producer_indexing.indexMap(),
      producer_indexing.extentMap(),
      producer_indexing.zeroDomains(),
      producer_indexing.zeroMergedIn());

  index_swizzle.run();

  auto producer_swizzled_index = index_swizzle;

  if (producer_tv->hasSwizzleOp()) {
    // Special handling needed on the new swizzle
    //  op pass:
    //  each swizzle op is local to the tensor,
    //  so ReplayPasC will not include the swizzle
    //  ops on the producer iterdomain. So would
    //  need to traverse forward the producer domain
    //  before the replay to get the swizzle ops.
    IndexSwizzle producer_swizzle2d(
        producer_tv,
        domain_guard.prevDomain(),
        producer_indexing.indexMap(),
        producer_indexing.extentMap(),
        producer_indexing.zeroDomains(),
        producer_indexing.zeroMergedIn());
    producer_swizzle2d.run();
    producer_swizzled_index = producer_swizzle2d;
  }

  // TODO: merge the two swizzle compute logic once the new one is ready.
  //  will need to replace cyclic shift swizzle with xor since swizzle2d
  //  doesn't have cyclic shift.
  const auto& index_map = producer_swizzled_index.indexMap();

  const auto& extent_map = producer_indexing.extentMap();
  const auto& zero_domain_map = producer_indexing.zeroDomains();
  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  const auto& alloc_dom = producer_tv->getMaybeAllocationDomain();

  // Figure out which alloc axes we don't need to index
  std::unordered_set<IterDomain*> skip_indexing;

  for (auto alloc_id : alloc_dom) {
    // Already taken care of because we can detect no indexing required
    if (alloc_id->isBroadcast() || alloc_id->isReduction() ||
        alloc_id->isStride() || alloc_id->isDeviceDim() ||
        (alloc_id->isThread() &&
         producer_tv->getMemoryType() == MemoryType::Local)) {
      skip_indexing.insert(alloc_id);
      continue;
    }

    // Already an entry for this allocation domain, continue
    if (index_map.find(alloc_id) != index_map.end()) {
      continue;
    }
  }

  std::vector<Val*> strided_inds(
      alloc_dom.size(), GpuLower::current()->kernel()->zeroVal());

  // MMA operation op is a special operation that our automatic "zero domain"
  // analysis of our current indexing approach does not work. So we need to
  // manually specify which dimensions are used for MMA allocation.
  std::function<bool(const IterDomain* id)> is_mma_allocation;
  if (is_mma_input) {
    int size = (int)alloc_dom.size();
    const IterDomain* allocation0 = alloc_dom.at(size - 3);
    const IterDomain* allocation1 = alloc_dom.at(size - 2);
    const IterDomain* allocation2 = alloc_dom.at(size - 1);
    is_mma_allocation = [=](const IterDomain* id) {
      return id == allocation0 || id == allocation1 || id == allocation2;
    };
  } else {
    is_mma_allocation = [](const IterDomain* id) { return false; };
  }

  for (const auto i : c10::irange(alloc_dom.size())) {
    if (skip_indexing.count(alloc_dom[i])) {
      continue;
    }

    auto override_it = override_index.find(alloc_dom[i]);
    const bool is_overriden = override_it != override_index.end();

    NVF_ERROR(
        is_overriden || index_map.find(alloc_dom[i]) != index_map.end(),
        "Couldn't find allocation mapping for ",
        producer_tv->toString(),
        " dim: ",
        i,
        " id: ",
        alloc_dom[i]->toString());

    auto alloc_ind_i =
        is_overriden ? override_it->second : index_map.at(alloc_dom[i]);

    alloc_ind_i = getProducerIndexWithHalo(
        producer_tv, i, alloc_ind_i, consumer_tv, is_overriden);

    alloc_ind_i = getProducerIndexWithGather(
        alloc_ind_i,
        i,
        producer_tv,
        consumer_tv,
        producer_indexing_from_idgraph.concrete_index.indexMap());

    alloc_ind_i = getProducerIndexWithPartialSplit(
        alloc_ind_i, alloc_dom[i], producer_tv, consumer_tv);

    if (alloc_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (const auto j : c10::irange(i + 1, alloc_dom.size())) {
      if (skip_indexing.count(alloc_dom[j])) {
        continue;
      }

      auto alloc_ext_j = (extent_map.find(alloc_dom[j]) == extent_map.end() ||
                          is_mma_allocation(alloc_dom[j]))
          ? alloc_dom[j]->extent()
          : extent_map.at(alloc_dom[j]);

      alloc_ext_j = getHaloExtentOfRootAxis(alloc_dom[j], alloc_ext_j);

      if (zero_domain_map.count(alloc_dom[j]) == 0 ||
          is_mma_allocation(alloc_dom[j])) {
        if (stride == nullptr) {
          stride = alloc_ext_j;
        } else {
          stride = SimplifyingIrBuilder::mulExpr(stride, alloc_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds[i] = SimplifyingIrBuilder::mulExpr(alloc_ind_i, stride);
    } else {
      strided_inds[i] = alloc_ind_i;
    }
  }

  if (producer_tv->isDoubleBuffered() || producer_tv->isCircularBuffered()) {
    auto db_loop = gpu_lower->doubleBufferInfo().getDoubleBufferLoop(
        producer_tv, loops, true);
    if (db_loop != nullptr) {
      const auto stage_depth =
          (int64_t)gpu_lower->doubleBufferInfo().getStageDepthFor(
              db_loop->iter_domain());
      auto loop_index = db_loop->indexOrStartIfTrivial();
      if (rotated_loops.count(db_loop) > 0) {
        loop_index = SimplifyingIrBuilder::addExpr(loop_index, db_loop->step());
      }
      auto db_switch_index = SimplifyingIrBuilder::modExpr(
          loop_index,
          SimplifyingIrBuilder::create<Val>(stage_depth, DataType::Index));
      auto original_alloc_size =
          gpu_lower->doubleBufferInfo().getOriginalAllocSize(producer_tv);
      auto db_strided_index =
          SimplifyingIrBuilder::mulExpr(db_switch_index, original_alloc_size);
      strided_inds.push_back(db_strided_index);
    }
  }

  return strided_inds;
}

Val* Index::getLinearLogicalIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops) {
  auto guard = ir_utils::allocateToRFactorDomainGuard(consumer_tv, true);
  return sumVals(
      getGlobalConsumerStridedIndices(consumer_tv, loops, rotated_loops));
}

std::vector<Val*> Index::getConsumerPerDimLogicalIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops) {
  auto guard = ir_utils::allocateToRFactorDomainGuard(consumer_tv, false);
  IndexFromIdGraph index_from_id_graph =
      getTensorIndexFromIdGraph(loops, rotated_loops, consumer_tv);
  return getConsumerAllocationIndices(consumer_tv, loops, index_from_id_graph);
}

std::vector<Val*> Index::getProducerPerDimLogicalIndex(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  auto guard = ir_utils::allocateToRFactorDomainGuard(producer_tv, false);
  return getProducerAllocationIndices(
      producer_tv, consumer_tv, loops, rotated_loops, override_index);
}

std::vector<Val*> Index::getStrides(TensorView* tv) {
  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  const auto& alloc_dom = tv->getMaybeAllocationDomain();

  std::vector<Val*> strides(
      alloc_dom.size(), GpuLower::current()->kernel()->oneVal());
  {
    int stride_i = 0;
    for (const auto i : c10::irange(alloc_dom.size())) {
      if (alloc_dom[i]->isReduction() || alloc_dom[i]->isStride()) {
        strides[i] = GpuLower::current()->kernel()->oneVal();
        continue;
      }
      strides[i] = IrBuilder::getItemExpr(
          IrBuilder::getAttrExpr(IrBuilder::metadataExpr(tv), "alloc_stride"),
          (int64_t)stride_i++);
    }
  }

  NVF_ERROR(alloc_dom.size() == tv->domain()->contiguity().size());
  Val* cur_contig_stride = GpuLower::current()->kernel()->oneVal();
  for (const auto i : c10::irange(alloc_dom.size())) {
    auto dim = alloc_dom.size() - i - 1;
    if (alloc_dom[dim]->isReduction() || alloc_dom[dim]->isStride()) {
      continue;
    }

    auto dim_contiguity = tv->domain()->contiguity().at(dim);
    if (alloc_dom[dim]->isBroadcast()) {
      strides[dim] = cur_contig_stride->fusion()->zeroVal();
      NVF_ERROR(!dim_contiguity.has_value());
    } else if (!dim_contiguity.has_value()) {
      NVF_ERROR(false, "Expected value for dimension contiguity");
    } else if (dim_contiguity.value()) {
      // If contig, used the stored stride which may be the previous
      // dimensions stride * previous dimensions size
      strides[dim] = cur_contig_stride;
      // Prepare for the next dimension which may also be contiguous, multiply
      // by extent of this dimension
      auto alloc_dim_extent = getHaloExtentOfRootAxis(alloc_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(cur_contig_stride, alloc_dim_extent);
    } else {
      // If non contiguous dimension, keep local stride information, set cur
      // stride to local stride * local raw extent
      cur_contig_stride = SimplifyingIrBuilder::mulExpr(
          strides[dim], getHaloExtentOfRootAxis(alloc_dom[dim]));
    }
  }
  return strides;
}

std::vector<Val*> Index::getConsumerAllocationIndices(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const IndexFromIdGraph& index_from_id_graph) {
  const auto& alloc_dom = tv->getMaybeAllocationDomain();
  auto indexing = index_from_id_graph.index;

  std::vector<Val*> alloc_inds(
      alloc_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : c10::irange(alloc_dom.size())) {
    // See a comment in indexing to allocation domains in
    // getGlobalProducerIndex.
    if (alloc_dom[i]->isReduction() || alloc_dom[i]->isBroadcast() ||
        alloc_dom[i]->isStride()) {
      continue;
    }

    NVF_ERROR(
        indexing.indexMap().find(alloc_dom[i]) != indexing.indexMap().end(),
        "Couldn't find allocation mapping for ",
        tv->toString(),
        " dim: ",
        i,
        " id: ",
        alloc_dom[i]->toString());

    auto alloc_ind = indexing.indexMap().at(alloc_dom[i]);

    alloc_ind = SimplifyingIrBuilder::addExpr(
        alloc_ind, getGlobalConsumerOffsetWithPartialSplit(alloc_dom[i]));
    alloc_inds[i] = alloc_ind;
  }
  return alloc_inds;
}

std::vector<Val*> Index::getProducerAllocationIndices(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getProducerAllocationIndices");
  // Replay producer to look like consumer so we can index on producer since
  // our loop nests look like consumer
  auto pairwise_map =
      PairwiseRootDomainMap(producer_tv, consumer_tv).mapBroadcast(true);

  TensorDomain* producerAsC = TransformReplay::replayPasC(
                                  producer_tv,
                                  consumer_tv,
                                  -1,
                                  pairwise_map,
                                  TransformReplayOptions().replayResize())
                                  .first;

  // Make the producer_tv look like consumer while performing indexing math
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // Map sent to best effort replay needs to match the exact incantation for
  // compute_at_mode.cpp with MappingMode::Index
  auto c2p_root_map = PairwiseRootDomainMap(producer_tv, consumer_tv)
                          .mapBroadcast(false)
                          .mapConsumerToProducer();

  // This replay has to be consistent with compute at index map.
  BestEffortReplay replay_producer_as_consumer(
      producer_tv->getLeafDomain(), consumer_tv->getLeafDomain(), c2p_root_map);

  auto c2p_map = replay_producer_as_consumer.getReplay();

  // Make sure at least root domains are mapped even when extents may
  // be different. This mapping is important for the indexing lookup
  // tensors of PyTorch gather as a producer. The IDs of a lookup
  // tensor may have larger extents than those of the corresponding
  // output tensor, but the index expressions to those output IDs can
  // still be used for the producer. Note that we always do not map
  // the indirectly accessed ID and its corresponding output ID. The
  // above relaxed mapping is only for the rest of the IDs.
  //
  // Note that when the consumer has swizzle, the swizzle are skipped. For
  // example, if we have:
  //   consumer:
  //     root: I0, I1, I2
  //     leaf: I0, I3, I4
  //   producer:
  //     root I5, I6, I7
  // where I3, I4 = swizzle(I1, I2) , then the c2p map will be I3->I6, I4->I7,
  // I1 and I2 are not mapped. For this case, we should allow the root unmapped,
  // If we add I1->I6 and I2->I7, the c2p map will no longer be injective, which
  // is not what we want.
  const auto p2c_map = invertOneToOneMap(c2p_map);
  for (const auto& kv : PairwiseRootDomainMap(producer_tv, consumer_tv)
                            .mapBroadcast(false)
                            .mapDifferentExtents(true)
                            .mapConsumerToProducer()) {
    auto consumer_root_id = kv.first;
    auto producer_root_id = kv.second;
    if (c2p_map.find(consumer_root_id) == c2p_map.end() &&
        p2c_map.find(producer_root_id) == p2c_map.end()) {
      c2p_map.emplace(consumer_root_id, producer_root_id);
    }
  }

  const auto& producer_indexing_from_idgraph = getTensorIndexFromIdGraph(
      loops, rotated_loops, consumer_tv, producer_tv, true, c2p_map);

  auto producer_indexing = producer_indexing_from_idgraph.index;

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  const auto& alloc_dom = producer_tv->getMaybeAllocationDomain();

  std::vector<Val*> alloc_inds(
      alloc_dom.size(), GpuLower::current()->kernel()->zeroVal());

  for (const auto i : c10::irange(alloc_dom.size())) {
    auto override_it = override_index.find(alloc_dom[i]);
    const bool is_overriden = override_it != override_index.end();

    if (alloc_dom[i]->isReduction() ||
        (alloc_dom[i]->isBroadcast() && !is_overriden)) {
      continue;
    }

    Val* alloc_ind = nullptr;
    if (is_overriden) {
      alloc_ind = override_it->second;
    } else if (
        producer_indexing.indexMap().find(alloc_dom[i]) !=
        producer_indexing.indexMap().end()) {
      alloc_ind = producer_indexing.indexMap().at(alloc_dom[i]);
    }

    NVF_ERROR(
        alloc_ind != nullptr,
        "Couldn't find allocation mapping for ",
        producer_tv->toString(),
        " dim: ",
        i,
        " id: ",
        alloc_dom[i]->toString());

    if (!alloc_dom[i]->isBroadcast() || !is_overriden) {
      // This is an Iteration domain or a non-padded broadcast domain
      alloc_ind = getProducerIndexWithHalo(
          producer_tv, i, alloc_ind, consumer_tv, is_overriden);

      alloc_ind = getProducerIndexWithGather(
          alloc_ind,
          i,
          producer_tv,
          consumer_tv,
          producer_indexing_from_idgraph.concrete_index.indexMap());

      alloc_ind = getProducerIndexWithPartialSplit(
          alloc_ind, alloc_dom[i], producer_tv, consumer_tv);
    }

    alloc_inds.at(i) = alloc_ind;
  }

  return alloc_inds;
}

std::vector<Val*> Index::getGlobalConsumerStridedIndices(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<int, Val*>& override_index) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getGlobalConsumerIndex");

  auto index_from_id_graph =
      getTensorIndexFromIdGraph(loops, rotated_loops, consumer_tv);
  auto consumer_indexing = index_from_id_graph.index;
  auto strides = getStrides(consumer_tv);
  // if we need to override index, we need to generate the index from each
  // allocation axis firstly.
  auto alloc_inds =
      getConsumerAllocationIndices(consumer_tv, loops, index_from_id_graph);

  // Global striding
  auto vectorize_shift =
      loops.empty() ? nullptr : loops.back()->vectorize_shift();
  std::vector<Val*> strided_inds(
      alloc_inds.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : c10::irange(alloc_inds.size())) {
    auto override_it = override_index.find((int)i);
    if (override_it != override_index.end()) {
      alloc_inds[i] = override_it->second;
    }
    if (alloc_inds[i]->isZeroInt()) {
      continue;
    } else {
      auto strided_ind =
          SimplifyingIrBuilder::mulExpr(alloc_inds[i], strides[i]);
      if (i == strides.size() - 1 && vectorize_shift != nullptr) {
        strided_inds[i] =
            SimplifyingIrBuilder::addExpr(strided_ind, vectorize_shift);
      } else {
        strided_inds[i] = strided_ind;
      }
    }
  }

  NVF_ERROR(
      strided_inds.size() == consumer_tv->getMaybeAllocationDomain().size());

  return strided_inds;
}

// Consumer index for either shared or local memory
std::vector<Val*> Index::getNonGlobalConsumerStridedIndices(
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  const auto gpu_lower = GpuLower::current();
  // At now, only ScatterOp set override_index, and the output of ScatterOp
  // is on global memory, so in this method, the override_index must be empty.
  NVF_ERROR(override_index.empty());
  auto consumer_indexing_from_idgraph = getTensorIndexFromIdGraph(
      loops,
      rotated_loops,
      consumer_tv,
      // Producer tv
      nullptr,
      // Index global
      false);

  auto consumer_indexing = consumer_indexing_from_idgraph.index;

  IndexSwizzle index_swizzle(
      consumer_tv,
      consumer_indexing.indexMap(),
      consumer_indexing.extentMap(),
      consumer_indexing.zeroDomains(),
      consumer_indexing.zeroMergedIn());

  index_swizzle.run();

  const auto& index_map = index_swizzle.indexMap();
  const auto& extent_map = consumer_indexing.extentMap();
  const auto& zero_domain_map = consumer_indexing.zeroDomains();

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  const auto& alloc_dom = consumer_tv->getMaybeAllocationDomain();
  std::vector<Val*> strided_inds(
      alloc_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : c10::irange(alloc_dom.size())) {
    if (alloc_dom[i]->isReduction() || alloc_dom[i]->isBroadcast() ||
        alloc_dom[i]->isStride() || alloc_dom[i]->isDeviceDim() ||
        (alloc_dom[i]->isThread() &&
         consumer_tv->getMemoryType() == MemoryType::Local)) {
      continue;
    }

    std::stringstream error_msg_loops;
    if (index_map.find(alloc_dom[i]) == index_map.end()) {
      for (auto loop : loops) {
        error_msg_loops << " " << loop->iter_domain()->toString();
      }
    }

    NVF_ERROR(
        index_map.find(alloc_dom[i]) != index_map.end(),
        "Couldn't find allocation mapping for ",
        consumer_tv->toString(),
        " dim: ",
        i,
        " id: ",
        alloc_dom[i]->toString(),
        ", loops: ",
        error_msg_loops.str());

    auto alloc_ind_i = index_map.at(alloc_dom[i]);
    if (alloc_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (const auto j : c10::irange(i + 1, alloc_dom.size())) {
      if (alloc_dom[j]->isBroadcast() || alloc_dom[j]->isReduction() ||
          alloc_dom[j]->isDeviceDim() || alloc_dom[j]->isStride()) {
        continue;
      }

      NVF_ERROR(
          index_map.find(alloc_dom[j]) != index_map.end(),
          "Couldn't find allocation mapping for ",
          consumer_tv->toString(),
          " dim: ",
          j,
          " id: ",
          alloc_dom[j]->toString());

      auto alloc_ext_j = extent_map.find(alloc_dom[j]) == extent_map.end()
          ? alloc_dom[j]->extent()
          : extent_map.at(alloc_dom[j]);

      alloc_ext_j = getHaloExtentOfRootAxis(alloc_dom[j], alloc_ext_j);

      if (zero_domain_map.count(alloc_dom[j]) == 0) {
        if (stride == nullptr) {
          stride = alloc_ext_j;
        } else {
          stride = SimplifyingIrBuilder::mulExpr(stride, alloc_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds[i] = SimplifyingIrBuilder::mulExpr(alloc_ind_i, stride);
    } else {
      strided_inds[i] = alloc_ind_i;
    }
  }

  // This check was originally done in getConsumerStridedIndices, but
  // the number of strided index values depends on the loop where the
  // consumer tensor is located. If it's double buffered and not in
  // the prologue loop, strided_inds ends up having one more
  // index, so it's just much simpler to check here before adding the
  // additional index for double buffering.
  NVF_ERROR(
      strided_inds.size() == consumer_tv->getMaybeAllocationDomain().size());

  if (consumer_tv->isDoubleBuffered() || consumer_tv->isCircularBuffered()) {
    auto db_loop =
        gpu_lower->doubleBufferInfo().getDoubleBufferLoop(consumer_tv, loops);
    auto stage_depth = (int64_t)gpu_lower->doubleBufferInfo().getStageDepthFor(
        db_loop->iter_domain());
    bool is_circular_buffer_loop = stage_depth > 2;
    bool is_prolog =
        db_loop->doubleBufferLoopStage() == DoubleBufferLoopStage::Prolog;

    Val* db_switch_index = nullptr;

    // In double buffered we don't materialize the prolog loop as there will
    //  be only one iteration. In circular buffer case we materialize the
    //  prolog loop as well covering the first N-1 iterations, N being the
    //  stage depth.
    if (!is_prolog || is_circular_buffer_loop) {
      if (is_prolog && is_circular_buffer_loop) {
        // The buffer switching logic is the same as original index
        //  in the case of circular buffer prolog.
        db_switch_index = db_loop->indexOrStartIfTrivial();
        if (rotated_loops.count(db_loop)) {
          db_switch_index =
              SimplifyingIrBuilder::addExpr(db_switch_index, db_loop->step());
        }
      } else {
        auto loop_index = db_loop->indexOrStartIfTrivial();
        if (rotated_loops.count(db_loop)) {
          loop_index =
              SimplifyingIrBuilder::addExpr(loop_index, db_loop->step());
        }
        // Switching index generated for main loop or epilog component.
        db_switch_index = SimplifyingIrBuilder::modExpr(
            SimplifyingIrBuilder::addExpr(
                loop_index,
                SimplifyingIrBuilder::create<Val>(
                    stage_depth - 1, DataType::Index)),
            SimplifyingIrBuilder::create<Val>(stage_depth, DataType::Index));
      }

      // Use the generated switching buffer index to access the buffer space.
      auto original_alloc_size =
          gpu_lower->doubleBufferInfo().getOriginalAllocSize(consumer_tv);
      auto db_strided_index =
          SimplifyingIrBuilder::mulExpr(db_switch_index, original_alloc_size);
      strided_inds.push_back(db_strided_index);
    }
  }
  return strided_inds;
}

Val* Index::getProducerStridedIndices(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index,
    bool generate_pointer) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getProducerStridedIndices");
  if (producer->domain()->noReductions().empty()) {
    if (generate_pointer) {
      return IrBuilder::baseAddressExpr(producer);
    } else {
      return GpuLower::current()->kernel()->zeroVal();
    }
  }

  if (producer->getMemoryType() == MemoryType::Global) {
    auto index = sumVals(getGlobalProducerStridedIndices(
        producer, consumer, loops, rotated_loops, override_index));
    if (generate_pointer) {
      return SimplifyingIrBuilder::addExpr(
          IrBuilder::baseAddressExpr(producer), index);
    } else {
      return index;
    }
  } else {
    auto index = sumVals(getNonGlobalProducerStridedIndices(
        producer, consumer, loops, rotated_loops, override_index));
    if (generate_pointer) {
      auto index_bytes = IrBuilder::mulExpr(
          index,
          IrBuilder::create<Val>(
              dataTypeSize(*producer->getDataType()), *index->getDataType()));
      return IrBuilder::addExpr(
          IrBuilder::baseAddressExpr(producer), index_bytes);
    } else {
      return index;
    }
  }
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index,
    bool generate_pointer,
    DataType as_type) {
  auto index = getProducerStridedIndices(
      producer,
      consumer,
      loops,
      rotated_loops,
      override_index,
      generate_pointer);
  index = GpuLower::current()->commonScalarMap().hoistScalar(index, loops);
  if (ir_utils::isLdMatrixOp(consumer->definition()) &&
      at::cuda::getCurrentDeviceProperties()->major < 8) {
    auto items_per_thread = ir_utils::getVectorizeSize(consumer);
    if (items_per_thread != 8) {
      // For Turing, unused indices for ldmatrix needs to be aligned, although
      // they are not used.
      auto orig_index = index;
      index = IrBuilder::create<Val>(index->dtype());
      UnaryOpType op = UnaryOpType::Print;
      if (items_per_thread == 2) {
        op = UnaryOpType::AdjustPartialLdMatrixAddrInTuring8;
      } else if (items_per_thread == 4) {
        op = UnaryOpType::AdjustPartialLdMatrixAddrInTuring16;
      } else {
        NVF_ERROR(
            false,
            "Unexpected output vectorizaiton for ldmatrix, expect 2, 4, or 8, get ",
            items_per_thread);
      }
      IrBuilder::create<UnaryOp>(op, index, orig_index);
    }
  }
  return IrBuilder::create<kir::TensorIndex>(producer, index, as_type);
}

Val* Index::getConsumerStridedIndices(
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<int, Val*>& override_index,
    bool generate_pointer) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getConsumerStridedIndices");
  if (consumer->domain()->noReductions().empty()) {
    if (generate_pointer) {
      return IrBuilder::baseAddressExpr(consumer);
    } else {
      return GpuLower::current()->kernel()->zeroVal();
    }
  }

  if (consumer->getMemoryType() == MemoryType::Global) {
    auto index = sumVals(getGlobalConsumerStridedIndices(
        consumer, loops, rotated_loops, override_index));
    if (generate_pointer) {
      return SimplifyingIrBuilder::addExpr(
          IrBuilder::baseAddressExpr(consumer), index);
    } else {
      return index;
    }
  } else {
    auto index = sumVals(
        getNonGlobalConsumerStridedIndices(consumer, loops, rotated_loops));
    if (generate_pointer) {
      auto index_bytes = IrBuilder::mulExpr(
          index,
          IrBuilder::create<Val>(
              dataTypeSize(*consumer->getDataType()), *index->getDataType()));
      return IrBuilder::addExpr(
          IrBuilder::baseAddressExpr(consumer), index_bytes);
    } else {
      return index;
    }
  }
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    const std::unordered_map<int, Val*>& override_index,
    bool generate_pointer,
    DataType as_type) {
  auto index = getConsumerStridedIndices(
      consumer, loops, rotated_loops, override_index, generate_pointer);
  index = GpuLower::current()->commonScalarMap().hoistScalar(index, loops);
  return SimplifyingIrBuilder::create<kir::TensorIndex>(
      consumer, index, as_type);
}

namespace {

struct PredicateDomainInfo {
 public:
  // Iteration domain to predicate
  IterDomain* id = nullptr;
  // The set of iteration domains that make up the id. If this is for
  // a non-divisible split, the set only contains the id itself. This
  // set is used to remove redundant predicates when gathering
  // unswitch predicates.
  std::unordered_set<IterDomain*> covered_ids;
  // True if this predicate is for an intermediate domain. Examples
  // include domains with non-divisible split and resized domains.
  bool is_intermediate_domain = false;
};

// Find iteration domains in the history of a consumer to predicate comprised
// only of merge operations. Only return iteration domains that are subsequently
// fed into a split, or are in the provided domain. In other words, we don't
// want to return every IterDomain that's contiguous, just the one closest to
// the leaves. Predicates are not associated with physical memory so we can
// treat all of them as contiguous merges.
//
// TODO: This seems to have a large overlap with ContigIDs. Consider
// refactoring.
std::vector<PredicateDomainInfo> getPredicateContigIds(
    TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& consumer_index_map) {
  const auto gpu_lower = GpuLower::current();

  // When there's a resize expr between the root and the rfactor
  // domains, predicate the rfactor domain. Otherwise, predicate the
  // root domain. The actual size of an IterDomain after resize
  // changes, and the output IterDomain needs to be used to generate
  // its predicate.
  const auto& consumer_root_domain = ir_utils::hasResizedRfactor(consumer_tv)
      ? consumer_tv->getMaybeRFactorDomain()
      : consumer_tv->getRootDomain();

  if (consumer_root_domain.empty()) {
    return std::vector<PredicateDomainInfo>();
  }

  std::unordered_map<IterDomain*, Val*> concrete_index_map;
  for (auto entry : consumer_index_map) {
    auto c_id = gpu_lower->caMap()->getConcreteMappedID(
        entry.first, IdMappingMode::EXACT);
    concrete_index_map[c_id] = entry.second;
  }

  std::unordered_set<IterDomain*> final_ids;
  for (auto root_i : c10::irange(consumer_root_domain.size())) {
    auto root_id = consumer_root_domain[root_i];
    if (root_id->maybePartial()) {
      final_ids.insert(root_id);
      continue;
    }
    // Shifted or gathered axes need to be predicated at the root domain
    auto shift_expr = dynamic_cast<ShiftOp*>(consumer_tv->definition());
    auto gather_expr = dynamic_cast<GatherOp*>(consumer_tv->definition());
    if ((shift_expr && shift_expr->offset(root_i) != 0) ||
        (gather_expr && root_i < gather_expr->windowShape().size() &&
         gather_expr->windowShape().at(root_i) != 1)) {
      final_ids.insert(root_id);
    }
  }

  ContigIDs contig_finder(
      consumer_tv->getLeafDomain(),
      consumer_root_domain,
      TensorDomain::getContiguityFilledWith(consumer_root_domain, true),
      final_ids,
      concrete_index_map,
      GpuLower::current()->divisibleSplitSet(),
      GpuLower::current()->caMap(),
      GpuLower::current()->haloInfo(),
      GpuLower::current()->concretizedBroadcastDomains(),
      {},
      false,
      true);

  std::vector<PredicateDomainInfo> contig_id_infos;
  std::unordered_set<IterDomain*> covered_roots;

  // Create entries and return them
  for (auto root_id : consumer_root_domain) {
    if (covered_roots.count(root_id) > 0) {
      continue;
    }

    if (root_id->isBroadcast()) {
      continue;
    }

    auto contig_id_it = contig_finder.allocToIndexedID().find(root_id);

    NVF_ERROR(
        contig_id_it != contig_finder.allocToIndexedID().end(),
        "Error in predicate contiguity analysis, missing index for root ",
        root_id->toString());

    auto contig_id = contig_id_it->second;

    // Pick inputs from the starting domains, i.e.,
    // reference_predicated_root_domain.
    auto contig_alloc_ids = contig_finder.indexedAllocIDs(contig_id);
    covered_roots.insert(contig_alloc_ids.begin(), contig_alloc_ids.end());
    PredicateDomainInfo contig_id_info;
    contig_id_info.id = contig_id;
    contig_id_info.covered_ids = std::unordered_set<IterDomain*>(
        contig_alloc_ids.begin(), contig_alloc_ids.end());
    contig_id_infos.push_back(contig_id_info);
  }
  return contig_id_infos;
}

std::vector<PredicateDomainInfo> getNonDivisibleConsumerDomainsToPredicate(
    TensorView* consumer_tv) {
  const auto& non_divisible_split_info =
      GpuLower::current()->nonDivisibleSplitInfo();

  std::vector<PredicateDomainInfo> pred_info_vec;

  auto it = non_divisible_split_info.splitsToPredicate().find(consumer_tv);
  if (it == non_divisible_split_info.splitsToPredicate().end()) {
    return {};
  }

  const auto& splits_to_predicate = it->second;

  for (auto split : splits_to_predicate) {
    PredicateDomainInfo info{split->in(), {split->in()}, true};
    pred_info_vec.emplace_back(info);
  }

  return pred_info_vec;
}

bool needsPadding(TensorView* tv) {
  auto shift_expr = dynamic_cast<ShiftOp*>(tv->definition());
  auto gather_expr = dynamic_cast<GatherOp*>(tv->definition());

  return (shift_expr != nullptr && shift_expr->hasPadding()) ||
      (gather_expr != nullptr && gather_expr->hasPadding());
}

// Get an additional offset of a stop index when building a predicate
// for unswitch. Initial stop indices generated at
// getPredicateIndexingFromIdGraph do not take halo into account, and the
// adjustment for halo is done as an additional offset to the final index value
// so that unswitch predicates can be compared with each other by just looking
// at the additional offsets.
//
// consumer_root_id: the domain for which a stop predicate is being built.
int64_t getUnswitchStopOffset(
    IterDomain* consumer_root_id,
    TensorView* consumer_tv) {
  const auto gpu_lower = GpuLower::current();

  AxisHaloInfo halo_info =
      gpu_lower->haloInfo()->getRootAxisInfo(consumer_root_id);

  // If the consumer root domain to predicate does not have halo, no
  // adjustment is required.
  if (!halo_info.hasHalo()) {
    return 0;
  }

  // Find if this contig_id is used in the unswitched domains
  auto unswitch_it = std::find_if(
      consumer_tv->getLeafDomain().begin(),
      consumer_tv->getLeafDomain().end(),
      [](IterDomain* id) {
        return id->getParallelType() == ParallelType::Unswitch ||
            id->getParallelType() == ParallelType::Unroll ||
            id->getParallelType() == ParallelType::Vectorize;
      });

  // If any of the unswitched leaf domains inherits the halo from the
  // root domain, the halo width needs to be added to the stop offset
  if (std::any_of(
          unswitch_it,
          consumer_tv->getLeafDomain().end(),
          [&gpu_lower, &consumer_root_id](auto leaf_id) {
            return gpu_lower->haloInfo()->isHaloInherited(
                consumer_root_id, leaf_id);
          })) {
    return halo_info.width();
  } else {
    return 0;
  }
}

std::pair<Val*, Val*> getStartAndStopOffsetsForShift(
    TensorView* consumer_tv,
    IterDomain* consumer_id,
    bool padding_predicate) {
  NVF_ERROR(consumer_id != nullptr);

  auto shift_expr = dynamic_cast<ShiftOp*>(consumer_tv->definition());

  // Adjustment is not necessary if not shift.
  // Even so, padding predicate does not need any adjustment.
  if (shift_expr == nullptr || padding_predicate) {
    return {
        GpuLower::current()->kernel()->zeroVal(),
        GpuLower::current()->kernel()->zeroVal()};
  }

  const auto root_axis_pos = consumer_tv->domain()->rootPosOf(consumer_id);

  // The first or last N elements, where N is the padding width,
  // correspond to the padding predicate.

  const auto shift_offset = shift_expr->offset(root_axis_pos);
  const auto pad_width = shift_expr->padWidth().at(root_axis_pos);

  int64_t start_offset = 0;
  int64_t stop_offset = 0;

  if (shift_offset > 0) {
    start_offset = -pad_width;
  } else if (shift_offset < 0) {
    stop_offset = pad_width;
  }

  return {
      SimplifyingIrBuilder::create<Val>(start_offset, DataType::Index),
      SimplifyingIrBuilder::create<Val>(stop_offset, DataType::Index)};
}

std::pair<Val*, Val*> getStartAndStopOffsetsForGather(
    TensorView* consumer_tv,
    IterDomain* consumer_id,
    const std::unordered_map<IterDomain*, Val*>& ref_start_index_map,
    const std::unordered_map<IterDomain*, Val*>& ref_stop_index_map,
    bool padding_predicate) {
  NVF_ERROR(consumer_id != nullptr);

  // Adjustment is not necessary if not gather. Even so, padding
  // predicate does not need any adjustment.
  if (!consumer_tv->definition()->isA<GatherOp>() || padding_predicate) {
    return {
        GpuLower::current()->kernel()->zeroVal(),
        GpuLower::current()->kernel()->zeroVal()};
  }

  const auto root_axis_pos = consumer_tv->domain()->rootPosOf(consumer_id);

  auto producer_start_offset = getProducerOffsetWithGather(
      root_axis_pos, consumer_tv, ref_start_index_map);

  auto producer_stop_offset = getProducerOffsetWithGather(
      root_axis_pos, consumer_tv, ref_stop_index_map);

  auto consumer_start_offset = GpuLower::current()->kernel()->zeroVal();
  auto consumer_stop_offset = GpuLower::current()->kernel()->zeroVal();

  if (producer_start_offset->isZeroInt() && producer_stop_offset->isZeroInt()) {
    return {consumer_start_offset, consumer_stop_offset};
  }

  Val* start_offset = nullptr;
  Val* stop_offset = nullptr;

  // In the normal case, take the minimum of the start and the
  // maximum of the stop offsets. If there's no padding, the producer
  // offset must be always larger than the consumer
  // offset. So, the consumer and produce offsets can be always used
  // for the start and stop offsets, respectively.
  const auto pad_left =
      consumer_tv->definition()->as<GatherOp>()->padWidth()[root_axis_pos][0];
  const auto pad_right =
      consumer_tv->definition()->as<GatherOp>()->padWidth()[root_axis_pos][1];
  const auto window_size =
      consumer_tv->definition()->as<GatherOp>()->windowShape()[root_axis_pos];

  // consumer index: index
  // producer index: index + window_index - pad_left
  //
  // consumer extent: ext
  // producer extent: ext + window_size - 1 - pad_left - pad_right
  //
  // consumer stop pred: index < ext
  // producer stop pred: index + window_index - pad_left < ext + window_size - 1
  // - pad_left - pad_right
  //                  -> index + window_index - pad_left - (window_size - 1 -
  //                  pad_left - pad_right) < ext
  //                  -> index + window_index - (window_size - 1 - pad_right) <
  //                  ext
  //
  // consumer start pred: index >= 0
  // producer start pred: index + window_index - pad_left >= 0

  const auto producer_ext_adj = window_size - 1 - pad_left - pad_right;
  producer_stop_offset = SimplifyingIrBuilder::subExpr(
      producer_stop_offset,
      SimplifyingIrBuilder::create<Val>(
          (int64_t)producer_ext_adj, DataType::Index));

  // As commented above, when pad_left is zero, the consumer predicate
  // is always more restrictive than the producer predicate.
  if (pad_left == 0) {
    start_offset = consumer_start_offset;
  } else {
    start_offset = SimplifyingIrBuilder::minExpr(
        consumer_start_offset, producer_start_offset);
  }

  // As commented above, when pad_right is zero, the consumer
  // predicate is always more restrictive than the producer
  // predicate.
  if (pad_right == 0) {
    stop_offset = consumer_stop_offset;
  } else {
    stop_offset = SimplifyingIrBuilder::maxExpr(
        consumer_stop_offset, producer_stop_offset);
  }

  NVF_ERROR(start_offset != nullptr);
  NVF_ERROR(stop_offset != nullptr);

  return {start_offset, stop_offset};
}

// Get the start and stop limit offsets that define the valid range to
// compute. In the simplest case, they are just 0 and
// IterDomain::extent. However, IterDomain may have non-zero start and
// stop that's different from extent. Also, when IterDomain has halo,
// the actual offsets of the logical start and stop positions are
// shifted.
std::pair<Val*, Val*> getStartAndStopLimitOffsets(
    IterDomain* consumer_id,
    bool padding_predicate,
    bool intemediate_domain_pred) {
  const auto gpu_lower = GpuLower::current();

  NVF_ERROR(consumer_id != nullptr);

  Val* start_limit = consumer_id->start();
  Val* stop_limit = SimplifyingIrBuilder::negExpr(consumer_id->stopOffset());

  if (!intemediate_domain_pred) {
    AxisHaloInfo halo_info =
        gpu_lower->haloInfo()->getRootAxisInfo(consumer_id);

    // Below, "left" and "right" halo mean halo at offset zero and
    // axis extent, respectively.
    //
    // The consumer axis looks like this:
    //
    // [0, left halo)[start_limit, stop_limit)[0, right halo)
    //
    if (!padding_predicate) {
      start_limit = SimplifyingIrBuilder::addExpr(
          start_limit, (int64_t)halo_info.width(0));
      stop_limit = SimplifyingIrBuilder::addExpr(
          stop_limit, (int64_t)halo_info.width(0));
    } else {
      // In case of the padding predicate, the whole range, including both left
      // and right halo regions, is computed.
      stop_limit =
          SimplifyingIrBuilder::addExpr(stop_limit, (int64_t)halo_info.width());
    }
  } else {
    // For non-divisible predicates, the index must be predicated such
    // that it is less than the extent of the predicated ID +
    // halo. Note that getRootAxisInfo doesn't work since consumer_id
    // isn't a root domain.
    if (gpu_lower->haloInfo()->hasHaloWidth(consumer_id)) {
      auto halo = gpu_lower->haloInfo()->getHaloWidth(consumer_id);
      stop_limit = SimplifyingIrBuilder::addExpr(stop_limit, (int64_t)halo);
    }
  }

  return {start_limit, stop_limit};
}

// Get the offsets for the start and stop predicates. The offsets
// are to be added to the index.
std::pair<Val*, Val*> getStartAndStopOffsets(
    IterDomain* consumer_id,
    TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& consumer_start_index_map,
    const std::unordered_map<IterDomain*, Val*>& consumer_stop_index_map,
    bool padding_predicate,
    bool unswitch,
    bool intermediate_domain_pred) {
  // By default, the offsets for the start and stop predicates are
  // just zero. All halo-related adjustments are done at root domains,
  // so consumer_id is not a root domain, no adjustment is required.
  if (consumer_id->definition() != nullptr && !intermediate_domain_pred) {
    return {
        GpuLower::current()->kernel()->zeroVal(),
        GpuLower::current()->kernel()->zeroVal()};
  }

  auto consumer_def = consumer_tv->definition();

  Val* start_offset = GpuLower::current()->kernel()->zeroVal();
  Val* stop_offset = GpuLower::current()->kernel()->zeroVal();

  // These adjustments are not required when predicating non-divisible splits
  if (!intermediate_domain_pred) {
    if (consumer_def->isA<ShiftOp>()) {
      std::tie(start_offset, stop_offset) = getStartAndStopOffsetsForShift(
          consumer_tv, consumer_id, padding_predicate);
    } else if (consumer_def->isA<GatherOp>()) {
      std::tie(start_offset, stop_offset) = getStartAndStopOffsetsForGather(
          consumer_tv,
          consumer_id,
          consumer_start_index_map,
          consumer_stop_index_map,
          padding_predicate);
    }

    // Adjustment for partial split
    auto partial_split_offset =
        getGlobalConsumerOffsetWithPartialSplit(consumer_id);
    start_offset =
        SimplifyingIrBuilder::addExpr(start_offset, partial_split_offset);
    stop_offset =
        SimplifyingIrBuilder::addExpr(stop_offset, partial_split_offset);

    // If generating a predicate for unswitch, adjust the stop offset to
    // accommodate the addition of halo to the loop stop. See the
    // comment in getPredicateIndexingFromIdGraph as well.
    if (unswitch) {
      NVF_ERROR(
          !padding_predicate, "Unswitch should not use the padding predicate");
      auto stop_unswitch_offset =
          getUnswitchStopOffset(consumer_id, consumer_tv);
      stop_offset =
          SimplifyingIrBuilder::addExpr(stop_offset, stop_unswitch_offset);
    }
  }

  // Get the boundaries of two ends
  auto limits = getStartAndStopLimitOffsets(
      consumer_id, padding_predicate, intermediate_domain_pred);

  // At this point, we have everything to create both start and stop
  // predicates as:
  //
  //  index + start_offset >= start_limit
  //  index + stop_offset  < extent + stop_limit
  //
  // In order to enable consolidating unswitch predicates, organize
  // the predicates as:
  //
  //  index + (start_offset - start_limit) >= 0
  //  index + (stop_offset - stop_limit)  < extent

  start_offset = SimplifyingIrBuilder::subExpr(start_offset, limits.first);
  stop_offset = SimplifyingIrBuilder::subExpr(stop_offset, limits.second);

  return {start_offset, stop_offset};
}

bool canOmitStopPredicate(
    Val* stop_index,
    Val* stop_offset,
    IterDomain* contig_id) {
  bool index_simple = stop_index->definition() == nullptr;
  // The definition may be just adding the magic zero, which can be
  // effectively considered "simple"
  if (!index_simple && isProtectedWithMagicZero(stop_index)) {
    // Make sure the lhs of stop_index is simple.
    auto lhs = stop_index->definition()->as<BinaryOp>()->lhs();
    if (lhs->definition() == nullptr) {
      index_simple = true;
    }
  }

  if (!index_simple) {
    return false;
  }

  const auto gpu_lower = GpuLower::current();

  auto stop_offset_val = stop_offset->value();

  // If they are not compile-time constant, can't prove the
  // condition.
  if (!stop_offset_val.hasValue()) {
    return false;
  }

  auto stop_index_val = stop_index->value();

  // If stop_index is a constant, then the expr can be in a trivial loop.
  // Trivial loop is not materialized, so it is not protected under the `for`
  // statement. If this is the case, we omit stop predicate only if we can
  // prove: stop_index + stop_offset < extent
  if (stop_index_val.hasValue()) {
    // Stop predicate: stop_index + stop_offset < extent
    auto lhs = stop_index_val + stop_offset_val;
    auto in_extent = IrBuilder::ltExpr(
        IrBuilder::create<Val>(lhs, *stop_index->getDataType()),
        contig_id->getMaybeExpandedExtent());
    auto expr_val = simplifyExpr(in_extent)->value();
    if (expr_val.hasValue() && expr_val.is<bool>() && expr_val.as<bool>()) {
      return true;
    } else {
      return false;
    }
  }

  // Stop predicate: stop_index + stop_offset < extent, where
  // stop_index ranges from 0 to (extent + halo), so this can be
  // omitted if extent + halo + stop_offset < extent, i.e., halo +
  // stop_offset < 0.

  // Note that when a root domain is halo extended, it is the domain
  // to be predicated, not its merged contig id even if it exists. So,
  // if contig_id does not have root axis info, contig_id is
  // guaranteed to have no halo.
  auto halo_ext = gpu_lower->haloInfo()->hasRootAxisInfo(contig_id)
      ? gpu_lower->haloInfo()->getRootAxisInfo(contig_id).width()
      : 0;

  if (halo_ext + stop_offset_val >= 0) {
    return false;
  }

  // When the domain is parallelized, the parallel dimension must be
  // exact. Otherwise, there would be extra threads/blocks that need
  // to be predicated out.
  if (isParallelTypeThread(contig_id->getParallelType())) {
    if (!lower_utils::isExtentEqualToMaxParallelTypeExtent(contig_id)) {
      return false;
    }
    // If the domain has halo, the loop is expanded by the halo
    // extent, so we can't prove the loop extent is the same as the
    // parallel dimension.
    if (halo_ext != 0) {
      return false;
    }
  }

  return true;
}

// Updates a loop index map with a loop index protected by magic zero
std::unordered_map<IterDomain*, Val*> updateInitialLoopIndexMap(
    const std::unordered_map<IterDomain*, Val*>& initial_loop_index_map,
    const IndexMagicZeroInfo& magic_zero_info) {
  if (magic_zero_info.original_loop_index != nullptr) {
    NVF_ERROR(magic_zero_info.protected_loop_index != nullptr);
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        magic_zero_info.loop_id, IdMappingMode::EXACT);
    auto updated_map = initial_loop_index_map;
    updated_map[concrete_loop_id] = magic_zero_info.protected_loop_index;
    return updated_map;
  } else {
    return initial_loop_index_map;
  }
}

} // namespace

// Returns predicates and the concrete (by loop map) root domains they cover
std::vector<RootPredicateInfo> Index::getReferenceRootPredicates(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    kir::ForLoop* unswitch_or_vec_loop,
    bool shift_padding) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getReferenceRootPredicates");

  const auto gpu_lower = GpuLower::current();

  const bool is_unswitch = unswitch_or_vec_loop != nullptr;

  // Nothing needs to be done when padding is not required.
  if (shift_padding && !needsPadding(consumer_tv)) {
    return {RootPredicateInfo::getFalseInfo()};
  }

  auto db_axis = gpu_lower->doubleBufferInfo().getDoubleBufferAxis(consumer_tv);

  // Generate start and stop indexing from idgraph.
  //
  // Both start and stop positions may need to be predicated. Indexing
  // differs when generating predicates for unswitch.
  // NOTE: If we could find-and-replace KIR nodes, we could just
  // generate one index map, clone it and replace the loop-to-index
  // mappings of unswitched loops for the start predicate.

  auto stop_indexing_from_idgraph = getPredicateIndexingFromIdGraph(
      loops, rotated_loops, consumer_tv, unswitch_or_vec_loop, db_axis, false);
  const auto consumer_stop_indexing = stop_indexing_from_idgraph.index;
  const auto& consumer_stop_index_map = consumer_stop_indexing.indexMap();

  // If not unswitch, share the same indexing map as the stop index
  // map
  const auto start_indexing_from_idgraph = is_unswitch
      ? getPredicateIndexingFromIdGraph(
            loops,
            rotated_loops,
            consumer_tv,
            unswitch_or_vec_loop,
            db_axis,
            true)
      : stop_indexing_from_idgraph;
  const auto consumer_start_indexing = start_indexing_from_idgraph.index;
  const auto& consumer_start_index_map = consumer_start_indexing.indexMap();

  // Get the contiguous ids we need to generate predicates for
  auto contig_id_infos =
      getPredicateContigIds(consumer_tv, consumer_stop_index_map);

  auto non_divisible_splits =
      getNonDivisibleConsumerDomainsToPredicate(consumer_tv);
  contig_id_infos.insert(
      contig_id_infos.end(),
      non_divisible_splits.begin(),
      non_divisible_splits.end());

  std::vector<RootPredicateInfo> pred_info_vec;

  for (const auto& contig_id_entry : contig_id_infos) {
    auto contig_id = contig_id_entry.id;
    // No predicates needed for braodcasted indices.
    if (contig_id->isBroadcast()) {
      continue;
    }

    auto root_ids = contig_id_entry.covered_ids;

    const auto consumer_stop_indexing_it =
        consumer_stop_index_map.find(contig_id);

    // First condition below happens with Misaligned predicates, where
    // inner-most vectorized loops are not included in the loops
    // parameter. Predicates involving vectorized loops are separately
    // generated in lower_misaligned_vectorization.
    //
    // Can not omit stop index even if it is zero. This is important for empty
    // tensor support, because in empty tensor the extent of an ID can be zero
    if (consumer_stop_indexing_it == consumer_stop_index_map.end()) {
      continue;
    }

    RootPredicateInfo info;

    // Compute offsets for start and stop predicate. For non-shift,
    // non-gather ops, there's only stop predicate as indices never be
    // negative. However, for shift and gather, the index may need to
    // be predicated so that it is >= zero.
    //
    // Furthermore, in case of gather, both producer and consumer
    // positions may need to be predicated, so there can be multiple
    // offset values.
    //
    // The final predicates will look like:
    // (index + start_offset) >= 0 && (index + stop_offset) < extent.

    std::tie(info.start_offset_, info.stop_offset_) = getStartAndStopOffsets(
        contig_id,
        consumer_tv,
        consumer_start_index_map,
        consumer_stop_index_map,
        shift_padding,
        unswitch_or_vec_loop != nullptr,
        contig_id_entry.is_intermediate_domain);

    auto stop_index = consumer_stop_indexing_it->second;
    auto start_index = consumer_start_index_map.at(contig_id);

    IndexMagicZeroInfo start_magic_zero_info;
    IndexMagicZeroInfo stop_magic_zero_info;

    // When the start and stop indices are not the same, apply the
    // magic-zero protection separately for both of them.
    if (stop_index != start_index) {
      start_magic_zero_info = protectPredicateIndexWithMagicZero(
          start_index, start_indexing_from_idgraph, loops);
      stop_magic_zero_info = protectPredicateIndexWithMagicZero(
          stop_index, stop_indexing_from_idgraph, loops);
    } else {
      stop_magic_zero_info = protectPredicateIndexWithMagicZero(
          stop_index, stop_indexing_from_idgraph, loops);
      start_magic_zero_info = stop_magic_zero_info;
    }

    start_index = start_magic_zero_info.index;
    stop_index = stop_magic_zero_info.index;

    // Build predicates for start positions as:
    //   start_index + start_offset >= 0
    auto offsetted_start_index =
        SimplifyingIrBuilder::addExpr(start_index, info.start_offset_);
    auto start_pred = SimplifyingIrBuilder::geExpr(
        offsetted_start_index, GpuLower::current()->kernel()->zeroVal());
    info.start_predicate_ = start_pred;

    // Build predicates for stop positions as:
    //   stop_index + stop_offset < IterDomain::extent
    auto stop_offset = info.stop_offset_;
    if (canOmitStopPredicate(stop_index, stop_offset, contig_id)) {
      info.stop_predicate_ = GpuLower::current()->kernel()->trueVal();
    } else {
      auto offsetted_stop_index =
          SimplifyingIrBuilder::addExpr(stop_index, stop_offset);
      auto stop_pred = SimplifyingIrBuilder::ltExpr(
          offsetted_stop_index, contig_id->extent());
      info.stop_predicate_ = stop_pred;
    }

    for (auto consumer_id : contig_id_entry.covered_ids) {
      info.root_ids_.insert(consumer_id);
    }
    pred_info_vec.emplace_back(info);
  }

  return pred_info_vec;
}

RootPredicateInfo RootPredicateInfo::getFalseInfo() {
  RootPredicateInfo info;
  info.start_predicate_ = GpuLower::current()->kernel()->falseVal();
  info.stop_predicate_ = GpuLower::current()->kernel()->falseVal();

  return info;
}

Val* Index::iota(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    Val* start,
    Val* step,
    DataType dtype) {
  auto linear_index =
      Index::getLinearLogicalIndex(consumer_tv, loops, rotated_loops);
  auto result = add(start, mul(step, linear_index));
  return GpuLower::current()->commonScalarMap().hoistScalar(result, loops);
}

Val* Index::eye(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops,
    DataType dtype) {
  auto indices =
      Index::getConsumerPerDimLogicalIndex(consumer_tv, loops, rotated_loops);
  NVF_ERROR(indices.size() == 2);
  auto result = maybeCastOp(dtype, eq(indices[0], indices[1]));
  return GpuLower::current()->commonScalarMap().hoistScalar(result, loops);
}

// Note [Schedule and indexing of TMA]
//
// TMA is a hardware feature that allows the GPU to load a tile from a tensor of
// up to 5D. Note that the dimensionality of the tensor in the eyes of TMA is
// not necessarily the same as the dimensionality of the tensor in the eyes of
// the user. For example, it is possible that there is a 1D tensor of shape
// 1000000 and the user wants to view it as (1000, 1000) and load a tile of (8,
// 8) from it. Another example is, the user has a tensor of shape (1000, 1000),
// and as long as the tensor is contiguous, the user can view it as a flattened
// shape 1000000 tensor, and load a tile of 8 from it. The dimensionality of the
// tensor in the eyes of TMA is determined by the schedule of the tensor.
// However, the dimensionality of the tensor in the eyes of TMA can not be
// smaller than the physical dimensionality of the tensor. For example, if the
// user has a tensor of shape (128, 128, 128, 128, 128, 128), and the tensor is
// fully discontiguous, then it is not possible to view it as a <= 5D tensor,
// thus TMA can not be used to load/store a tile from it.
//
// The gmem tensor of TMA store, or the gmem tensor of TMA load when replayed as
// consumer, must be scheduled like this:
//
// Diagram 1:
//                         ---------------------
//                         | Allocation Domain |
//                         ---------------------
//                           | | | | | | | | |
//                         IterDomain Expressions
//                           | | | | | | | | |
//            -----------------------------------------------
// TMA domain |  I0           I1           I2           I3  |
//            -----------------------------------------------
//               |            |            |            |
//              ...         split        split          |
//                          /   \        /   |          |
//                         I4   I5      I6   I7         |
//                         |    |       |    |          |
//                        ...   |      ... split        |
//                              |          /   |        |
//                              |         I8   I9       |
//                              |         |    |        /
//                              |         |   ...      /
//                              |         |           /
//                              IterDomain Expressions
//                               |    |    |    |    |
//                              Bulk Bulk Bulk Bulk Bulk
//
// To schedule TMA, the overall process is:
//
// First, we need to specify how TMA should view the tensor. In the above
// diagram, TMA view the tensor as a 4D tensor [I0, I1, I2, I3]. Note that the
// IterDomain expressions between the allocation domain and [I0, I1, I2, I3]
// must be compatible with the allocation domain, for example, we can not merge
// discontiguous IterDomains, and we can not have indivisible splits either.
// We call [I0, I1, I2, I3] the "TMA domain" of the tensor.
//
// Second, we need to tile each dimension of the TMA domain of the tensor. If we
// want a contiguous box dim (see Definition 3: "boxId" below), then we do one
// split to get the box. If we want a strided box dim, then we do two splits,
// the first split is to get the box, and the second split is to get the stride.
// Note that if a box dimension has size 1, we can skip tiling, for this case,
// we call it "implicitly one tiled". In the above diagram, I0 is implicitly one
// tiled, I1 has a contiguous box dim, and I2 has a strided box dim. If we want
// to treat the entire dimension as a tile, we can also skip tiling, for this
// case, we call it "whole tiled". In the above diagram, I3 is whole tiled.
//
// Third, we can schedule the "bulk" IterDomains (see Definition 1: "bulk"
// below) and "non-bulk" IterDomains separately in whatever way we want.
// However, the "bulk" IterDomains and "non-bulk" IterDomains can not be mixed
// together, just like we not merge a IterType::Iteration IterDomain with a
// IterType::Reduction IterDomain.
//
// Because we allow "implicitly one tiled" and "whole tiled" IterDomains, a
// single schedule may be interpreted as multiple different ways. For example,
// if we have a schedule like this:
//   I1 -- split -> I2 (inner, bulk)
//              \-> I3 (outer, non-bulk)
// We may interpret it as: The TMA domain is {I1}; we want to use 1D TMA, and
// we created a tile I2. Or, we may interpret it as: The TMA domain is {I2, I3};
// we view the tensor as 2D to use 2D TMA; we whole tiled I2, and implicitly one
// tiled I3. For this specific example, the first interpretation is better in
// the sense that it does not require the split to be divisible. When the
// interpretation is ambiguous, we will prefer the one with less number of
// "implicitly one tiled" and "whole tiled" IterDomains.
//
// Before further explain the schedule, let's define some terminologies:
//
// Definition 1: An IterDomain is "bulk" if any of the following is true:
//  - It has parallel type "Bulk"
//  - All its children are "bulk"
//
// In the above diagram, I3, I5 and I8 are bulk IterDomains, but I0, I1, I2, I4,
// I6, I7, and I9 are not. Please note that the concept "bulk IterDomain" is
// different from the concept "IterDomain parallelized as IterType::Bulk".
// Because we only parallelize leaf domains, the set "IterDomain parallelized as
// IterType::Bulk" and the set "bulk IterDomains" has the following
// relationship:
// - IterDomains parallelized as IterType::Bulk <= bulk IterDomains
// - bulk IterDomains = IterDomain parallelized as IterType::Bulk & leaf domain
// where "<=" refers to subset, and "&" refers to set intersection.
//
// Definition 2: A bulk IterDomain is "originating" if it has no parent or its
// parents are not bulk.
//
// In the above diagram, I5 and I8 are clearly originating bulk IterDomains. I3
// may or may not be an originating bulk IterDomain, depending on the IterDomain
// expressions between the allocation domain and I3. Because the TMA domain is
// ambiguous like mentioned above, we decide to define the TMA domain as a
// domain that does not have any non-originating bulk IterDomains. So, I3 is
// also an originating bulk IterDomain.
//
// If an originating bulk IterDomains is an inner output of a split, such as I5
// in the above diagram, we consider it as a dense box. That is, for example,
// when we have a tile/box size (8, 8), we want all the 8*8=64 elements to to be
// transfered. For this case, the extents of the originating bulk IterDomains
// are the box sizes (the `boxDim` parameter in `cuTensorMapEncodeTiled`).
//
// If an originating bulk IterDomains is an outer output of a split, such as I8
// in the above diagram, we consider it as a sparse box. That is, for example,
// if we consider a box dim of (8, 8), and we want every other element from it
// to be transferred, that is, items at (0, 0), (0, 2), (0, 4), ..., (2, 0), (2,
// 2), (2, 4), ..., (6, 6). We will transfer the data in the (8, 8) sparse box
// as (4, 4) dense tile of elements. In this scenario, the `elementStrides` in
// `cuTensorMapEncodeTiled` will have a value [2, 2], and the `boxDim` will have
// a value [8, 8].
//
// If an originating bulk IterDomains has no parent, or its definition is not a
// split, such as I3 in the above diagram, then this originating bulk is whole
// tiled. It will also be a dense box.
//
// With this understanding, we can now define more terminologies:
//
// Definition 3: "boxId" is a function that takes an originating bulk IterDomain
// as its input, and returns an IterDomain. Given an originating bulk IterDomain
// I1, the return value, boxId(I1) is:
//   - The parent of I1 if I1 is an outer output of a split
//   - I1 if otherwise
//
// In the above diagram, boxId(I5) is I5, and boxId(I8) is I7, and boxId(I3) is
// I3.
//
// Definition 4: "tmaGlobalId" is a function that takes an originating bulk
// IterDomain as its input, and returns another IterDomain. Given an originating
// bulk IterDomain I1, the return value, tmaGlobalId(I1) is:
//   - The parent of boxId(I1) if boxId(I1) is an inner output of a split
//   - boxId(I1) if otherwise
//
// In the above diagram, tmaGlobalId(I5) is I1, and tmaGlobalId(I8) is I2, and
// tmaGlobalId(I3) is I3.
//
// Definition 5: "innerId" is a function that takes an originating bulk who is
// the outer of the split that defines it, and returns its sibling IterDomain.
//
// In the above diagram, innerId(I8) is I9.
//
// Definition 6: An IterDomain I1 is "TMA-global" if there exists an originating
// bulk IterDomain I2 such that I1 == tmaGlobalId(I2).
//
// In the above diagram, I1, I2 and I3 are TMA-global IterDomains. We do not
// consider I0 as a TMA-global IterDomain.
//
// Note that the number of TMA-global IterDomains should always be equal to the
// number of originating bulk IterDomains.
//
// Also note that TMA-global IterDomains can not have dependencies between each
// other, that is, no one TMA-global IterDomains can be an ancestor or
// descendant of another TMA-global IterDomains. That is, schedules like this is
// not allowed:
//   I1 -> split -> Bulk (inner)
//              \-> I2 (outer) -> split -> Bulk (inner)
//                                     \-> I3 (outer)
// Also note that the set of all TMA-global IterDomains plus implicitly tiled 1
// IterDomains must be "equivalent" to the allocation domain of the tensor
// (ignoring unrelated IterTypes like broadcast and reduction), as defined in
// `ir_utils::validateDomainEquivalence`.
//
// Definition 7: The "TMA domain" of a tensor is the set of all TMA-global
// IterDomains plus implicitly tiled 1 IterDomains, with a specific order. The
// order is determined by the order of the allocation domain of the tensor.
//
//
// Examples: (upper are inner)
//
// Example 1:
//
// -> I0 -> split -> I1 -> split -> I3 (Bulk)
//              \               \-> I4
//               \-> I2 -> split -> I5 (Bulk)
//                              \-> I6
// -> I7
// -> I8 (Bulk)
//
// Originating bulk IterDomains: { I3, I5, I8 }
// Box IterDomains: { I3, I5, I8 }
// Inner IterDomains: {}
// TMA-global IterDomains: { I1, I2, I8 }
// TMA domain: [I8, I7, I2, I1].
// Implicitly one tiled dimensions: { I7 }
// Whole tiled dimensions: { I8 }
//
// Example 2:
//
// -> I0 -> split -> I1 -> split -> I3 (Bulk)
//              \               \-> I4
//               \-> I2 -> split -> I5 -> split -> I7
//                              \-> I6         \-> I8 (Bulk)
//
// Originating bulk IterDomains: { I3, I8 }
// Box IterDomains: { I3, I5 }
// Inner IterDomains: { I7 }
// TMA-global IterDomains: { I1, I2 }
// TMA domain: [I2, I1].
// Implicitly one tiled dimensions: {}
// Whole tiled dimensions: {}
//
// Example 3:
//
// -> I0 -> merge -> I2 -> split -> I3 (Bulk)
// -> I1 ---^                   \-> I4
//
// Originating bulk IterDomains: { I3 }
// Box IterDomains: { I3 }
// Inner IterDomains: {}
// TMA-global IterDomains: { I2 }
// TMA domain: [I2].
// Implicitly one tiled dimensions: {}
// Whole tiled dimensions: {}
namespace {

int64_t getCpAsyncBulkTensorSwizzleSize(TensorView* smem_tv) {
  auto exprs = DependencyCheck::getAllExprsBetween(
      {smem_tv->getMaybeRFactorDomain().begin(),
       smem_tv->getMaybeRFactorDomain().end()},
      {smem_tv->getMaybeAllocationDomain().begin(),
       smem_tv->getMaybeAllocationDomain().end()});
  for (auto expr : exprs) {
    if (auto s = dynamic_cast<Swizzle*>(expr)) {
      return s->inX()->extent()->evaluate().as<int64_t>();
    }
  }
  return 1;
}

} // namespace

// Analyze the schedule of the gmem tensor (for TMA load, it needs to be
// replayed as its consumer) and create IR nodes that compute the N-dimensional
// coordinate and the tensor map descriptor. Also return the byte of transfer.
// We first need to find the TMA domain based on the schedule, which is done by
// finding originating bulk IterDomains first and analyze their definitions.
// After finding all these IterDomains we are interested in, we can compute the
// quantities easily: The N-dimensional coordinate is just the indices of the
// TMA domain in the index map. To compute the tensor map descriptor, we need to
// find the box dims, the element strides, global dims, and global strides. The
// box dims are the extents of the box IterDomains. The element strides are
// either 1 or the extents of the inner IterDomains if any. The global dims are
// the extents of IterDomains in the TMA domain. The global strides are inferred
// based on IterDomain expressions between the allocation domain and TMA domain.
// The byte of transfer is the product of extents of originating bulk
// IterDomains.
std::pair<Val*, Val*> Index::getCpAsyncBulkGmemIndex(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    Val* mbarrier,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_set<kir::ForLoop*>& rotated_loops) {
  FUSER_PERF_SCOPE("Index::getCpAsyncBulkGmemIndex");

  bool is_load = false;
  TensorView *smem_tv = nullptr, *gmem_tv = nullptr;
  if (producer_tv->getMemoryType() == MemoryType::Shared) {
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Global);
    smem_tv = producer_tv;
    gmem_tv = consumer_tv;
    is_load = false;
  } else {
    NVF_ERROR(producer_tv->getMemoryType() == MemoryType::Global);
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Shared);
    smem_tv = consumer_tv;
    gmem_tv = producer_tv;
    is_load = true;
  }

  // For TMA load, we need to replay the gmem tensor as consumer.
  std::unique_ptr<ir_utils::TVDomainGuard> domain_guard;

  std::unique_ptr<IndexCompute> indexing;

  // Convert an id from the consumer tensor to its corresponding id in the
  // gmem tensor. If the consumer tensor is already a gmem tensor, then the
  // function is the identity function. Otherwise, the function is the
  // consumer-to-producer map.
  std::function<IterDomain*(IterDomain*)> consumer_to_gmem;

  if (is_load) {
    // Replay producer to look like consumer so we can index on producer since
    // our loop nests look like consumer
    auto pairwise_map =
        PairwiseRootDomainMap(producer_tv, consumer_tv).mapBroadcast(true);

    TensorDomain* producerAsC = TransformReplay::replayPasC(
                                    producer_tv,
                                    consumer_tv,
                                    -1,
                                    pairwise_map,
                                    TransformReplayOptions().replayResize())
                                    .first;

    // Make the producer_tv look like consumer while performing indexing math
    domain_guard =
        std::make_unique<ir_utils::TVDomainGuard>(producer_tv, producerAsC);

    // Map sent to best effort replay needs to match the exact incantation for
    // compute_at_mode.cpp with MappingMode::Index
    auto c2p_root_map = PairwiseRootDomainMap(producer_tv, consumer_tv)
                            .mapBroadcast(false)
                            .mapConsumerToProducer();

    // This replay has to be consistent with compute at index map.
    BestEffortReplay replay_producer_as_consumer(
        producer_tv->getLeafDomain(),
        consumer_tv->getLeafDomain(),
        c2p_root_map);

    const auto& c2p_map = replay_producer_as_consumer.getReplay();

    consumer_to_gmem = [=](IterDomain* id) -> IterDomain* {
      return c2p_map.at(id);
    };

    const auto& producer_indexing_from_idgraph = getTensorIndexFromIdGraph(
        loops, rotated_loops, consumer_tv, producer_tv, true, c2p_map);

    indexing =
        std::make_unique<IndexCompute>(producer_indexing_from_idgraph.index);
  } else {
    consumer_to_gmem = [](IterDomain* id) -> IterDomain* { return id; };
    auto index_from_id_graph =
        getTensorIndexFromIdGraph(loops, rotated_loops, consumer_tv);
    indexing = std::make_unique<IndexCompute>(index_from_id_graph.index);
  }

  auto allocation_domain = TensorDomain::noBroadcasts(
      TensorDomain::noReductions(gmem_tv->getMaybeAllocationDomain()));
  std::unordered_set<Val*> allocation_domain_set(
      allocation_domain.begin(), allocation_domain.end());

  // Step 1: Get all bulk IterDomains and originating bulk IterDomains

  // Get all bulk IterDomains
  std::unordered_set<IterDomain*> bulk_ids;
  // Bulk IterDomains that we need to check its definition to see if it is an
  // originating bulk IterDomain.
  std::deque<IterDomain*> pending;
  pending.push_back(nullptr); // use nullptr as a checkpoint
  // Start from leaf domain, where all the bulk IterDomains in the leaf domain
  // must be parallelized as ParallelType::Bulk.
  for (auto id : consumer_tv->getLeafDomain()) {
    if (id->getParallelType() == ParallelType::Bulk) {
      id = consumer_to_gmem(id);
      bulk_ids.insert(id);
      pending.push_back(id);
    }
  }
  // Use a BFS-like (not exactly BFS) algorithm to propagate back to get all
  // bulk IterDomains
  bool updated = true;
  while (true) {
    auto id = pending.front();
    pending.pop_front();
    if (id == nullptr) {
      if (updated) {
        // We discovered new bulk IterDomains in the last round, so we need to
        // continue start a new round to see if we can discover more bulk
        // IterDomains.
        pending.push_back(nullptr);
        updated = false;
        continue;
      } else {
        // We have visited all IterDomains in pending for one round, but nothing
        // has changed. This means that all IterDomains in pending are
        // originating bulk IterDomains, so we can no longer propagate further.
        break;
      }
    }

    auto def = id->definition();
    bool should_propagate = false;
    if (allocation_domain_set.count(id) == 0) {
      // We only continue propagating if we have not reached the allocation
      // domain yet.
      NVF_ERROR(
          def != nullptr,
          "Allocation domain is unreachable from ",
          id->toString());

      if (bulk_ids.count(def->input(0)->as<IterDomain>()) > 0) {
        // already processed from another path
        continue;
      }

      should_propagate = std::all_of(
          def->outputs().begin(), def->outputs().end(), [&](Val* out) {
            return bulk_ids.count(out->as<IterDomain>()) > 0;
          });
    }

    if (should_propagate) {
      updated = true;
      for (auto id : def->inputs()) {
        if (bulk_ids.insert(id->as<IterDomain>()).second) {
          pending.push_back(id->as<IterDomain>());
        }
      }
    } else {
      pending.push_back(id);
    }
  }

  // Get originating bulk IterDomains. Use VectorOfUniqueEntries instead of
  // std::unordered_set to make the algorithm deterministic. However, the
  // order here has no meaning, especially, is is not the order specifying which
  // IterDomain is inner and which is outer. The actual order must be determined
  // by propagating from the allocation domain.
  VectorOfUniqueEntries<IterDomain*> originating_bulk_ids;
  for (auto id : pending) {
    if (id == nullptr) {
      continue;
    }
    originating_bulk_ids.pushBack(id);
  }

  // Step 2: Get boxId and tmaGlobalId, and innerId for each originating bulk
  // IterDomain. Similarily, the order of the `tma_global_ids` has no meaning.
  // We are using a std::vector<Val*> just to make the algorithm deterministic,
  // not because we care about its order.

  std::vector<Val*> tma_global_ids;
  std::unordered_map<IterDomain*, IterDomain*> tma_global_id_to_box_id;
  std::unordered_map<IterDomain*, IterDomain*> tma_global_id_to_orig_bulk_id;
  std::unordered_map<IterDomain*, IterDomain*> tma_global_id_to_inner_id;
  for (auto id : originating_bulk_ids) {
    auto def = dynamic_cast<Split*>(id->definition());
    IterDomain* box_id =
        (def != nullptr && def->outer() == id ? def->in() : id);
    IterDomain* inner_id =
        (def != nullptr && def->outer() == id ? def->inner() : nullptr);
    auto bdef = dynamic_cast<Split*>(box_id->definition());
    IterDomain* tma_global_id =
        (bdef != nullptr && bdef->inner() == box_id ? bdef->in() : box_id);

    tma_global_ids.push_back(tma_global_id);
    tma_global_id_to_box_id[tma_global_id] = box_id;
    tma_global_id_to_orig_bulk_id[tma_global_id] = id;
    if (inner_id != nullptr) {
      tma_global_id_to_inner_id[tma_global_id] = inner_id;
    }
  }

  // Stpe 3: Propagate from the allocation domain to TMA-global IterDomains,
  // compute the order, contiguity, and stride of TMA-global IterDomains. Note
  // that this order is meaningful, and it is the order that defines which is
  // inner and which is outer. The strides are also meaningful, and they are the
  // `globalStrides` of the `cuTensorMapEncodeTiled`. After propagation,
  // `frontier` will be the TMA domain.

  std::list<std::tuple<IterDomain*, /*contiguity*/ bool, /*stride*/ Val*>>
      frontier;
  // Initialize frontier as the allocation domain
  auto metadata = IrBuilder::metadataExpr(gmem_tv);
  auto alloc_strides = IrBuilder::getAttrExpr(metadata, "alloc_stride");
  for (auto i : c10::irange((int64_t)allocation_domain.size())) {
    auto id = allocation_domain.at(i);
    // TODO: should I use i below, or should I instead use the position of id in
    // the allocation domain with broadcast? I don't remember the detail, but
    // I will just use i for now and leave the support for broadcast for future.
    auto stride = IrBuilder::getItemExpr(alloc_strides, i);
    frontier.emplace_back(id, gmem_tv->getContiguity().at(i).value(), stride);
  }
  // Propagate forward from the allocation domain to TMA-global IterDomains
  for (Expr* expr : DependencyCheck::getAllExprsBetween(
           allocation_domain_set, tma_global_ids)) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      auto in = split->in();
      auto in_it =
          std::find_if(frontier.begin(), frontier.end(), [in](auto tuple) {
            return std::get<0>(tuple) == in;
          });
      NVF_ERROR(
          in_it != frontier.end(),
          "The set of all TMA-global IterDomains must be equivalent to the allocation domain, but ",
          in->toString(),
          " is not on the path.");
      Val* is_divisible = SimplifyingIrBuilder::eqExpr(
          SimplifyingIrBuilder::modExpr(in->extent(), split->factor()),
          gmem_tv->fusion()->zeroVal());
      GpuLower::current()->validate(
          is_divisible,
          "Invalid view in TMA: the extent of ",
          in,
          " must be divisible by ",
          split->factor());
      frontier.insert(
          in_it,
          std::make_tuple(
              split->outer(),
              true,
              SimplifyingIrBuilder::mulExpr(
                  std::get<2>(*in_it), split->factor())));
      std::get<0>(*in_it) = split->inner();
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      auto outer = merge->outer();
      auto outer_it =
          std::find_if(frontier.begin(), frontier.end(), [outer](auto tuple) {
            return std::get<0>(tuple) == outer;
          });
      NVF_ERROR(
          outer_it != frontier.end(),
          "The set of all TMA-global IterDomains must be equivalent to the allocation domain, but ",
          outer->toString(),
          " is not on the path.");
      auto inner = merge->inner();
      auto inner_it = std::next(outer_it);
      NVF_ERROR(
          inner_it != frontier.end(),
          "The set of all TMA-global IterDomains must be equivalent to the allocation domain, but ",
          inner->toString(),
          " is not on the path.");
      NVF_ERROR(
          std::get<0>(*inner_it) == inner && std::get<1>(*outer_it),
          "Can not merge discontiguous IterDomains, but ",
          outer->toString(),
          " is merged with ",
          inner->toString());
      std::get<0>(*inner_it) = merge->out();
      frontier.erase(outer_it);
    } else {
      NVF_ERROR(
          false,
          "Unsupported expression between the allocation domain and the TMA-global IterDomains: ",
          expr->toString());
    }
  }

  NVF_ERROR(
      std::get<1>(frontier.back()),
      "The innermost IterDomain of the allocation domain must be contiguous");

  // Validate that frontier is a superset of tma_global_ids, otherwise there is
  // something wrong in the schedule.
  std::unordered_set<IterDomain*> seen;
  std::unordered_set<Val*> pending_tma_global_ids(
      tma_global_ids.begin(), tma_global_ids.end());
  for (auto tuple : frontier) {
    auto id = std::get<0>(tuple);
    NVF_ERROR(
        seen.insert(id).second,
        "Mistake in schedule. Duplicate IterDomain found: ",
        id->toString());
    pending_tma_global_ids.erase(id);
  }
  NVF_ERROR(
      pending_tma_global_ids.empty(),
      "The set of all TMA-global IterDomains must a subset of the TMA domain, but ",
      ir_utils::toString(pending_tma_global_ids),
      " is not in the TMA domain.");

  // Step 4: Compute the tensor map descriptor and the index

  // As required by the hardware, tensors used by TMA must be in column major
  // that is, stride[0] must be implicitly 1 (therefore omitted)

  std::vector<Val*> tensor_sizes_inner_to_outer;
  std::vector<Val*> tensor_strides_inner_to_outer;
  std::vector<Val*> box_sizes_inner_to_outer;
  std::vector<Val*> element_strides_inner_to_outer;
  std::vector<Val*> indices_inner_to_outer;

  int64_t itemsize = dataTypeSize(gmem_tv->dtype());

  // The frontier is now the TMA domain. If the TMA domain contains <= 5
  // IterDomains, it is possible to treat each IterDomain as a dimension of TMA.
  // However, in order to be able to still use TMA if the TMA domain contains
  // more than 5 IterDomains, we decide to merge contiguous IterDomains together
  // here to reduce the dimensionality when possible.
  //
  // There can only be three types of IterDomains in the TMA domain:
  // -  NG: non-TMA-global ID
  // -  BG: bulk TMA-global ID
  // - NBG: non-bulk TMA-global ID
  //
  // An NG ID is an IterDomain that is not a TMA-global ID, in another word,
  // none of its decendants are bulk. They are the IDs that are implicitly one
  // tiled. In the Diagram 1 above, I0 is an NG ID. A BG ID is an IterDomain
  // that is both TMA-global and bulk. It comes from the user marking a
  // dimension of the TMA domain as bulk without creating a box. That is, this
  // dimension is whole tiled. In the Diagram 1 above, I3 is such an IterDomain.
  // An NBG ID is an IterDomain that is TMA-global but not bulk. It comes from
  // the user creating a box with a split operation for this dimension. In the
  // Diagram 1 above, I1 and I2 are NBG IDs. Depending on the contiguity and
  // types of the IterDomains in the TMA domain, we may or may not be able to
  // merge them together. Here are a few examples:
  //
  // Example 1: If my TMA domain is [I1, I2, I3], with contiguity [T, T, T],
  // I1 and I2 are NG IDs, and I3 is an NBG ID, then we will choose to use 2D
  // TMA: [(I1, I2), (I3)], where I1 and I2 are considered as "merged" together
  // because: Assuming that we did an imaginary transformation:
  //   Im = merge(I1, I2);
  //   I12, Ib = split(Im, 1);
  //   Ib->parallelize(ParallelType::Bulk);
  //   I1', I2' = split(I12, I2->extent());
  // then I1' and I2' would be completely equivalent to I1 and I2. If we pluged
  // in the above imaginary transformation into the TensorDomain and used I1'
  // and I2' to substitute I1 and I2, we would not change what we were
  // computing, but after this change, we would have TMA domain [Im, I3] which
  // has less dimension than the original TMA domain. The box size of the first
  // dim, Im, was one.
  //
  // Example 2: Similar to Example 1, except that the contiguity is [F, T, T].
  // For this case, we will choose to use 3D TMA: [(I1), (I2), (I3)]. I1 and I2
  // will not be considered as "merged" this time, because they are not
  // contiguous. Merging discontiguous IterDomains in the allocation domain to
  // get TMA domain is not allowed.
  //
  // Example 3: Similar to Example 1, except that I3 is a BG ID. This time, we
  // will choose to use 1D TMA [(I1, I2, I3)], where the box size is the extent
  // of I3. Because, if we did an imaginary transformation:
  //   Im = merge(merge(I1, I2), I3);
  //   I12, I3' = split(Im, I3->extent());
  //   I3'->parallelize(ParallelType::Bulk);
  //   I1', I2' = split(I12, I2->extent());
  // then I1', I2', and I3' would be completely equivalent to I1, I2 and I3. If
  // we pluged in the above imaginary transformation into the TensorDomain and
  // used I1', I2' and I3' to substitute I1, I2 and I3, we would not change what
  // we were computing, but after this change, we would have TMA domain [Im]
  // which has less dimension than the original TMA domain. The box size of the
  // only dim, Im, was I3->extent().
  //
  // Example 4: If my TMA domain is [I1, I2, I3, I4], with contiguity all true.
  // I1 and I2 are NG IDs, I3 and I4 are BG IDs. Then similar to the above
  // argument, we will choose to use 1D TMA: [(I1, I2, I3, I4)], with the box
  // size I3->extent() * I4->extent().
  //
  // Example 5: Similar to Example 4, except that the contiguity is [T, F, T,
  // T]. For this case, the algorithm will choose to use 2D TMA: [(I1, I2), (I3,
  // I4)]. The box size of the first dim, (I1, I2), is one. The second dim,
  // (I3, I4), has only one box, whose size is the tensor size at this
  // dimension.
  //
  // The algorithm that does the above analysis is as follows: We run a 3-state
  // machine. The state machine is initialized as START. After setting the
  // initial state, we loop through the frontier from inner to outer. During the
  // loop, for each IterDomain we see, we take an action and change the state of
  // the machine. The action and target state depend on the current state of the
  // machine, and the type and contiguity of the IterDomain we encounter. The
  // actions and transition of states are shown in the following diagram:
  //
  //                                   NBG:
  //                              create new dim
  //                              .-------------.
  //                              |             |
  //                              '-- [START] <-'
  //                        BG:     / ^ NBG: ^ \     NG:
  //                      create   / / create \ \   create
  //                       new    / /   new    \ \   new
  //                       dim   / /    dim     \ \  dim
  //                            v /              \ v
  //                .-- [PENDING BULK] -----> [PENDING NON-BULK] <-.
  //                |           ^ ^      NG:      | |              |
  //                '-----------' |    create     | '--------------'
  //               BG: create new |    new dim    | NG: create new
  //         dim if discontiguous |      if       | dim if discontiguous
  //              merge with prev | discontiguous | merge with prev
  //                dim otherwise |               | dim otherwise
  //                              '---------------'
  //                              BG: create new dim
  //
  // There are three states in the machine. The meaning of these states are:
  // - START: The machine is at the beginning of the loop, or the most recently
  //   discovered dimension says that "please leave me alone and don't try to
  //   merge me with anyone".
  // - PENDING BULK: The most recently discovered dimension is whole tiled so
  //   far. Please check if we can merge more BG IDs to grow the box size, or
  //   check if we can merge an NG ID to make this box part of a dimension
  //   instead of a whole dimension.
  // - PENDING NON-BULK: The most recently discovered dimension is not whole
  //   tiled so far. Please check if we can merge more NG IDs to grow the
  //   whole dimension size.
  enum { START, PENDING_BULK, PENDING_NON_BULK } state = START;
  for (auto it = frontier.rbegin(); it != frontier.rend(); it++) {
    auto id = std::get<0>(*it);
    bool contiguous = std::get<1>(*it);
    auto box_id_it = tma_global_id_to_box_id.find(id);
    if (box_id_it == tma_global_id_to_box_id.end()) {
      // non-TMA-global ID
      bool should_create_new_dim = !(state != START && contiguous);
      if (should_create_new_dim) {
        tensor_sizes_inner_to_outer.push_back(id->extent());
        if (it != frontier.rbegin()) {
          tensor_strides_inner_to_outer.push_back(
              SimplifyingIrBuilder::mulExpr(std::get<2>(*it), itemsize));
        }
        box_sizes_inner_to_outer.push_back(gmem_tv->fusion()->oneVal());
        element_strides_inner_to_outer.push_back(gmem_tv->fusion()->oneVal());

        auto index_it = indexing->indexMap().find(id);
        NVF_ERROR(
            index_it != indexing->indexMap().end(),
            "Can not find index for ",
            id->toString());
        indices_inner_to_outer.push_back(index_it->second);
      } else {
        auto index_it = indexing->indexMap().find(id);
        NVF_ERROR(
            index_it != indexing->indexMap().end(),
            "Can not find index for ",
            id->toString());
        indices_inner_to_outer.back() = SimplifyingIrBuilder::addExpr(
            indices_inner_to_outer.back(),
            SimplifyingIrBuilder::mulExpr(
                tensor_sizes_inner_to_outer.back(), index_it->second));
        tensor_sizes_inner_to_outer.back() = SimplifyingIrBuilder::mulExpr(
            tensor_sizes_inner_to_outer.back(), id->extent());
      }
      state = PENDING_NON_BULK;
    } else {
      // TMA-global ID
      bool should_create_new_dim =
          !(state == PENDING_BULK && bulk_ids.count(id) > 0 && contiguous);
      if (should_create_new_dim) {
        tensor_sizes_inner_to_outer.push_back(id->extent());
        if (it != frontier.rbegin()) {
          tensor_strides_inner_to_outer.push_back(
              SimplifyingIrBuilder::mulExpr(std::get<2>(*it), itemsize));
        }
        box_sizes_inner_to_outer.push_back(box_id_it->second->extent());

        auto inner_it = tma_global_id_to_inner_id.find(id);
        if (it == frontier.rbegin()) {
          NVF_ERROR(
              inner_it == tma_global_id_to_inner_id.end(),
              "When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE ",
              "(this is always the case for nvFuser now)",
              ", the first element of elementStrides must be one");
        }
        if (inner_it != tma_global_id_to_inner_id.end()) {
          element_strides_inner_to_outer.push_back(inner_it->second->extent());
        } else {
          element_strides_inner_to_outer.push_back(gmem_tv->fusion()->oneVal());
        }

        auto index_it = indexing->indexMap().find(id);
        NVF_ERROR(
            index_it != indexing->indexMap().end(),
            "Can not find index for ",
            id->toString());
        indices_inner_to_outer.push_back(index_it->second);
      } else {
        tensor_sizes_inner_to_outer.back() = SimplifyingIrBuilder::mulExpr(
            tensor_sizes_inner_to_outer.back(), id->extent());
        box_sizes_inner_to_outer.back() = SimplifyingIrBuilder::mulExpr(
            box_sizes_inner_to_outer.back(), id->extent());
      }
      state = bulk_ids.count(id) > 0 ? PENDING_BULK : START;
    }
  }

  int64_t dim = (int64_t)tensor_sizes_inner_to_outer.size();
  auto global_address = IrBuilder::getAttrExpr(metadata, "data");

  Val* global_stride =
      (dim > 1
           ? IrBuilder::arrayExpr(tensor_strides_inner_to_outer)
           : IrBuilder::create<Val>(
                 std::vector<int64_t>{},
                 ArrayType{std::make_shared<DataType>(DataType::Index), 0}));

  auto descriptor = encodeTensorMapTiled(
      gmem_tv->dtype(),
      global_address,
      IrBuilder::arrayExpr(tensor_sizes_inner_to_outer),
      global_stride,
      IrBuilder::arrayExpr(box_sizes_inner_to_outer),
      IrBuilder::arrayExpr(element_strides_inner_to_outer),
      tma::TensorMapInterleave::NoInterleave,
      getSwizzleFromBytes(
          getCpAsyncBulkTensorSwizzleSize(smem_tv) * core_matrix_width_bytes),
      tma::TensorMapL2Promotion::NoL2Promotion,
      tma::TensorMapFloatOOBFill::NoOOBFill);

  auto coordinate = IrBuilder::arrayExpr(indices_inner_to_outer);

  Val* index = nullptr;

  if (is_load) {
    std::stringstream ss;
    ss << "Hopper::CpAsyncBulkTensorTileG2SIndex<" << dim << ">";
    index = IrBuilder::structExpr(
        {{"descriptor", IrBuilder::addressExpr(descriptor)},
         {"coordinate", coordinate},
         {"mbarrier", mbarrier}},
        ss.str());
  } else {
    std::stringstream ss;
    ss << "Hopper::CpAsyncBulkTensorTileS2GIndex<" << dim << ">";
    index = IrBuilder::structExpr(
        {{"descriptor", IrBuilder::addressExpr(descriptor)},
         {"coordinate", coordinate}},
        ss.str());
  }

  index = GpuLower::current()->commonScalarMap().hoistScalar(index, loops);

  // Step 5: Compute the expected bytes for the complete_tx mechanism

  Val* expected_bytes = IrBuilder::create<Val>(itemsize, DataType::Index);
  // Note that we need to use the extents of the originating bulk IterDomains
  // to compute the expected bytes, not the extents of the box IterDomains.
  // They are different when element strides are not 1.
  for (auto id : originating_bulk_ids) {
    expected_bytes =
        SimplifyingIrBuilder::mulExpr(expected_bytes, id->extent());
  }
  expected_bytes =
      SimplifyingIrBuilder::maybeCastExpr(DataType::UInt32, expected_bytes);
  expected_bytes =
      GpuLower::current()->commonScalarMap().hoistScalar(expected_bytes, loops);
  auto is_multiple_of_16B = SimplifyingIrBuilder::eqExpr(
      SimplifyingIrBuilder::modExpr(
          expected_bytes, IrBuilder::create<Val>(16, DataType::Index)),
      expected_bytes->fusion()->zeroVal());
  GpuLower::current()->validate(
      is_multiple_of_16B,
      "The expected bytes must be a multiple of 16 bytes, but ",
      expected_bytes,
      " is not.");

  return {IrBuilder::create<kir::TensorIndex>(gmem_tv, index), expected_bytes};
}

} // namespace nvfuser

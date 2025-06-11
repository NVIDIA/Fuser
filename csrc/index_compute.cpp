// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <index_compute.h>

#include <ATen/cuda/CUDAContext.h>

#include <contiguity.h>
#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/circular_buffer.h>
#include <device_lower/pass/magic_zero.h>
#include <device_lower/pass/unroll.h>
#include <device_lower/validation.h>
#include <expr_simplifier.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <ops/arith.h>
#include <swizzle.h>
#include <transform_iter.h>
#include <transform_replay.h>

#include <memory>

namespace nvfuser {

bool IndexCompute::hasUnswitchedDependentDomains(IterDomain* id) const {
  auto concrete_id = maybeGetExactMapConcreteID(id);
  auto it = unswitched_domain_map_.find(concrete_id);
  return it != unswitched_domain_map_.end() && !it->second.empty();
}

void IndexCompute::initializeUnswitchDomainMap() {
  NVF_ERROR(unswitched_domain_map_.empty());
  for (auto id : unswitched_loop_domains_) {
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

    // If the stride is a multiple of the inner extent, the loop
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
    // a resize shows up as part of root to logical transformations, the input
    // to the resize is not indexed as the indexing is done using the logical
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
    std::unordered_set<IterDomain*> preferred_paths)
    : IndexCompute(
          _td,
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_domains),
          std::move(zero_merged_in),
          ContigIDs::getNonContigIDs(),
          std::move(preferred_paths)) {}

IndexCompute::IndexCompute(
    const TensorDomain* _td,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in,
    const ContigIDs& contig_finder,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_set<IterDomain*> unswitched_loop_domains)
    : td_(_td),
      index_map_(std::move(initial_index_map)),
      extent_map_(std::move(extent_map)),
      zero_domains_(std::move(zero_domains)),
      zero_merged_in_(std::move(zero_merged_in)),
      contig_ids_{contig_finder.contigIDs()},
      preferred_paths_(std::move(preferred_paths)),
      unswitched_loop_domains_(std::move(unswitched_loop_domains)) {
  FUSER_PERF_SCOPE("GpuLower::Lower::IndexCompute::IndexCompute");

  if (isOptionDisabled(DisableOption::ContigIndexing)) {
    contig_ids_.clear();
  }

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
    std::unordered_set<IterDomain*> unswitched_loop_domains)
    : td_{nullptr},
      index_map_(std::move(initial_index_map)),
      zero_domains_(std::move(zero_domains)),
      preferred_paths_(std::move(preferred_paths)),
      concrete_id_pass_{true},
      swizzle_mode_{SwizzleMode::Loop},
      unswitched_loop_domains_(std::move(unswitched_loop_domains)) {
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
  const std::vector<Val*> domain_vals(td_->loop().begin(), td_->loop().end());
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

      if (auto it = unswitched_loop_domains_.find(prev_id);
          it != unswitched_loop_domains_.end()) {
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
      updated_unswitched_domains);

  updated_index_compute.run();

  return updated_index_compute;
}

namespace {
// Map indices down to the loop domains for applying swizzle
class UpdateLeafIndices : public IterVisitor {
 public:
  UpdateLeafIndices(
      const TensorDomain* td,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> extent_map)
      : td_(td),
        index_map_(std::move(initial_index_map)),
        extent_map_(std::move(extent_map)) {
    const std::vector<Val*> domain_vals(td_->loop().begin(), td_->loop().end());

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

Val* getExtentOfRootAxis(IterDomain* id, Val* normal_extent = nullptr) {
  // If id is device dim, ignore the extent which holds the unsharded extent.
  if (id->isDeviceDim()) {
    normal_extent = GpuLower::current()->kernel()->oneVal();
  } else if (normal_extent == nullptr) {
    normal_extent = id->extent();
  }

  return normal_extent;
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
    UpdateLeafIndices update_loop(td_, indexMap(), extentMap());
    index_map_ = update_loop.indexMap();
    extent_map_ = update_loop.extentMap();
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
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      [&](IterDomain* tv_id) {
        // Matching is done using the index and loop maps. See
        // validateParallelize as well.
        return ca_map->areMapped(loop_id, tv_id, IdMappingMode::EXACT) ||
            ca_map->areMapped(loop_id, tv_id, IdMappingMode::PERMISSIVE);
      });

  // There's no mapped producer ID. Zero substitution shouldn't be
  // done.
  if (it == tv->getLoopDomain().end()) {
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
        "Found invalid parallelization that should have been detected by the "
        "parallel validation: loop ID: ",
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
std::pair<std::unordered_map<ForLoop*, Val*>, std::unordered_set<ForLoop*>>
indexMapFromTV(
    const TensorView* tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    ForLoop* alloc_loop,
    bool as_consumer,
    ForLoop* circular_buffer_loop) {
  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  const bool is_global = tv->getMemoryType() == MemoryType::Global;
  const bool is_shared = tv->getMemoryType() == MemoryType::Shared;

  std::unordered_map<ForLoop*, Val*> loop_to_ind_map;

  // Check if the current op has an implicit loop implemented
  //  within an mma instruction.
  bool within_mma_loops =
      std::any_of(loops.begin(), loops.end(), [](ForLoop* fl) {
        return fl->iter_domain()->isMma();
      });

  // Track domains that do not contibute to the resulting
  // index. Previously, index->isZeroInt() was used to detect such
  // domains, but that's not a reliable method as we may set an
  // initial index to zero for unswitch.
  std::unordered_set<ForLoop*> zero_loops;

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

    if (loop == circular_buffer_loop) {
      const int64_t prefetch_distance =
          GpuLower::current()
              ->circularBufferInfo()
              .getCircularBufferOptionsFor(loop->iter_domain())
              .prefetch;
      idx = SimplifyingIrBuilder::addExpr(
          idx,
          SimplifyingIrBuilder::create<Val>(
              prefetch_distance, DataType::Index));
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
    ForLoop* alloc_loop,
    const std::vector<ForLoop*>& loops,
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
    if (loop->vectorize() ||
        nvfuser::ir_utils::isMemoryPartitionedAcross(
            tv->getMemoryType(), loop_id->getParallelType())) {
      continue;
    }
    // Look for a domain that is mapped with the loop. If mapped in
    // the loop map, the loop index should be used for indexing of the
    // tensor, except for broadcast and reduction domains.
    auto it = std::find_if(
        tv->getLoopDomain().begin(),
        tv->getLoopDomain().end(),
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
    if (it != tv->getLoopDomain().end()) {
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
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getGlobalProducerIndex");

  auto alloc_indices = getProducerAllocationIndices(
      producer_tv, consumer_tv, loops, rotated_loops, override_index);

  const auto& alloc_dom = producer_tv->getMaybeAllocationDomain();

  // TODO: Abstract stride logic to reuse with consumer indexing
  std::vector<Val*> strides(alloc_dom.size(), nullptr);
  {
    int stride_i = 0;
    for (const auto i : arange(alloc_dom.size())) {
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
  for (const auto i : arange(alloc_dom.size())) {
    auto dim = alloc_dom.size() - i - 1;
    if (alloc_dom[dim]->isReduction()) {
      continue;
    }

    auto producer_dim_contiguity = producer_tv->domain()->contiguity().at(dim);
    if (alloc_dom[dim]->isBroadcast()) {
      strides[dim] = cur_contig_stride->fusion()->zeroVal();
      NVF_ERROR(!producer_dim_contiguity.has_value());
    } else if (!producer_dim_contiguity.has_value()) {
      NVF_THROW("Expected value for dimension contiguity");
    } else if (producer_dim_contiguity.value()) {
      // If contig, used the stored stride which may be the previous
      // dimensions stride * previous dimensions size
      strides[dim] = cur_contig_stride;
      // Prepare for the next dimension which may also be contiguous, multiply
      // by extent of this dimension
      auto alloc_dim_extent = getExtentOfRootAxis(alloc_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(cur_contig_stride, alloc_dim_extent);
    } else {
      // If non contiguous dimension, keep local stride information, set cur
      // stride to local stride * local raw extent
      auto alloc_dim_extent = getExtentOfRootAxis(alloc_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(strides[dim], alloc_dim_extent);
    }
  }

  auto vectorize_shift =
      loops.empty() ? nullptr : loops.back()->vectorize_shift();

  // Global striding
  std::vector<Val*> strided_inds(
      alloc_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : arange(alloc_dom.size())) {
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
      PairwiseLogicalDomainMap(producer_tv, consumer_tv));

  // Grab consumer domain entries and reverse replay map. TODO: Maybe
  // TransformReplay::replayPasC could return this map
  for (auto id : consumer_tv->getLoopDomain()) {
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
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  bool is_mma_input = consumer_tv->definition()->isA<MmaOp>();
  const auto gpu_lower = GpuLower::current();
  // Replay producer to look like consumer so we can index on producer since our
  // loop nests look like consumer
  auto pairwise_map = PairwiseLogicalDomainMap(producer_tv, consumer_tv);
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
  // map. All producer id's don't exist in the compute at map. The logical axes
  // all may be, but since I haven't proven that to be the case, going to do a
  // more conservative approach, which is to use the consumer as a proxy between
  // producer to reference.
  std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_producer;
  std::unordered_map<IterDomain*, IterDomain*> c2p_index_map;

  // Map sent to best effort replay needs to match the exact incantation for
  // compute_at_mode.cpp with MappingMode::Index
  auto c2p_logical_map = PairwiseLogicalDomainMap(producer_tv, consumer_tv)
                             .mapBroadcast(false)
                             .mapConsumerToProducer();

  // This replay has to be consistent with compute at index map.
  BestEffortReplay replay_producer_as_consumer(
      producer_tv->getLoopDomain(),
      consumer_tv->getLoopDomain(),
      c2p_logical_map);

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

  for (const auto i : arange(alloc_dom.size())) {
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

    if (alloc_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (const auto j : arange(i + 1, alloc_dom.size())) {
      if (skip_indexing.count(alloc_dom[j])) {
        continue;
      }

      auto alloc_ext_j = (extent_map.find(alloc_dom[j]) == extent_map.end() ||
                          is_mma_allocation(alloc_dom[j]))
          ? alloc_dom[j]->extent()
          : extent_map.at(alloc_dom[j]);

      alloc_ext_j = getExtentOfRootAxis(alloc_dom[j], alloc_ext_j);

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

  if (producer_tv->isCircularBuffered()) {
    auto db_loop = gpu_lower->circularBufferInfo().getCircularBufferLoop(
        producer_tv, loops, true);
    if (db_loop != nullptr) {
      const int64_t stage_depth =
          gpu_lower->circularBufferInfo()
              .getCircularBufferOptionsFor(db_loop->iter_domain())
              .stage;
      auto loop_index = db_loop->indexOrStartIfTrivial();
      if (rotated_loops.count(db_loop) > 0) {
        loop_index = SimplifyingIrBuilder::addExpr(loop_index, db_loop->step());
      }
      auto db_switch_index = SimplifyingIrBuilder::modExpr(
          loop_index,
          SimplifyingIrBuilder::create<Val>(stage_depth, DataType::Index));
      auto original_alloc_size =
          gpu_lower->circularBufferInfo().getOriginalAllocSize(producer_tv);
      auto db_strided_index =
          SimplifyingIrBuilder::mulExpr(db_switch_index, original_alloc_size);
      strided_inds.push_back(db_strided_index);
    }
  }

  return strided_inds;
}

Val* Index::getLinearLogicalIndex(
    TensorView* consumer_tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops) {
  if (!ir_utils::hasRootToLoopLinearTransformations(consumer_tv) ||
      ir_utils::isCpAsyncBulkLoad(consumer_tv->definition()) ||
      GpuLower::current()->idModelOptions().consumerIndex() ||
      GpuLower::current()->tmemInfo().hasTMemTensor()) {
    NVF_ERROR(rotated_loops.empty(), "Loop rotation is not supported");
    const TensorIndexer& indexer = GpuLower::current()->tensorIndexer();
    auto per_dim_indices = indexer.getIndexFor(
        consumer_tv->definition(),
        /*as_consumer=*/true,
        consumer_tv->getLogicalDomain(),
        loops,
        /*use_magic_zero=*/true);
    Val* stride = consumer_tv->fusion()->oneVal();
    for (const auto [i, logical_id] :
         enumerate(consumer_tv->getLogicalDomain()) | std::views::reverse) {
      auto per_dim_index = per_dim_indices.at(i);
      auto per_dim_strided_index =
          SimplifyingIrBuilder::mulExpr(per_dim_index, stride);
      per_dim_indices.at(i) = per_dim_strided_index;
      stride = SimplifyingIrBuilder::mulExpr(stride, logical_id->extent());
    }
    return sumVals(per_dim_indices);
  } else {
    auto guard = ir_utils::allocateToLogicalDomainGuard(consumer_tv, true);
    return sumVals(
        getGlobalConsumerStridedIndices(consumer_tv, loops, rotated_loops));
  }
}

std::vector<Val*> Index::getConsumerPerDimLogicalIndex(
    TensorView* consumer_tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops) {
  if (!ir_utils::hasRootToLoopLinearTransformations(consumer_tv) ||
      GpuLower::current()->idModelOptions().consumerIndex() ||
      GpuLower::current()->tmemInfo().hasTMemTensor()) {
    NVF_ERROR(rotated_loops.empty(), "Loop rotation is not supported");
    const TensorIndexer& indexer = GpuLower::current()->tensorIndexer();
    return indexer.getIndexFor(
        consumer_tv->definition(),
        /*as_consumer=*/true,
        consumer_tv->getLogicalDomain(),
        loops);
  } else {
    auto guard = ir_utils::allocateToLogicalDomainGuard(consumer_tv, false);
    IndexFromIdGraph index_from_id_graph =
        getTensorIndexFromIdGraph(loops, rotated_loops, consumer_tv);
    return getConsumerAllocationIndices(
        consumer_tv, loops, index_from_id_graph);
  }
}

std::vector<Val*> Index::getProducerPerDimLogicalIndex(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  if (!ir_utils::hasRootToLoopLinearTransformations(producer_tv) ||
      GpuLower::current()->idModelOptions().producerIndex() ||
      GpuLower::current()->tmemInfo().hasTMemTensor()) {
    NVF_ERROR(rotated_loops.empty(), "Loop rotation is not supported");
    const TensorIndexer& indexer = GpuLower::current()->tensorIndexer();
    return indexer.getIndexFor(
        consumer_tv->definition(),
        /*as_consumer=*/false,
        producer_tv->getLogicalDomain(),
        loops);
  } else {
    auto guard = ir_utils::allocateToLogicalDomainGuard(producer_tv, false);
    return getProducerAllocationIndices(
        producer_tv, consumer_tv, loops, rotated_loops, override_index);
  }
}

std::vector<Val*> Index::getStrides(TensorView* tv) {
  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  const auto& alloc_dom = tv->getMaybeAllocationDomain();

  std::vector<Val*> strides(
      alloc_dom.size(), GpuLower::current()->kernel()->oneVal());
  {
    int stride_i = 0;
    for (const auto i : arange(alloc_dom.size())) {
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
  for (const auto i : arange(alloc_dom.size())) {
    auto dim = alloc_dom.size() - i - 1;
    if (alloc_dom[dim]->isReduction() || alloc_dom[dim]->isStride()) {
      continue;
    }

    auto dim_contiguity = tv->domain()->contiguity().at(dim);
    if (alloc_dom[dim]->isBroadcast()) {
      strides[dim] = cur_contig_stride->fusion()->zeroVal();
      NVF_ERROR(!dim_contiguity.has_value());
    } else if (!dim_contiguity.has_value()) {
      NVF_THROW("Expected value for dimension contiguity");
    } else if (dim_contiguity.value()) {
      // If contig, used the stored stride which may be the previous
      // dimensions stride * previous dimensions size
      strides[dim] = cur_contig_stride;
      // Prepare for the next dimension which may also be contiguous, multiply
      // by extent of this dimension
      auto alloc_dim_extent = getExtentOfRootAxis(alloc_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(cur_contig_stride, alloc_dim_extent);
    } else {
      // If non contiguous dimension, keep local stride information, set cur
      // stride to local stride * local raw extent
      cur_contig_stride = SimplifyingIrBuilder::mulExpr(
          strides[dim], getExtentOfRootAxis(alloc_dom[dim]));
    }
  }
  return strides;
}

std::vector<Val*> Index::getConsumerAllocationIndices(
    const TensorView* tv,
    const std::vector<ForLoop*>& loops,
    const IndexFromIdGraph& index_from_id_graph) {
  const auto& alloc_dom = tv->getMaybeAllocationDomain();
  auto indexing = index_from_id_graph.index;

  std::vector<Val*> alloc_inds(
      alloc_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : arange(alloc_dom.size())) {
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

    alloc_inds[i] = alloc_ind;
  }
  return alloc_inds;
}

std::vector<Val*> Index::getProducerAllocationIndices(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getProducerAllocationIndices");
  // Replay producer to look like consumer so we can index on producer since
  // our loop nests look like consumer
  auto pairwise_map =
      PairwiseLogicalDomainMap(producer_tv, consumer_tv).mapBroadcast(true);

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
  auto c2p_logical_map = PairwiseLogicalDomainMap(producer_tv, consumer_tv)
                             .mapBroadcast(false)
                             .mapConsumerToProducer();

  // This replay has to be consistent with compute at index map.
  BestEffortReplay replay_producer_as_consumer(
      producer_tv->getLoopDomain(),
      consumer_tv->getLoopDomain(),
      c2p_logical_map);

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
  //     loop: I0, I3, I4
  //   producer:
  //     root I5, I6, I7
  // where I3, I4 = swizzle(I1, I2) , then the c2p map will be I3->I6, I4->I7,
  // I1 and I2 are not mapped. For this case, we should allow the root unmapped,
  // If we add I1->I6 and I2->I7, the c2p map will no longer be injective, which
  // is not what we want.
  const auto p2c_map = invertOneToOneMap(c2p_map);
  for (const auto& kv : PairwiseLogicalDomainMap(producer_tv, consumer_tv)
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

  for (const auto i : arange(alloc_dom.size())) {
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

    alloc_inds.at(i) = alloc_ind;
  }

  return alloc_inds;
}

std::vector<Val*> Index::getGlobalConsumerStridedIndices(
    TensorView* consumer_tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
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
  for (const auto i : arange(alloc_inds.size())) {
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
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
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
  for (const auto i : arange(alloc_dom.size())) {
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
    for (const auto j : arange(i + 1, alloc_dom.size())) {
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

      alloc_ext_j = getExtentOfRootAxis(alloc_dom[j], alloc_ext_j);

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
  // consumer tensor is located. If it's circular buffered and not in
  // the prologue loop, strided_inds ends up having one more
  // index, so it's just much simpler to check here before adding the
  // additional index for circular buffering.
  NVF_ERROR(
      strided_inds.size() == consumer_tv->getMaybeAllocationDomain().size());

  if (consumer_tv->isCircularBuffered()) {
    auto db_loop = gpu_lower->circularBufferInfo().getCircularBufferLoop(
        consumer_tv, loops);
    const auto& opt =
        gpu_lower->circularBufferInfo().getCircularBufferOptionsFor(
            db_loop->iter_domain());
    bool is_circular_buffer_loop = opt.stage > 2;
    bool is_prolog =
        db_loop->circularBufferLoopStage() == CircularBufferLoopStage::Prolog;

    Val* db_switch_index = nullptr;

    // In circular buffered we don't materialize the prolog loop as there will
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
                    opt.prefetch, DataType::Index)),
            SimplifyingIrBuilder::create<Val>(opt.stage, DataType::Index));
      }

      // Use the generated switching buffer index to access the buffer space.
      auto original_alloc_size =
          gpu_lower->circularBufferInfo().getOriginalAllocSize(consumer_tv);
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
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
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
              dataTypeSizeByte(*producer->getDataType()), *index->getDataType()));
      return IrBuilder::addExpr(
          IrBuilder::baseAddressExpr(producer), index_bytes);
    } else {
      return index;
    }
  }
}

namespace {

bool shouldUseTensorIndexer(
    const TensorView* producer,
    const TensorView* consumer,
    const std::unordered_set<ForLoop*>& rotated_loops) {
  // Check if TensorIndexer is definitely required
  auto is_tensor_indexer_required = [&]() -> bool {
    bool is_producer_tma_op = producer->definition() != nullptr &&
        producer->definition()->isA<LoadStoreOp>() &&
        ir_utils::isCpAsyncBulkLoad(producer->definition());
    bool is_consumer_tma_op = consumer->definition() != nullptr &&
        consumer->definition()->isA<LoadStoreOp>() &&
        ir_utils::isCpAsyncBulkLoad(consumer->definition());

    return !ir_utils::hasRootToLoopLinearTransformations(producer) ||
        (consumer->definition()->isA<MmaOp>() &&
         isHopper(consumer->definition()->as<MmaOp>()->macro())) ||
        is_producer_tma_op || is_consumer_tma_op ||
        GpuLower::current()->tmemInfo().hasTMemTensor();
  };

  // Check if TensorIndexer is supported.
  auto is_tensor_indexer_supported = [&](bool assert) -> bool {
    bool is_producer_ldmatrix_op = producer->definition() != nullptr &&
        producer->definition()->isA<LoadStoreOp>() &&
        producer->definition()->as<LoadStoreOp>()->opType() ==
            LoadStoreOpType::LdMatrix;
    bool is_producer_stmatrix_op_with_no_alloc_domain =
        producer->definition() != nullptr &&
        producer->definition()->isA<LoadStoreOp>() &&
        producer->definition()->as<LoadStoreOp>()->opType() ==
            LoadStoreOpType::StMatrix &&
        !producer->hasAllocation();

    if (assert) {
      NVF_ERROR(
          !is_producer_ldmatrix_op,
          "TensorIndexer required but not supported as the producer is "
          "produced by ldmatrix: ",
          producer->definition()->toString());
      NVF_ERROR(
          !is_producer_stmatrix_op_with_no_alloc_domain,
          "TensorIndexer required but not supported as the producer is "
          "produced by stmatrix and it does not have allocation domain: ",
          producer->definition()->toString());
      NVF_ERROR(
          rotated_loops.empty(),
          "TensorIndexer required but not supported as loop rotation is used");
    }

    return !is_producer_ldmatrix_op &&
        !is_producer_stmatrix_op_with_no_alloc_domain && rotated_loops.empty();
  };

  // TensorIndexer is always used if it's required
  if (is_tensor_indexer_required()) {
    // Make sure it's supported
    is_tensor_indexer_supported(/*assert=*/true);
    return true;
  }

  // If opted in, TensorIndexer is used as long as it's supported
  if (GpuLower::current()->idModelOptions().producerIndex() &&
      is_tensor_indexer_supported(/*assert=*/false)) {
    return true;
  }

  return false;
}

} // namespace

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index,
    bool generate_pointer,
    DataType as_type) {
  Val* index = nullptr;

  if (shouldUseTensorIndexer(producer, consumer, rotated_loops)) {
    index = GpuLower::current()->tensorIndexer().getLinearIndex(
        producer, consumer->definition(), loops, override_index);
    if (generate_pointer) {
      auto address_offset = index;
      if (producer->getMemoryType() == MemoryType::Shared) {
        auto producer_dt = producer->getDataType();
        NVF_ERROR(producer_dt.has_value());
        auto index_dt = index->getDataType();
        NVF_ERROR(index_dt.has_value());
        address_offset = SimplifyingIrBuilder::mulExpr(
            address_offset,
            IrBuilder::create<Val>(dataTypeSizeByte(*producer_dt), *index_dt));
      }
      index = SimplifyingIrBuilder::addExpr(
          IrBuilder::baseAddressExpr(producer), address_offset);
    }
  } else {
    index = getProducerStridedIndices(
        producer,
        consumer,
        loops,
        rotated_loops,
        override_index,
        generate_pointer);
  }

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
        NVF_THROW(
            "Unexpected output vectorizaiton for ldmatrix, expect 2, 4, or 8, "
            "get ",
            items_per_thread);
      }
      IrBuilder::create<UnaryOp>(op, index, orig_index);
    }
  }
  return IrBuilder::create<kir::TensorIndex>(producer, index, as_type);
}

Val* Index::getConsumerStridedIndices(
    TensorView* consumer,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
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
              dataTypeSizeByte(*consumer->getDataType()), *index->getDataType()));
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
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    const std::unordered_map<int, Val*>& override_index,
    bool generate_pointer,
    DataType as_type) {
  Val* index = nullptr;
  if (!ir_utils::hasRootToLoopLinearTransformations(consumer) ||
      ir_utils::isCpAsyncBulkLoad(consumer->definition()) ||
      GpuLower::current()->idModelOptions().consumerIndex() ||
      GpuLower::current()->tmemInfo().hasTMemTensor()) {
    NVF_ERROR(rotated_loops.empty(), "Loop rotation is not supported");
    index = GpuLower::current()->tensorIndexer().getLinearIndex(
        consumer, consumer->definition(), loops);
    if (generate_pointer) {
      auto address_offset = index;
      if (consumer->getMemoryType() == MemoryType::Shared) {
        auto consumer_dt = consumer->getDataType();
        NVF_ERROR(consumer_dt.has_value());
        auto index_dt = index->getDataType();
        NVF_ERROR(index_dt.has_value());
        address_offset = SimplifyingIrBuilder::mulExpr(
            index,
            IrBuilder::create<Val>(dataTypeSizeByte(*consumer_dt), *index_dt));
      }
      index = SimplifyingIrBuilder::addExpr(
          IrBuilder::baseAddressExpr(consumer), address_offset);
    }
  } else {
    index = getConsumerStridedIndices(
        consumer, loops, rotated_loops, override_index, generate_pointer);
  }

  index = GpuLower::current()->commonScalarMap().hoistScalar(index, loops);
  return SimplifyingIrBuilder::create<kir::TensorIndex>(
      consumer, index, as_type);
}

namespace {

// Find iteration domains in the history of a consumer to predicate comprised
// only of merge operations. Only return iteration domains that are subsequently
// fed into a split, or are in the provided domain. In other words, we don't
// want to return every IterDomain that's contiguous, just the one closest to
// the loop domain. Predicates are not associated with physical memory so we can
// treat all of them as contiguous merges.
//
// TODO: This seems to have a large overlap with ContigIDs. Consider
// refactoring.
std::vector<PredicateDomainInfo> getPredicateContigIds(
    TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& consumer_index_map) {
  const auto gpu_lower = GpuLower::current();

  // When there's a resize expr between the root and the logical
  // domains, predicate the logical domain. Otherwise, predicate the
  // root domain. The actual size of an IterDomain after resize
  // changes, and the output IterDomain needs to be used to generate
  // its predicate.
  const auto& consumer_root_domain = ir_utils::hasResizedRfactor(consumer_tv)
      ? consumer_tv->getLogicalDomain()
      : consumer_tv->getMaybeRootDomain();

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
  for (auto root_i : arange(consumer_root_domain.size())) {
    auto root_id = consumer_root_domain[root_i];
    if (root_id->maybePartial()) {
      final_ids.insert(root_id);
      continue;
    }
  }

  ContigIDs contig_finder(
      consumer_tv->getLoopDomain(),
      consumer_root_domain,
      TensorDomain::getContiguityFilledWith(consumer_root_domain, true),
      final_ids,
      concrete_index_map,
      GpuLower::current()->divisibleSplitSet(),
      GpuLower::current()->caMap(),
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

// Get the start and stop limit offsets that define the valid range to
// compute. In the simplest case, they are just 0 and
// IterDomain::extent. However, IterDomain may have non-zero start and
// stop that's different from extent.
std::pair<Val*, Val*> getStartAndStopLimitOffsets(IterDomain* consumer_id) {
  NVF_ERROR(consumer_id != nullptr);

  Val* start_limit = consumer_id->start();
  Val* stop_limit = SimplifyingIrBuilder::negExpr(consumer_id->stopOffset());

  return {start_limit, stop_limit};
}

// Get the offsets for the start and stop predicates. The offsets
// are to be added to the index.
std::pair<Val*, Val*> getStartAndStopOffsets(
    IterDomain* consumer_id,
    TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& consumer_start_index_map,
    const std::unordered_map<IterDomain*, Val*>& consumer_stop_index_map,
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

  // Get the boundaries of two ends
  auto limits = getStartAndStopLimitOffsets(consumer_id);

  // At this point, we have everything to create both start and stop
  // predicates as:
  //
  //  index + start_offset >= start_limit
  //  index + stop_offset  < extent + stop_limit
  //
  // start_offset and stop_limit are both zero (was not the case with shift)
  //
  // In order to enable consolidating unswitch predicates, organize
  // the predicates as:
  //
  //  index + (start_offset - start_limit) >= 0
  //  index + (stop_offset - stop_limit)  < extent

  auto start_offset = SimplifyingIrBuilder::negExpr(limits.first);
  auto stop_offset = SimplifyingIrBuilder::negExpr(limits.second);

  return {start_offset, stop_offset};
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

// Returns predicates and the concrete (by loop map) root domains they cover
std::vector<PredicateInfo> Index::getReferenceRootPredicates(
    TensorView* consumer_tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    ForLoop* unswitch_or_vec_loop) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getReferenceRootPredicates");

  const auto gpu_lower = GpuLower::current();

  const bool is_unswitch = unswitch_or_vec_loop != nullptr;

  auto db_axis =
      gpu_lower->circularBufferInfo().getCircularBufferAxis(consumer_tv);

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

  std::vector<PredicateInfo> pred_info_vec;

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

    PredicateInfo info;

    // The final predicates will look like:
    // (index + start_offset) >= 0 && (index + stop_offset) < extent.

    std::tie(info.start_offset_, info.stop_offset_) = getStartAndStopOffsets(
        contig_id,
        consumer_tv,
        consumer_start_index_map,
        consumer_stop_index_map,
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
    auto offsetted_stop_index =
        SimplifyingIrBuilder::addExpr(stop_index, stop_offset);
    auto stop_pred =
        SimplifyingIrBuilder::ltExpr(offsetted_stop_index, contig_id->extent());
    info.stop_predicate_ = stop_pred;

    for (auto consumer_id : contig_id_entry.covered_ids) {
      info.predicated_domains_.insert(consumer_id);
    }
    pred_info_vec.emplace_back(info);
  }

  return pred_info_vec;
}

PredicateInfo PredicateInfo::getFalseInfo() {
  PredicateInfo info;
  info.start_predicate_ = GpuLower::current()->kernel()->falseVal();
  info.stop_predicate_ = GpuLower::current()->kernel()->falseVal();

  return info;
}

Val* Index::iota(
    TensorView* consumer_tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
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
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    DataType dtype) {
  auto indices =
      Index::getConsumerPerDimLogicalIndex(consumer_tv, loops, rotated_loops);
  NVF_ERROR(indices.size() == 2);
  auto result = maybeCastOp(dtype, eq(indices[0], indices[1]));
  return GpuLower::current()->commonScalarMap().hoistScalar(result, loops);
}

std::pair<Val*, Val*> Index::getCpAsyncBulkGmemIndex(
    const LoadStoreOp* ldst,
    Val* mbarrier,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops) {
  FUSER_PERF_SCOPE("Index::getCpAsyncBulkGmemIndex");

  TensorView* producer_tv = ldst->in()->as<TensorView>();
  TensorView* consumer_tv = ldst->out()->as<TensorView>();

  bool is_load = false;
  TensorView* gmem_tv = nullptr;
  if (producer_tv->getMemoryType() == MemoryType::Shared) {
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Global);
    gmem_tv = consumer_tv;
    is_load = false;
  } else {
    NVF_ERROR(producer_tv->getMemoryType() == MemoryType::Global);
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Shared);
    gmem_tv = producer_tv;
    is_load = true;
  }

  NVF_ERROR(
      GpuLower::current()->consumerToTMAInfo().count(consumer_tv),
      "Unable to find TMA info for consumer_tv: ",
      consumer_tv->toString());
  const TMAInfo& tma_info =
      GpuLower::current()->consumerToTMAInfo().at(consumer_tv);
  int64_t dim = (int64_t)tma_info.dims().size();
  Val* expected_bytes = SimplifyingIrBuilder::maybeCastExpr(
      DataType::UInt32, tma_info.tileSizeBytes());
  expected_bytes =
      GpuLower::current()->commonScalarMap().hoistScalar(expected_bytes, loops);
  Val* index = nullptr;

  // 1D TMA without tensor map
  if (ldst->opType() == LoadStoreOpType::CpAsyncBulk) {
    if (is_load) {
      std::stringstream ss;
      ss << "Hopper::CpAsyncBulkG2SIndex";
      auto gmem_address = getProducerIndex(
          producer_tv, consumer_tv, loops, rotated_loops, {}, true);
      index = IrBuilder::structExpr(
          {{"raw_gmem_addr", gmem_address},
           {"bytes", expected_bytes},
           {"mbarrier", mbarrier}},
          ss.str());
    } else {
      std::stringstream ss;
      ss << "Hopper::CpAsyncBulkS2GIndex";
      auto gmem_address =
          getConsumerIndex(consumer_tv, loops, rotated_loops, {}, true);
      index = IrBuilder::structExpr(
          {{"raw_gmem_addr", gmem_address}, {"bytes", expected_bytes}},
          ss.str());
    }
  } else {
    // ND TMA with tensor map
    ValGroups groups_to_index = tma_info.getTMADomain();
    // TensorIndexer needs IterDomain instead of ValGroup to work around
    // the resize indexing issue
    std::vector<IterDomain*> ids_to_index;
    ids_to_index.reserve(groups_to_index.size());
    const auto tma_all_ids = is_load ? consumer_tv->domain()->allIDs()
                                     : producer_tv->domain()->allIDs();
    for (const auto& group : groups_to_index) {
      auto it = std::find_if(
          tma_all_ids.begin(), tma_all_ids.end(), [&](IterDomain* gmem_id) {
            return group->has(gmem_id);
          });
      if (it != tma_all_ids.end()) {
        ids_to_index.push_back(*it);
      } else {
        ids_to_index.push_back(group->front()->as<IterDomain>());
      }
    }

    const TensorIndexer& indexer = GpuLower::current()->tensorIndexer();
    auto indices_inner_to_outer =
        indexer.getIndexFor(ldst, !is_load, ids_to_index, loops);

    // These are the box coordinates of the TMA box, which must be of type
    // int32_t. Possible overflow in each of these dims should be checked
    // elsewhere.
    for (size_t i : arange(indices_inner_to_outer.size())) {
      indices_inner_to_outer[i] =
          IrBuilder::maybeCastExpr(DataType::Int32, indices_inner_to_outer[i]);
    }

    auto coordinate = IrBuilder::arrayExpr(indices_inner_to_outer);
    auto descriptor = tma_info.tensorMap();
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
  }

  index = GpuLower::current()->commonScalarMap().hoistScalar(index, loops);

  auto is_multiple_of_16B = SimplifyingIrBuilder::eqExpr(
      SimplifyingIrBuilder::modExpr(
          expected_bytes, IrBuilder::create<Val>(16, DataType::Index)),
      expected_bytes->fusion()->zeroVal());
  GpuLower::current()->validate(
      is_multiple_of_16B,
      "The expected bytes must be a multiple of 16 bytes, but ",
      expected_bytes->toInlineString(),
      " is not.");

  return {IrBuilder::create<kir::TensorIndex>(gmem_tv, index), expected_bytes};
}

} // namespace nvfuser

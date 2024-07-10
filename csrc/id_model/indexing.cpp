// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <expr_simplifier.h>
#include <id_model/indexing.h>
#include <id_model/to_string.h>
#include <index_compute.h>
#include <ir/builder.h>
#include <ir/graphviz.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>
#include <swizzle.h>
#include <val_graph_visitor.h>

#include <algorithm>
#include <fstream>

namespace nvfuser {

namespace {

// Get the promotion domain of a given loop domain.
IterDomain* getLoopPromotion(IterDomain* loop_id, const IdModel& id_model) {
  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);
  const auto& loop_promotion_map = id_model.loopPromotionMap();
  const auto& loop_group = loop_graph.toGroup(loop_id);

  auto loop_promotion_map_it = loop_promotion_map.find(loop_group);
  NVF_ERROR(
      loop_promotion_map_it != loop_promotion_map.end(),
      "No loop promotion found: ",
      loop_id->toString(),
      ". Loop group: ",
      nvfuser::toString(loop_group));

  return loop_promotion_map_it->second;
}

// True if a given domain is a loop domain of a given tensor and its
// loop is partitioned with respect to the memory type of the tensor
bool isPartitionedLoop(TensorView* tv, IterDomain* id) {
  // False if id is not a loop ID
  if (std::find(tv->getLoopDomain().begin(), tv->getLoopDomain().end(), id) ==
      tv->getLoopDomain().end()) {
    return false;
  }

  // If the memory of this domain is partitioned with respect to the
  // parallel type of the domain, there's no allocation for the domain
  return ir_utils::isMemoryPartitionedAcross(
      tv->getMemoryType(), id->getParallelType());
}

bool isSizeOneDomain(IterDomain* id) {
  return id->isBroadcast() || id->extent()->isOneInt();
}

// True if a given domain of a tensor *may* require allocation
bool mayRequireAllocation(TensorView* tv, IterDomain* id) {
  // Conditions to consider:
  // - Fully partitioned
  // - Size one: Allocation is done based on the promotion ID, but as
  // long as the original ID has size one, its allocation should
  // remain size one.
  // - Reduction: Check the original ID, not the promotion, which may
  //   be a reduction ID even though the original ID is not a reduction
  return !isPartitionedLoop(tv, id) && !isSizeOneDomain(id) &&
      !id->isReduction();
}

// Get the allocation stride of a given allocation domain
Val* getStrideOfGlobalMemoryTensor(TensorView* tv, int64_t alloc_dim) {
  NVF_ERROR(tv->getMemoryType() == MemoryType::Global);

  // Allocation domains can include reduction domains, but
  // alloc_stride arrays do not.
  const auto& alloc_dom = tv->getMaybeAllocationDomain();
  int64_t stride_dim = -1;
  for (const auto i : c10::irange(alloc_dim + 1)) {
    if (alloc_dom.at(i)->isReduction()) {
      continue;
    }
    ++stride_dim;
  }

  NVF_ERROR(stride_dim != -1);

  return IrBuilder::getItemExpr(
      IrBuilder::getAttrExpr(IrBuilder::metadataExpr(tv), "alloc_stride"),
      stride_dim);
}

// Preparing allocation info for indexing. Because of broadcasting,
// just looking at the loop groups of a tensor may not be enough to
// determine the allocation of the tensor. For example, this happens
// when a tensor is broadcast and inlined, where the original
// pre-broadcast tensor may not have corresponding domains. If that
// missing domain is annotated with ParallelType::Unroll, which
// affects all inner loops, just looking at the inlined tensor itself
// would miss the unrolling. Since unrolling changes allocation
// shapes, missing unroll can result in incorrect allocations.
//
// TODO: Refactor this and the allocation lowering pass
class AllocationDomainSetup : private kir::IrVisitor {
 public:
  using IrVisitor::dispatch;

  // Set allocation domain info for all tensors
  void setup(const std::vector<Expr*>& exprs) {
    // Find out correct allocation domains for all consumer
    // tensors. Input tensors are handled after this
    for (auto expr : exprs) {
      dispatch(expr);
    }

    // Make sure all tensors have allocation domains
    for (TensorView* producer_tv : used_as_producer) {
      auto it = tv_alloc_info_map.find(producer_tv);
      if (it != tv_alloc_info_map.end()) {
        continue;
      }

      // Not yet set. This must be an input tensor.
      NVF_ERROR(
          producer_tv->isFusionInput(),
          "Expected a fusion input: ",
          producer_tv->toString());

      // For fusion input, we can just use getMaybeAllocationDomain.

      auto alloc_info = getIndexingAllocationInfo(
          producer_tv,
          producer_tv->getMaybeAllocationDomain(),
          producer_tv->domain()->contiguity());

      tv_alloc_info_map.emplace(producer_tv, alloc_info);
    }
  }

  void dispatch(Expr* expr) override {
    if (ir_utils::isTvOp(expr)) {
      for (auto out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        // Note that since we are dealing with a Kernel IR, a single
        // tensor may show up as consumers multiple times, e.g.,
        // zero initialization and actual definition. Using the last
        // expr should give us correct allocation info. See
        // IndexingTest.InlinedUnroll for a concrete
        // example. Specifically, the initization expression of t2
        // doesn't have an unrolling loop, so the allocation info
        // obtained from that expression would fail to give the
        // correct allocation domains.
        auto [alloc_domains, contiguity] =
            getAllocationDomainsAndContiguity(out_tv, for_loops_);
        auto alloc_info =
            getIndexingAllocationInfo(out_tv, alloc_domains, contiguity);
        tv_alloc_info_map[out_tv] = alloc_info;
      }
      for (auto in_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
        used_as_producer.insert(in_tv);
      }
    } else {
      IrVisitor::dispatch(expr);
    }
  }

  // Get the allocation domains and contiguity of a given tensor
  //
  // TODO: Ideally, all tensors should have their correct allocation
  // domains, but that isn't always the case at this moment. The logic
  // here is duplicated in multiple locations and should be cleaned up.
  std::pair<std::vector<IterDomain*>, std::vector<std::optional<bool>>>
  getAllocationDomainsAndContiguity(
      TensorView* tv,
      const std::vector<ForLoop*>& for_loops) {
    std::vector<IterDomain*> allocation_domains;
    std::vector<std::optional<bool>> contiguity;

    // In general, if the tensor has an allocation domain set, it
    // should be used with no change. However, set allocation domains
    // are not always right allocation domains. For example,
    // AliasTest.NotAllOutputAlias_Reduction has a tensor, tv6, that
    // is a Local tensor with CA position of 4 but has an allocation
    // domain that's just a permutation of its logical domain. Such
    // invalid allocations need to be ignored. If there doesn't seem
    // to be any clear condition when the set domain can be used, so
    // it needs to be inferred. Here's what seems to be working
    // reasonably well.
    bool use_set_allocation_domain = false;
    if (tv->hasAllocation()) {
      // Honor the allocation domain if the tensor is global memory
      if (tv->getMemoryType() == MemoryType::Global) {
        use_set_allocation_domain = true;
      } else if (tv->getMemoryType() == MemoryType::Shared) {
        // If it's a shared memory tensor, the set domain is likely
        // valid if Swizzle or Bulk is used. Also, if the allocation
        // domain is just a permutation of the loop domain, use the
        // set allocation domain. This seems to happen only with
        // AllocationDomainTest.TransposedIntermediate.
        if (std::any_of(
                tv->getAllocationDomain().begin(),
                tv->getAllocationDomain().end(),
                [](IterDomain* allocation_domain) {
                  return dynamic_cast<Swizzle*>(
                             allocation_domain->definition()) != nullptr ||
                      allocation_domain->getParallelType() ==
                      ParallelType::Bulk;
                }) ||
            std::is_permutation(
                tv->getLoopDomain().begin(),
                tv->getLoopDomain().end(),
                tv->getAllocationDomain().begin())) {
          use_set_allocation_domain = true;
        }
      }
    }

    if (use_set_allocation_domain) {
      allocation_domains = tv->getAllocationDomain();
      contiguity = tv->domain()->contiguity();
    } else {
      // If allocation domain is not set, assume that:
      // - Global: logical domains
      // - Local/Shared: loop domains to the right of the CA position
      if (tv->getMemoryType() == MemoryType::Global) {
        allocation_domains = tv->getLogicalDomain();
        contiguity = tv->domain()->contiguity();
      } else {
        // Allocation position is not always the same as the CA
        // position. See also lower_utils::getAllocInformation.
        int64_t allocation_pos =
            lower_utils::getAllocInformation(tv, for_loops).alloc_pos;
        for (const auto i : c10::irange(tv->nDims())) {
          auto loop_id = tv->getLoopDomain().at(i);
          auto pt = loop_id->getParallelType();
          if (!mayRequireAllocation(tv, loop_id)) {
            continue;
          }

          // If the position is left of the inlining position, no need to
          // allocate the domain unless it's shared. For example, if this
          // is a Shared tensor and the domain is parallelized with TID,
          // even if it's outside of the CA position, since the domain
          // is shared, it must be allocated.
          if (i < allocation_pos &&
              !ir_utils::isMemorySharedAcross(tv->getMemoryType(), pt)) {
            continue;
          }

          allocation_domains.push_back(loop_id);
        }
        // Assume Local and Shared are always fully contiguous
        contiguity =
            std::vector<std::optional<bool>>(allocation_domains.size(), true);
      }

      if (auto reordered_domains =
              reorderAllocationDomains(tv, allocation_domains);
          reordered_domains.has_value()) {
        allocation_domains = reordered_domains.value();
        NVF_ERROR(
            std::all_of(
                contiguity.begin(),
                contiguity.end(),
                [](auto b) { return b.has_value() && b.value(); }),
            tv->toString());
      }
    }

    return {allocation_domains, contiguity};
  }

  // Get allocation info used for indexing. Loop promotion is
  // considered. Strides are also calculated.
  IndexingAllocationInfo getIndexingAllocationInfo(
      TensorView* tv,
      std::vector<IterDomain*> allocation_domains,
      std::vector<std::optional<bool>> contiguity) {
    const IdModel& id_model = GpuLower::current()->idModel();

    std::vector<IterDomain*> promoted_allocation_domains;
    promoted_allocation_domains.reserve(allocation_domains.size());

    // Loop promotion may affect allocations. Promotions of intermediate
    // domains may not be defined correctly. Only consider loop domains
    // for now.
    for (const auto& allocation_domain : allocation_domains) {
      bool is_loop = std::find(
                         tv->getLoopDomain().begin(),
                         tv->getLoopDomain().end(),
                         allocation_domain) != tv->getLoopDomain().end();
      IterDomain* promotion_domain = nullptr;
      if (is_loop) {
        promotion_domain = getLoopPromotion(allocation_domain, id_model);
      } else {
        promotion_domain = allocation_domain;
      }
      promoted_allocation_domains.push_back(promotion_domain);
    }

    // Compute the strides from innermost to outermost domains
    std::vector<Val*> strides(allocation_domains.size(), nullptr);
    Val* cur_contig_stride = tv->fusion()->oneVal();
    for (const auto i : c10::irange(allocation_domains.size())) {
      auto dim = allocation_domains.size() - i - 1;
      auto allocation_domain = allocation_domains.at(dim);
      auto promotion_domain = promoted_allocation_domains.at(dim);

      if (!mayRequireAllocation(tv, allocation_domain)) {
        continue;
      }

      const std::optional<bool> contig_flag = contiguity.at(dim);
      // Broadcast doesn't have contig flag but it must have been
      // already filtered out
      NVF_ERROR(contig_flag.has_value());

      if (contig_flag.value()) {
        strides[dim] = cur_contig_stride;
        cur_contig_stride = SimplifyingIrBuilder::mulExpr(
            cur_contig_stride, promotion_domain->extent());
      } else {
        // Assume that the tensor should always be a Global memory
        // tensor if it has non-contig allocation domains
        NVF_ERROR(tv->getMemoryType() == MemoryType::Global);
        strides[dim] = getStrideOfGlobalMemoryTensor(tv, (int64_t)dim);
        cur_contig_stride = SimplifyingIrBuilder::mulExpr(
            strides[dim], promotion_domain->extent());
      }
    }

    // Filter out non-allocated domains. This is already done for Local
    // and Shared tensors with no set allocation domains, but not for
    // the other cases. For example, a reduction output tensor that is
    // also a fusion output may still have reduction domains in their
    // allocation domains, which aren't relevant for indexing
    std::vector<IterDomain*> actual_allocation_domains;
    std::vector<Val*> actual_strides;
    for (const auto i : c10::irange(allocation_domains.size())) {
      auto allocation_domain = allocation_domains.at(i);
      auto promotion_domain = promoted_allocation_domains.at(i);
      if (!mayRequireAllocation(tv, allocation_domain)) {
        continue;
      }
      auto stride = strides.at(i);
      NVF_ERROR(stride != nullptr);
      actual_allocation_domains.push_back(promotion_domain);
      actual_strides.push_back(stride);
    }

    return IndexingAllocationInfo{actual_allocation_domains, actual_strides};
  }

  // Reorder non-logical allocation domains to follow the ordering of
  // the logical domain. This is necessary when an allocation domain
  // includes a vectorized loop iter domain since it must be at the
  // innermost position but that may not be the case in the loop
  // domain. Not strictly necessary otherwise, but this should also
  // minimize the deviation from the old indexing scheme which always
  // uses the logical domain to index.
  //
  // Returns reordered allocation domains if reordering is done.
  std::optional<std::vector<IterDomain*>> reorderAllocationDomains(
      const TensorView* tv,
      const std::vector<IterDomain*>& allocation_domains) const {
    auto exprs = DependencyCheck::getAllExprsBetween(
        {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()},
        {allocation_domains.begin(), allocation_domains.end()});

    if (exprs.empty()) {
      return std::nullopt;
    }

    // Replay exprs from the logical domain to get the non-reordered
    // domains
    auto ordered_domains = tv->getLogicalDomain();
    for (auto expr : exprs) {
      // Find the position to insert the outputs.
      int64_t insertion_pos = -1;
      for (auto inp : expr->inputs()) {
        auto it =
            std::find(ordered_domains.begin(), ordered_domains.end(), inp);
        if (it == ordered_domains.end()) {
          continue;
        }
        // Insert right after the input
        int64_t pos = std::distance(ordered_domains.begin(), it) + 1;
        if (insertion_pos == -1 || pos > insertion_pos) {
          insertion_pos = pos;
        }
      }
      NVF_ERROR(
          insertion_pos >= 0,
          "Failed to replay: ",
          expr->toString(),
          " in ",
          tv->toString());
      // Insert the outputs
      for (auto out : expr->outputs()) {
        ordered_domains.insert(
            ordered_domains.begin() + insertion_pos, out->as<IterDomain>());
        ++insertion_pos;
      }
      // Delete the inputs
      for (auto inp : expr->inputs()) {
        auto it =
            std::find(ordered_domains.begin(), ordered_domains.end(), inp);
        if (it == ordered_domains.end()) {
          continue;
        }
        ordered_domains.erase(it);
      }
    }

    // At this point, all domains of allocation_domains must exist in
    // domains.
    for (auto alloc_dom : allocation_domains) {
      auto it =
          std::find(ordered_domains.begin(), ordered_domains.end(), alloc_dom);
      NVF_ERROR(
          it != ordered_domains.end(),
          "Missing allocation domain: ",
          alloc_dom->toString(),
          ", domains: ",
          toDelimitedString(ordered_domains));
    }

    // Pick only the allocation domains from the ordered domains
    std::vector<IterDomain*> reordered_allocation_domains;
    reordered_allocation_domains.reserve(allocation_domains.size());

    for (auto dom : ordered_domains) {
      auto it =
          std::find(allocation_domains.begin(), allocation_domains.end(), dom);
      if (it == allocation_domains.end()) {
        continue;
      }
      reordered_allocation_domains.push_back(dom);
    }

    // If it's the same order, just return nullopt to tell nothing
    // needs to be reordered
    if (reordered_allocation_domains == allocation_domains) {
      return std::nullopt;
    }

    return reordered_allocation_domains;
  }

  std::unordered_map<TensorView*, IndexingAllocationInfo> tv_alloc_info_map;
  std::unordered_set<TensorView*> used_as_producer;
};

// Similar to IndexCompute but adapted for the graph-based indexing
class IdGraphIndexCompute : public OptOutDispatch {
 public:
  IdGraphIndexCompute(
      const ValGraph& traversal_graph,
      std::unordered_map<ValGroup, Val*> initial_index_map)
      : traversal_graph_(traversal_graph),
        index_map_(std::move(initial_index_map)) {}

  // Propagate the index map through a given expr of a specified
  // direction.
  void propagate(const ExprGroup& expr_group, Direction direction) {
    NVF_ERROR(!expr_group->empty());
    // This looks a little ugly but the dispatch interface doesn't
    // have a way to pass arguments
    current_direction_ = direction;
    dispatch(expr_group->front());
    current_direction_ = Direction::Undefined;
  }

  const std::unordered_map<ValGroup, Val*> indexMap() const {
    return index_map_;
  }

 private:
  using OptOutDispatch::handle;

  void handle(Split* split) override;

  void handle(Merge* merge) override;

  void handle(Swizzle* swizzle) override;

  bool isForward(Expr* expr) const;

  bool hasIndex(IterDomain* id) const {
    return indexMap().find(toGroup(id)) != indexMap().end();
  }

  Val* getIndex(IterDomain* id) const {
    auto it = index_map_.find(toGroup(id));
    NVF_ERROR(it != index_map_.end(), "Index not found: ", id->toString());
    return it->second;
  }

  void setIndex(IterDomain* id, Val* idx) {
    index_map_.emplace(toGroup(id), idx);
  }

  const ValGroup& toGroup(IterDomain* id) const {
    return traversal_graph_.toGroup(id);
  }

 private:
  const ValGraph& traversal_graph_;
  std::unordered_map<ValGroup, Val*> index_map_;
  Direction current_direction_ = Direction::Undefined;
};

bool IdGraphIndexCompute::isForward(Expr* expr) const {
  return current_direction_ == Direction::Forward;
}

void IdGraphIndexCompute::handle(Split* split) {
  const bool is_forward = isForward(split);

  auto inner_extent = split->inner()->extent();

  if (is_forward) {
    auto in_idx = getIndex(split->in());
    auto outer_idx = SimplifyingIrBuilder::divExpr(in_idx, inner_extent);
    Val* inner_idx = SimplifyingIrBuilder::modExpr(in_idx, inner_extent);
    setIndex(split->outer(), outer_idx);
    setIndex(split->inner(), inner_idx);
  } else {
    auto outer_idx = getIndex(split->outer());
    auto inner_idx = getIndex(split->inner());
    auto in_idx = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(outer_idx, inner_extent), inner_idx);
    setIndex(split->in(), in_idx);
  }
}

void IdGraphIndexCompute::handle(Merge* merge) {
  const bool is_forward = isForward(merge);

  auto inner_ext = merge->inner()->extent();

  if (is_forward) {
    auto outer_idx = getIndex(merge->outer());
    auto inner_idx = getIndex(merge->inner());
    auto out_idx = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(inner_ext, outer_idx), inner_idx);
    setIndex(merge->out(), out_idx);
  } else {
    auto out_idx = getIndex(merge->out());
    auto outer_idx = SimplifyingIrBuilder::divExpr(out_idx, inner_ext);
    setIndex(merge->outer(), outer_idx);
    Val* inner_idx = SimplifyingIrBuilder::modExpr(out_idx, inner_ext);
    setIndex(merge->inner(), inner_idx);
  }
}

void IdGraphIndexCompute::handle(Swizzle* swizzle) {
  const bool is_forward = isForward(swizzle);

  auto x_ext = swizzle->inX()->extent();
  auto y_ext = swizzle->inY()->extent();

  if (is_forward) {
    auto x_idx = getIndex(swizzle->inX());
    auto y_idx = getIndex(swizzle->inY());
    auto [result_x, result_y] =
        dispatchUnSwizzle(swizzle->swizzleType(), x_idx, y_idx, x_ext, y_ext);
    setIndex(swizzle->outX(), result_x);
    setIndex(swizzle->outY(), result_y);
  } else {
    auto x_idx = getIndex(swizzle->outX());
    auto y_idx = getIndex(swizzle->outY());
    auto [result_x, result_y] =
        dispatchSwizzle(swizzle->swizzleType(), x_idx, y_idx, x_ext, y_ext);
    setIndex(swizzle->inX(), result_x);
    setIndex(swizzle->inY(), result_y);
  }
}

} // namespace

TensorIndexer::TensorIndexer(IdModel& id_model) : id_model_(id_model) {
  buildLoopIndexMap();
}

namespace {
ParallelType getParallelType(const ValGroup& loop_group) {
  ParallelType common_pt = ParallelType::Serial;
  for (const auto val : *loop_group) {
    auto pt = val->as<IterDomain>()->getParallelType();
    if (common_pt == pt || pt == ParallelType::Serial) {
      continue;
    } else if (common_pt == ParallelType::Serial) {
      common_pt = pt;
    } else {
      // Inconsistent parallelization
      NVF_ERROR(
          false,
          "Inconsistent parallelization detected. ",
          "Known type: ",
          common_pt,
          "New type: ",
          pt);
    }
  }

  return common_pt;
}
} // namespace

void TensorIndexer::buildLoopIndexMap() {
  if (id_model_.empty()) {
    return;
  }

  Fusion* fusion = id_model_.fusion();

  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }
    // It's assumed that all sibling outputs share the same for-loops,
    // thus only one of the outputs is considered.
    auto tv_output = ir_utils::getTvOutput(expr);
    for (auto loop_id : tv_output->getLoopDomain()) {
      const ValGroup& loop_group =
          id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_id);

      if (loop_index_map_.find(loop_group) != loop_index_map_.end()) {
        // Index already assigned
        continue;
      }

      Val* loop_index = nullptr;

      ParallelType ptype = getParallelType(loop_group);
      if (isParallelTypeThread(ptype)) {
        loop_index = NamedScalar::getParallelIndex(ptype);
      } else if (
          // TODO: Cleanup needed. ir_utils::isMemoryPartitionedAcross
          // should be used, but that means we would need to consider
          // multiple outputs with different memory types, though it
          // should be uncommon in practice.
          shouldUseZeroIndex(loop_group) || isParallelTypeDeviceDim(ptype)) {
        loop_index = fusion->zeroVal();
      } else {
        // Until the transition to the IdModel-based indexing is
        // completed, use the index Vals assigned for ComputeAtMap
        // groups if available.
        if (GpuLower::hasCurrent()) {
          const auto& ca_map = GpuLower::current()->caMap();
          for (const auto& id :
               ir_utils::filterByType<IterDomain>(loop_group->vector())) {
            if (!ca_map->getIdSets(IdMappingMode::LOOP).mappingExists(id)) {
              continue;
            }
            loop_index = ca_map->getIndexVariable(id);
            break;
          }
          NVF_ERROR(
              loop_index != nullptr,
              "No existing index found for ",
              nvfuser::toString(loop_group));
        } else {
          loop_index = IrBuilder::create<Val>(DataType::Index);
        }
      }

      loop_index_map_[loop_group] = loop_index;
    }
  }
}

bool TensorIndexer::shouldUseZeroIndex(const ValGroup& loop_group) const {
  // Trivial loop
  auto promotion_id =
      getLoopPromotion(loop_group->front()->as<IterDomain>(), id_model_);
  if (promotion_id->isBroadcast() ||
      simplifyExpr(promotion_id->extent())->isOneInt()) {
    return true;
  }

  return false;
}

Val* TensorIndexer::getLoopIndex(IterDomain* loop_id) const {
  // loop_id must be a loop domain.
  const auto& loop_group =
      id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_id);
  auto loop_index_map_it = loop_index_map_.find(loop_group);
  NVF_ERROR(
      loop_index_map_it != loop_index_map_.end(),
      "No loop index found for ",
      loop_id->toString());

  Val* loop_index = loop_index_map_it->second;
  return loop_index;
}

std::unordered_map<ValGroup, Val*> TensorIndexer::getInitialIndexMap(
    const std::vector<IterDomain*>& loop_domains) const {
  std::unordered_map<ValGroup, Val*> initial_index_map;

  // For a given list of the loop domains, assign its corresponding
  // index Val.
  for (IterDomain* loop_id : loop_domains) {
    Val* loop_index = getLoopIndex(loop_id);
    const auto& almost_exact_group = traversalGraph().toGroup(loop_id);

    if (initial_index_map.find(almost_exact_group) != initial_index_map.end()) {
      // Initial index already set. This can happen as this is an
      // almost exact group. It should be just size-1 domain.
      NVF_ERROR(
          loop_index->isZeroInt(),
          "Unexpected initial index: ",
          loop_index->toInlineString());
      auto existing_index = initial_index_map.at(almost_exact_group);
      NVF_ERROR(
          existing_index->isZeroInt(),
          "Unexpected initial index: ",
          existing_index->toInlineString());
      continue;
    }

    initial_index_map.emplace(almost_exact_group, loop_index);
  }

  return initial_index_map;
}

std::vector<Val*> TensorIndexer::getIndexFor(
    const Expr* expr,
    const ValGroups& index_groups) const {
  auto info = computeIndex(expr, index_groups);
  const auto& replacement_map =
      getIndexReplacementMap(info.loop_domains, info.index_map);

  std::vector<Val*> result;
  result.reserve(index_groups.size());
  for (const auto& g : index_groups) {
    auto it = info.index_map.find(g);
    NVF_ERROR(
        it != info.index_map.end(), "Index not found for ", g->toString());
    result.push_back(
        ir_utils::replaceValRecursively(it->second, replacement_map));
  }
  return result;
}

Val* TensorIndexer::getLinearIndex(TensorView* tv, const Expr* expr) const {
  NVF_ERROR(tv != nullptr);
  NVF_ERROR(expr != nullptr);
  NVF_ERROR(
      (std::find(expr->inputs().begin(), expr->inputs().end(), tv) !=
       expr->inputs().end()) ||
          (std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
           expr->outputs().end()),
      "Inconsistent tensor and expr. Tensor, ",
      tv->toString(),
      " not found in ",
      expr->toString());

  const auto alloc_info = getIndexingAllocationInfo(tv);

  auto indices =
      getIndexFor(expr, traversalGraph().toGroups(alloc_info.domains));
  NVF_ERROR(indices.size() == alloc_info.domains.size());

  // Linearize the indices with strides.
  // TODO: Contiguous indexing
  Val* index = tv->fusion()->zeroVal();
  for (const auto i : c10::irange(alloc_info.domains.size())) {
    Val* stride = alloc_info.strides.at(i);
    index = SimplifyingIrBuilder::addExpr(
        index, SimplifyingIrBuilder::mulExpr(stride, indices.at(i)));
  }

  return index;
}

// Get the loop domains of a given expr, which are (potentially
// promoted) loop domains of the consumer tensor.
std::vector<IterDomain*> TensorIndexer::getLoopDomains(const Expr* expr) const {
  // Assume consumer-based indexing. Needs to revisit for ops like
  // scatter
  auto loop_domains = ir_utils::getTvOutput(expr)->getLoopDomain();

  for (auto& loop_id : loop_domains) {
    loop_id = getLoopPromotion(loop_id, id_model_);
  }

  return loop_domains;
}

IndexingInfo TensorIndexer::computeIndex(
    const Expr* expr,
    const ValGroups& index_groups) const {
  const auto loop_domains = getLoopDomains(expr);

  const ValGroups loop_groups = traversalGraph().toGroups(loop_domains);
  const ExprPath traversal_path =
      ValGraphBFS::getExprsBetween(traversalGraph(), loop_groups, index_groups);

  const std::unordered_map<ValGroup, Val*> initial_index_map =
      getInitialIndexMap(loop_domains);

  IdGraphIndexCompute index_compute(traversalGraph(), initial_index_map);

  for (const auto& [expr_group, direction] : traversal_path) {
    index_compute.propagate(expr_group, direction);
  }

  IndexingInfo info{loop_domains, traversal_path, index_compute.indexMap()};
  return info;
}

std::unordered_map<Val*, Val*> TensorIndexer::getIndexReplacementMap(
    const std::vector<IterDomain*>& loop_domains,
    const std::unordered_map<ValGroup, Val*>& index_map) const {
  std::unordered_map<Val*, Val*> replacement_map;

  for (const auto loop_id : loop_domains) {
    // Replace the index of a vectorized/bulk domain with zero. Note that
    // vectorized domains may need to use N-1, where N is the extent
    // of the domain, for predication, so the replacement is not
    // always done with zero.
    if (loop_id->getParallelType() != ParallelType::Vectorize &&
        loop_id->getParallelType() != ParallelType::Bulk) {
      continue;
    }
    const ValGroup& loop_group = traversalGraph().toGroup(loop_id);
    auto index_it = index_map.find(loop_group);
    NVF_ERROR(index_it != index_map.end());
    Val* cur_index = index_it->second;
    replacement_map.emplace(cur_index, cur_index->fusion()->zeroVal());
  }

  return replacement_map;
}

void TensorIndexer::setupAllocationDomains(const std::vector<Expr*>& exprs) {
  AllocationDomainSetup alloc_setup;
  alloc_setup.setup(exprs);
  alloc_info_ = std::move(alloc_setup.tv_alloc_info_map);
}

} // namespace nvfuser

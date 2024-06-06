// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
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

// True if a given domain is a loop doamin of a given tensor and its
// loop is partitioned with respect to the memory type of the tensor
bool isPartitionedLoop(TensorView* tv, IterDomain* id) {
  // False if id is not a loop ID
  if (std::find(tv->getLeafDomain().begin(), tv->getLeafDomain().end(), id) ==
      tv->getLeafDomain().end()) {
    return false;
  }

  // If the memory of this domain is partitioned with respect to the
  // parallel type of the domain, there's no allocation for the domain
  return ir_utils::isMemoryPartitionedAcross(
      tv->getMemoryType(), id->getParallelType());
}

bool isSizeOneDomain(IterDomain* id) {
  return id->isBroadcast() || id->isReduction() || id->extent()->isOneInt();
}

// True if a given domain of a tensor *may* require allocation
bool mayRequireAllocation(TensorView* tv, IterDomain* id) {
  return !isPartitionedLoop(tv, id) && !isSizeOneDomain(id);
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

// Get the allocation domains of a given tensor. Also returns its
// strides.
//
// TODO: Ideally, all tensors should have their correct allocation
// domains, but that isn't always the case at this moment. The logic
// here is duplicated in multiple locations and should be cleaned up.
std::tuple<std::vector<IterDomain*>, std::vector<Val*>> getAllocationDomains(
    TensorView* tv,
    const IdModel& id_model) {
  std::vector<IterDomain*> allocation_domains;
  std::vector<std::optional<bool>> contiguity;

  // Use the allocation domain if set for the tensor
  if (tv->hasAllocation()) {
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
      const auto inlining_pos = tv->getComputeAtPosition();
      for (const auto i : c10::irange(tv->nDims())) {
        auto loop_id = tv->getLeafDomain().at(i);
        auto pt = loop_id->getParallelType();
        if (!mayRequireAllocation(tv, loop_id)) {
          continue;
        }

        // If the position is left of the inlining position, no need to
        // allocate the domain unless it's shared. For example, if this
        // is a Shared tensor and the domain is parallelized with TID,
        // even if it's outside of the CA position, since the domain
        // is shared, it must be allocated.
        if (i < inlining_pos &&
            !ir_utils::isMemorySharedAcross(tv->getMemoryType(), pt)) {
          continue;
        }

        allocation_domains.push_back(loop_id);
      }
      // Assume Local and Shared are always fully contiguous
      contiguity =
          std::vector<std::optional<bool>>(allocation_domains.size(), true);
    }
  }

  // Loop promotion may affect allocations. Promotions of intermediate
  // domains may not be defined correctly. Only consider loop domains
  // for now.
  for (auto& allocation_domain : allocation_domains) {
    bool is_loop = std::find(
                       tv->getLeafDomain().begin(),
                       tv->getLeafDomain().end(),
                       allocation_domain) != tv->getLeafDomain().end();
    if (!is_loop) {
      continue;
    }
    allocation_domain = getLoopPromotion(allocation_domain, id_model);
  }

  // Compute the strides from innermost to outermost domains
  std::vector<Val*> strides(allocation_domains.size(), nullptr);
  Val* cur_contig_stride = tv->fusion()->oneVal();
  for (const auto i : c10::irange(allocation_domains.size())) {
    auto dim = allocation_domains.size() - i - 1;
    auto allocation_domain = allocation_domains.at(dim);

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
          allocation_domains.at(dim)->extent(), cur_contig_stride);
    } else {
      // Assume that the tensor should always be a Global memory
      // tensor if it has non-contig allocation domains
      NVF_ERROR(tv->getMemoryType() == MemoryType::Global);
      strides[dim] = getStrideOfGlobalMemoryTensor(tv, (int64_t)dim);
      cur_contig_stride = strides[dim];
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
    if (!mayRequireAllocation(tv, allocation_domain)) {
      continue;
    }
    auto stride = strides.at(i);
    NVF_ERROR(stride != nullptr);
    actual_allocation_domains.push_back(allocation_domain);
    actual_strides.push_back(stride);
  }

  return {actual_allocation_domains, actual_strides};
}

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
        SimplifyingIrBuilder::mulExpr(outer_idx, inner_ext), inner_idx);
    setIndex(merge->out(), out_idx);
  } else {
    auto out_idx = getIndex(merge->out());
    auto outer_idx = SimplifyingIrBuilder::divExpr(out_idx, inner_ext);
    setIndex(merge->outer(), outer_idx);
    Val* inner_idx = SimplifyingIrBuilder::modExpr(out_idx, inner_ext);
    setIndex(merge->inner(), inner_idx);
  }
}

} // namespace

TensorIndexer::TensorIndexer(const IdModel& id_model) : id_model_(id_model) {
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
    for (auto leaf_id : tv_output->getLeafDomain()) {
      const ValGroup& loop_group =
          id_model_.idGraph(IdMappingMode::LOOP).toGroup(leaf_id);

      if (loop_index_map_.find(loop_group) != loop_index_map_.end()) {
        // Index already assigned
        continue;
      }

      Val* loop_index = nullptr;

      ParallelType ptype = getParallelType(loop_group);
      if (isParallelTypeThread(ptype)) {
        loop_index = NamedScalar::getParallelIndex(ptype);
      } else if (shouldUseZeroIndex(loop_group)) {
        loop_index = fusion->zeroVal();
      } else {
        loop_index = IrBuilder::create<Val>(DataType::Index);
      }

      loop_index_map_[loop_group] = loop_index;
    }
  }
}

bool TensorIndexer::shouldUseZeroIndex(const ValGroup& loop_group) const {
  // All loops in this set are non-parallel, non-concretized broadcast
  //  iterdomains, their "index variable" should be zero. This
  //  condition should be included in the next triviality check, but
  //  just checking isBroadcast should be more efficient.
  if (std::all_of(loop_group->begin(), loop_group->end(), [](Val* val) {
        return val->as<IterDomain>()->isBroadcast();
      })) {
    return true;
  }

  // Trivial loop
  auto leaf_id =
      getLoopPromotion(loop_group->front()->as<IterDomain>(), id_model_);
  if (!leaf_id->maybePartial() && simplifyExpr(leaf_id->extent())->isOneInt()) {
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

Val* TensorIndexer::getLinearIndex(TensorView* tv, const Expr* expr) {
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

  const auto [allocation_domains, strides] =
      getAllocationDomains(tv, id_model_);

  const auto& index_info = computeIndex(expr, allocation_domains);
  const auto& index_map = index_info.index_map;

  // Linearize the indices with strides.
  // TODO: Contiguous indexing
  Val* index = tv->fusion()->zeroVal();
  for (const auto i : c10::irange(allocation_domains.size())) {
    // Traverse from innermost to outermost
    IterDomain* allocation_domain =
        allocation_domains.at(allocation_domains.size() - 1 - i);

    Val* stride = strides.at(allocation_domains.size() - 1 - i);

    auto idx_it = index_map.find(traversalGraph().toGroup(allocation_domain));
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        allocation_domain->toString());
    Val* idx = idx_it->second;
    index = SimplifyingIrBuilder::addExpr(
        index, SimplifyingIrBuilder::mulExpr(idx, stride));
  }

  return index;
}

// Get the loop domains of a given expr, which are (potentially
// promoted) loop domains of the consumer tensor.
std::vector<IterDomain*> TensorIndexer::getLoopDomains(const Expr* expr) const {
  // Assume consumer-based indexing. Needs to revisit for ops like
  // scatter
  auto loop_domains = ir_utils::getTvOutput(expr)->getLeafDomain();

  for (auto& loop_id : loop_domains) {
    loop_id = getLoopPromotion(loop_id, id_model_);
  }

  return loop_domains;
}

IndexingInfo TensorIndexer::computeIndex(
    const Expr* expr,
    const std::vector<IterDomain*>& index_domains) const {
  const auto loop_domains = getLoopDomains(expr);

  const ValGroups loop_groups = traversalGraph().toGroups(loop_domains);
  const ValGroups index_groups = traversalGraph().toGroups(index_domains);
  const ExprPath traversal_path =
      ValGraphBFS::getExprsBetween(traversalGraph(), loop_groups, index_groups);

  const std::unordered_map<ValGroup, Val*> initial_index_map =
      getInitialIndexMap(loop_domains);

  IdGraphIndexCompute index_compute(traversalGraph(), initial_index_map);

  for (const auto& [expr_group, direction] : traversal_path) {
    index_compute.propagate(expr_group, direction);
  }

  IndexingInfo info{traversal_path, index_compute.indexMap()};
  return info;
}

} // namespace nvfuser

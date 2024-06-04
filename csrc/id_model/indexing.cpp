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
#include <id_model/utils.h>
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

// Get the loop domains of a given expr, which are (potentially
// promoted) loop domains of the consumer tensor.
std::vector<IterDomain*> getLoopDomains(
    const Expr* expr,
    const IdModel& id_model) {
  // Assume consumer-based indexing. Needs to revisit for ops like
  // scatter
  auto loop_domains = ir_utils::getTvOutput(expr)->getLeafDomain();

  // If this is an expr initializing a buffer for a reduction, there
  // should be no loops for reduction domains
  if (lower_utils::isReductionInitExpr(expr)) {
    loop_domains.erase(
        std::remove_if(
            loop_domains.begin(),
            loop_domains.end(),
            [](IterDomain* id) -> bool { return id->isReduction(); }),
        loop_domains.end());
  }

  for (auto& loop_id : loop_domains) {
    loop_id = getLoopPromotion(loop_id, id_model);
  }

  return loop_domains;
}

bool isAllocated(IterDomain* id, const TensorView* tv) {
  // If the extent is 1, it's effectively the same as broadcast.
  return ir_utils::isShared(tv->getMemoryType(), id->getParallelType()) &&
      !id->isBroadcast() && !id->isReduction() && !id->extent()->isOneInt();
}

Val* getAllocationStride(TensorView* tv, int64_t alloc_dim) {
  const auto& alloc_dom = tv->getMaybeAllocationDomain();
  int64_t stride_dim = -1;
  for (const auto i : c10::irange(alloc_dim + 1)) {
    if (alloc_dom.at(i)->isReduction()) {
      continue;
    }
    ++stride_dim;
  }
  if (stride_dim == -1) {
    return nullptr;
  }

  return IrBuilder::getItemExpr(
      IrBuilder::getAttrExpr(IrBuilder::metadataExpr(tv), "alloc_stride"),
      stride_dim);
}

// Currently it's only Shared or Local but Global can be the case
// too.
bool isAllocationBasedOnLeaf(TensorView* tv) {
  return tv->getMemoryType() == MemoryType::Shared ||
      tv->getMemoryType() == MemoryType::Local;
}

// Get the allocation domains of a given tensor. Also returns its
// strides.
std::tuple<std::vector<IterDomain*>, std::vector<Val*>> getAllocationDomains(
    TensorView* tv,
    const IdModel& id_model) {
  std::vector<IterDomain*> allocation_domains;

  auto inlining_pos = tv->getComputeAtPosition();

  bool use_set_allocatin_domain = false;

  if (tv->hasAllocation()) {
    if (tv->getMemoryType() == MemoryType::Shared ||
        tv->getMemoryType() == MemoryType::Local) {
      if (std::is_permutation(
              tv->getLeafDomain().begin(),
              tv->getLeafDomain().end(),
              tv->getAllocationDomain().begin())) {
        use_set_allocatin_domain = true;
      }
    } else {
      use_set_allocatin_domain = true;
    }
  }

  // Ignore allocation of non-global tensors for now
  if (use_set_allocatin_domain) {
    allocation_domains = tv->getAllocationDomain();
    NVF_ERROR(!tv->isDoubleBuffered());
  } else {
    // If allocation domain is not set, assume that:
    // Local/Shared: leaf domains to the right of the CA position
    // Global: rfactor domains
    if (tv->getMemoryType() == MemoryType::Global) {
      VERBOSE() << "Tv does not have allocation of " << tv->toString() << ", "
                << toDelimitedString(tv->getMaybeAllocationDomain())
                << std::endl;
      allocation_domains = tv->getRFactorDomain();
      NVF_ERROR(!tv->isDoubleBuffered());
    } else if (tv->getMemoryType() == MemoryType::Shared) {
      for (const auto i : c10::irange(tv->nDims())) {
        auto leaf_id = tv->axis(i);
        VERBOSE() << "Smem leaf domain: " << leaf_id->toString() << " of "
                  << tv->toString() << std::endl;
        auto pt = leaf_id->getParallelType();
        if (isParallelTypeDeviceDim(pt) || isParallelTypeBlockDim(pt)) {
          continue;
        }
        if (i < inlining_pos && !isParallelTypeThreadDim(pt)) {
          continue;
        }
        allocation_domains.push_back(leaf_id);
      }
    } else {
      allocation_domains = {
          tv->getLeafDomain().begin() + inlining_pos,
          tv->getLeafDomain().end()};
    }
  }

  auto tv_for_promotion = tv;

  std::vector<Val*> strides(allocation_domains.size(), nullptr);
  for (const auto i : c10::irange(allocation_domains.size())) {
    auto dim = allocation_domains.size() - i - 1;
    auto index_domain = allocation_domains.at(dim);

    if (index_domain->isReduction() || index_domain->isBroadcast()) {
      continue;
    }

    strides[dim] = getAllocationStride(tv, (int64_t)dim);
  }

  std::vector<IterDomain*> actual_index_domains;
  std::vector<Val*> actual_strides;
  std::vector<bool> actual_contiguity;
  for (const auto i : c10::irange(allocation_domains.size())) {
    auto index_domain = allocation_domains.at(i);
    if (!isAllocated(index_domain, tv)) {
      continue;
    }

    // If it's a leaf domain, the promoted domain is the true domain
    // for allocation and indexing.
    bool is_leaf = std::find(
                       tv_for_promotion->getLeafDomain().begin(),
                       tv_for_promotion->getLeafDomain().end(),
                       index_domain) != tv->getLeafDomain().end();
    auto actual_id =
        is_leaf ? getLoopPromotion(index_domain, id_model) : index_domain;
    VERBOSE() << "Index domain: " << index_domain->toString()
              << ", actual domain (promotion domain): " << actual_id->toString()
              << std::endl;

    actual_index_domains.push_back(actual_id);
    actual_strides.push_back(strides.at(i));
    NVF_ERROR(
        actual_strides.back() != nullptr,
        "Stride unknown for ",
        index_domain->toString(),
        " (promoted to ",
        actual_id->toString(),
        ")");
  }

  NVF_ERROR(actual_index_domains.size() == actual_strides.size());

  return {actual_index_domains, actual_strides};
}

// Similar to IndexCompute but adapted for the graph-based indexing
class IdGraphIndexCompute : public OptOutDispatch {
 public:
  IdGraphIndexCompute(
      const ValGraph& traversal_graph,
      const std::unordered_map<ValGroup, Val*>& initial_index_map)
      : traversal_graph_(traversal_graph), index_map_(initial_index_map) {}

  // Propagate the index map through a given expr of a specified
  // direction.
  void propagate(const ExprGroup& expr_group, Direction direction) {
    NVF_ERROR(!expr_group->empty());
    dispatch(expr_group->front());
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
    // If it's a broadcast, its index is always zero.
    if (id->isBroadcast()) {
      return true;
    }
    return indexMap().find(toGroup(id)) != indexMap().end();
  }

  Val* getIndex(IterDomain* id) const {
    // If it's a broadcast, its index is always zero.
    if (id->isBroadcast()) {
      return id->fusion()->zeroVal();
    }
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
};

// TODO: Should use the explicit direction
bool IdGraphIndexCompute::isForward(Expr* expr) const {
  bool ready = true;
  for (const auto inp : ir_utils::filterByType<IterDomain>(expr->inputs())) {
    if (!hasIndex(inp)) {
      VERBOSE() << "No index for input: " << inp->toString() << std::endl;
      ready = false;
      break;
    }
  }
  if (ready) {
    return true;
  }

  // Can just return false here. Just make sure the outputs are
  // already processed
  for (const auto out : ir_utils::filterByType<IterDomain>(expr->outputs())) {
    NVF_ERROR(hasIndex(out), "Output index not found: ", out->toString());
  }

  return false;
}

void IdGraphIndexCompute::handle(Split* split) {
  const bool is_forward = isForward(split);

  VERBOSE() << "IdGraphIndexCompute handle (" << (is_forward ? "fwd" : "bwd")
            << "): " << split->toString();

  if (is_forward) {
    auto in_idx = getIndex(split->in());
    auto inner_extent = split->inner()->extent();
    auto outer_idx = SimplifyingIrBuilder::divExpr(in_idx, inner_extent);
    Val* inner_idx = nullptr;
    inner_idx = SimplifyingIrBuilder::modExpr(in_idx, inner_extent);
    setIndex(split->outer(), outer_idx);
    setIndex(split->inner(), inner_idx);
  } else {
    auto outer_idx = getIndex(split->outer());
    auto inner_idx = getIndex(split->inner());
    auto inner_extent = split->inner()->extent();
    auto in_idx = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(outer_idx, inner_extent), inner_idx);
    setIndex(split->in(), in_idx);
  }
}

void IdGraphIndexCompute::handle(Merge* merge) {
  const bool is_forward = isForward(merge);

  VERBOSE() << "IdGraphIndexCompute handle (" << (is_forward ? "fwd" : "bwd")
            << "): " << merge->toString();

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
    auto tv_output = ir_utils::getTvOutput(expr);
    for (auto leaf_id : tv_output->getLeafDomain()) {
      const ValGroup& loop_group =
          id_model_.idGraph(IdMappingMode::LOOP).toGroup(leaf_id);

      if (loop_index_map_.find(loop_group) != loop_index_map_.end()) {
        // Index already assigned
        continue;
      }

      Val* loop_index = nullptr;

      // First allocate thread and grid parallel indices:
      //  The validation pass will check that the parallel bindings within the
      //  loop nodes are consistent so all the loops within this disjoint set
      //  will be realized implicitly using parallel index variables.
      ParallelType ptype = getParallelType(loop_group);
      if (isParallelTypeThread(ptype)) {
        loop_index = NamedScalar::getParallelIndex(ptype);
      } else if (shouldUseZeroIndex(loop_group)) {
        loop_index = fusion->zeroVal();
      } else {
        // Everything now should be serial concrete loops. For the mean
        // time, just use the same index integer val generated for
        // ComputeAtMap if available.
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
          NVF_ERROR(loop_index != nullptr,
                    "No existing index found for ",
                    nvfuser::toString(loop_group));
        } else {
          // Not reusing the ComputeATMap index assignments
          loop_index = IrBuilder::create<Val>(DataType::Index);
        }
      }

      NVF_ERROR(loop_index != nullptr);
      loop_index_map_[loop_group] = loop_index;
    }
  }
}

bool TensorIndexer::shouldUseZeroIndex(const ValGroup& loop_group) const {
  // For parallelized domains that have index NamedScalar's such as
  // threadIdx.x, just use the NamedScalar. It doesn't automatically
  // mean such parallel indices are actually used in the final index
  // expr. For example, TID-parallelized Local tensors won't have
  // TID-parallelized iter domains as allocation domains, so threadIdx
  // won't appear in the final index expr.
  ParallelType ptype = getParallelType(loop_group);
  if (isParallelTypeThread(ptype)) {
    return false;
  }

  // Note that the device paralle type is not included in
  // "isThread". This is necessary because we don't have a NamedScalar
  // for DID. Since it's always partitioned in any memory space
  // currently supported, it's guaranteed to be zero.
  if (isParallelTypeDeviceDim(ptype)) {
    return true;
  }

  // All loops in this set are non-parallel, non-concretized broadcast
  //  iterdomains, their "index variable" should be zero.
  if (std::all_of(loop_group->begin(), loop_group->end(), [](Val* val) {
    return val->as<IterDomain>()->isBroadcast();
  })) {
    return true;
  }

  // Trivial loop
  auto leaf_id =
      getLoopPromotion(loop_group->front()->as<IterDomain>(), id_model_);
  if (!leaf_id->maybePartial() &&
      simplifyExpr(leaf_id->extent())->isOneInt()) {
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

Val* TensorIndexer::getLinearIndex(
    TensorView* tv,
    const Expr* expr) {
  VERBOSE() << "getIndex of " << tv->toString() << " in " << expr->toString();

  const auto [allocation_domains, strides] =
      getAllocationDomains(tv, id_model_);

  VERBOSE() << "Allocation domains: " << toDelimitedString(allocation_domains)
            << std::endl;

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
    VERBOSE() << "Index of " << allocation_domain->toString() << ": "
              << idx->toInlineString() << std::endl;

    index = SimplifyingIrBuilder::addExpr(
        index, SimplifyingIrBuilder::mulExpr(idx, stride));
  }

  VERBOSE() << "Final index: " << index->toInlineString() << std::endl;

  return index;
}

IndexingInfo TensorIndexer::computeIndex(
    const Expr* expr,
    const std::vector<IterDomain*>& index_domains) const {
  const auto loop_domains = getLoopDomains(expr, id_model_);
  VERBOSE() << "Loop domains: " << toDelimitedString(loop_domains) << std::endl;

  VERBOSE() << "Index domains: " << toDelimitedString(index_domains)
            << std::endl;

  const ValGroups loop_groups = traversalGraph().toGroups(loop_domains);
  const ValGroups index_groups = traversalGraph().toGroups(index_domains);  
  const ExprPath traversal_path = ValGraphBFS::getExprsBetween(
      traversalGraph(), loop_groups, index_groups);

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

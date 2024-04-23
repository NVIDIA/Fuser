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
#include <ir/utils.h>
#include <val_graph_visitor.h>

#include <algorithm>
#include <fstream>

namespace nvfuser {

namespace {

class IndexingTraversal : public ValGraphBFS {
 public:
  IndexingTraversal(
      const ValGraph& graph,
      std::vector<GroupType> from_groups,
      std::vector<GroupType> to_groups,
      const std::unordered_set<Resize*>& resize_paths)
      : ValGraphBFS(graph, from_groups, to_groups),
        resize_paths_(resize_paths) {}

  virtual ~IndexingTraversal() = default;

  using ValGraphBFS::isVisited;

  bool isDependencySatisfied(const GroupType& group) const override {
    if (const ValGroup* vg = std::get_if<ValGroup>(&group);
        vg != nullptr && (*vg)->front()->as<IterDomain>()->isBroadcast()) {
      std::cerr << "Dependency satisfied as it's broadcast" << std::endl;
      return true;
    }
    return ValGraphBFS::isDependencySatisfied(group);
  }

  bool excludeFromTraversal(const GroupType& group) const override {
    if (const ExprGroup* eg = std::get_if<ExprGroup>(&group)) {
      if ((*eg)->empty()) {
        return false;
      }
      auto resize = dynamic_cast<Resize*>((*eg)->front());
      if (resize == nullptr) {
        return false;
      }
      if (std::none_of((*eg)->begin(), (*eg)->end(), [&](Expr* expr) -> bool {
            return resize_paths_.find(expr->as<Resize>()) !=
                resize_paths_.end();
          })) {
        return true;
      }
    }
    return false;
  }

  // This isn't necessary anymore as taken care by isDependencySatisfied
#if 0
  void traverse() override {
    // Set all broadcast groups as visited before traversal as there's
    // no need to actually visit them to get indices. Do not add their
    // neighbors to the to-visit list, though. Traversal paths should
    // still be discovered from the starting groups.
    for (const ValGroup& id_group : graph_.disjointValSets().disjointSets()) {
      if (id_group->at(0)->as<IterDomain>()->isBroadcast()) {
        //setVisited(id_group);
      }
    }

    ValGraphBFS::traverse();
  }
#endif

 private:
  const std::unordered_set<Resize*>& resize_paths_;
};

IterDomain* getLoopPromotion(IterDomain* id, const IdModel& id_model) {
  // TODO: Loop promotion should only be defined for loop domains. The
  // loop promotion map should be fixed.

  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);
  const auto& loop_promotion_map = id_model.loopPromotionMap();
  const auto& loop_group = loop_graph.toGroup(id);

  auto loop_promotion_map_it = loop_promotion_map.find(loop_group);
  NVF_ERROR(
      loop_promotion_map_it != loop_promotion_map.end(),
      "No loop promotion found: ",
      id->toString(),
      ". Loop group: ",
      nvfuser::toString(loop_group));

  return loop_promotion_map_it->second;
}

std::vector<IterDomain*> getLoopDomains(
    const Expr* expr,
    const IdModel& id_model) {
  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);

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
    NVF_ERROR(loop_graph.hasGroup(loop_id));

    auto promotion_id = getLoopPromotion(loop_id, id_model);

    loop_id = promotion_id;
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

std::optional<std::vector<IterDomain*>>
getAllocationDomainOfTransposedSmemTensor(
    const TensorView* tv,
    const ValGraph& exact_graph) {
  if (tv->getMemoryType() != MemoryType::Shared) {
    return std::nullopt;
  }

  // Can there be multiple stores with a single smem buffer?
  if (tv->uses().size() != 1) {
    return std::nullopt;
  }

  auto ls_op = dynamic_cast<LoadStoreOp*>(tv->uses().front());
  if (ls_op == nullptr) {
    return std::nullopt;
  }

  auto consumer = ls_op->out()->as<TensorView>();

  if (consumer->getMemoryType() != MemoryType::Global) {
    return std::nullopt;
  }

  // the non-inlined domains must be derived from a domain that merges
  // two constant-sized domains.

  auto getOriginatingMerge = [](IterDomain* id) -> Merge* {
    while (id != nullptr) {
      auto def = id->definition();
      if (def == nullptr) {
        return nullptr;
      } else if (auto merge = dynamic_cast<Merge*>(def)) {
        return merge;
      } else if (auto split = dynamic_cast<Split*>(def)) {
        id = split->in();
      } else {
        // Unsupported op
        NVF_ERROR(
            false,
            "Unsupported domain to get originating merge: ",
            id->toString());
      }
    }
    return nullptr;
  };

  // Find the dominating merge output domain

  std::vector<IterDomain*> non_inlined_domains{
      tv->getLeafDomain().begin() + tv->getComputeAtPosition(),
      tv->getLeafDomain().end()};

  if (non_inlined_domains.empty()) {
    return std::nullopt;
  }

  Merge* producer_common_merge =
      getOriginatingMerge(non_inlined_domains.front());
  if (producer_common_merge == nullptr) {
    return std::nullopt;
  }

  // Make sure all non inlined domains are derived from the same merge
  for (auto non_inlined_domain : non_inlined_domains) {
    auto merge = getOriginatingMerge(non_inlined_domain);
    if (merge != producer_common_merge) {
      return std::nullopt;
    }
  }

  std::cerr << "Common merge op: " << producer_common_merge->toString();

  std::vector<IterDomain*> consumer_non_inlined_domains{
      consumer->getLeafDomain().begin() +
          (consumer->nDims() - non_inlined_domains.size()),
      consumer->getLeafDomain().end()};

  Merge* consumer_common_merge =
      getOriginatingMerge(consumer_non_inlined_domains.front());
  if (consumer_common_merge == nullptr) {
    return std::nullopt;
  }
  // Make sure all non inlined domains are derived from the same merge
  for (auto non_inlined_domain : consumer_non_inlined_domains) {
    auto merge = getOriginatingMerge(non_inlined_domain);
    if (merge != consumer_common_merge) {
      return std::nullopt;
    }
  }

  // Check if the inputs to the common merge ops match
  if (exact_graph.toGroup(producer_common_merge->inner()) !=
      exact_graph.toGroup(consumer_common_merge->outer())) {
    return std::nullopt;
  }

  if (exact_graph.toGroup(producer_common_merge->outer()) !=
      exact_graph.toGroup(consumer_common_merge->inner())) {
    return std::nullopt;
  }

  // At this point, it should be safe to use the consumer non-inlined
  // domains as the allocation domain of hte producer
  return consumer_non_inlined_domains;
}

std::pair<std::vector<IterDomain*>, std::vector<Val*>> getIndexDomains(
    TensorView* tv,
    const Expr* expr,
    const IdModel& id_model) {
  // TODO: Contig merged indexing

  std::vector<IterDomain*> index_domains;
  std::vector<std::optional<bool>> contiguity;

  auto inlining_pos = tv->getComputeAtPosition();
  if (tv->isDoubleBuffered()) {
    std::cerr << "TV" << tv->name()
              << ": DB pos: " << getDoubleBufferAxisPosition(tv) << std::endl;
    std::cerr << "inlining_pos: " << inlining_pos << std::endl;
    inlining_pos = getDoubleBufferAxisPosition(tv) + 1;
  }

  // Ignore allocation of non-global tensors for now
  if (tv->hasAllocation() && tv->getMemoryType() == MemoryType::Global) {
    VERBOSE() << "Tv has allocation of " << tv->toString() << ", "
              << toDelimitedString(tv->getAllocationDomain()) << std::endl;
    index_domains = tv->getAllocationDomain();
    contiguity = tv->domain()->contiguity();
    NVF_ERROR(!tv->isDoubleBuffered());
  } else {
    // If allocation domain is not set, assume that:
    // Local/Shared: leaf domains to the right of the CA position
    // Global: rfactor domains
    if (tv->getMemoryType() == MemoryType::Global) {
      VERBOSE() << "Tv does not have allocation of " << tv->toString() << ", "
                << toDelimitedString(tv->getMaybeAllocationDomain())
                << std::endl;
      index_domains = tv->getMaybeRFactorDomain();
      contiguity = tv->domain()->contiguity();
      NVF_ERROR(!tv->isDoubleBuffered());
    } else if (tv->getMemoryType() == MemoryType::Shared) {
      for (const auto i : c10::irange(tv->nDims())) {
        auto leaf_id = tv->axis(i);
        std::cerr << "Smem leaf domain: " << leaf_id->toString() << " of "
                  << tv->toString() << std::endl;
        auto pt = leaf_id->getParallelType();
        if (isParallelTypeDeviceDim(pt) || isParallelTypeBlockDim(pt)) {
          continue;
        }
        if (i < inlining_pos && !isParallelTypeThreadDim(pt)) {
          continue;
        }
        index_domains.push_back(leaf_id);
      }
      contiguity = std::vector<std::optional<bool>>(index_domains.size(), true);
    } else {
      index_domains = {
          tv->getLeafDomain().begin() + inlining_pos,
          tv->getLeafDomain().end()};
      contiguity = std::vector<std::optional<bool>>(index_domains.size(), true);
    }
  }

  // TODO: Fix alloation domains with vectorization
  // This is an ugly workaround, but the allocation domain of a tensor
  // with vectorized domains may not be the same as the leaf fomain
  // since the vectorized domain must be at the innermost position in
  // the allocation domain, but it's allowed to be located anywhwere
  // in the leaf domain.
  // This shouldn't be necessary for global memory tensors as their
  // allocation domains are rfactor domains
  {
    if (tv->getMemoryType() != MemoryType::Global) {
      IterDomain* id_to_move_back = nullptr;
      // Vectorized load
      if (tv->definition() != nullptr && tv->definition()->isA<LoadStoreOp>() &&
          tv->definition()->as<LoadStoreOp>()->opType() ==
              LoadStoreOpType::Set) {
        auto vec_it = std::find_if(
            index_domains.begin(),
            index_domains.end(),
            [](auto index_domain) -> bool {
              return isParallelTypeVectorize(index_domain->getParallelType());
            });
        if (vec_it != index_domains.end() && *vec_it != index_domains.back()) {
          id_to_move_back = *vec_it;
        }
      } else {
        for (const auto ls_use :
             ir_utils::filterByType<LoadStoreOp>(tv->uses())) {
          if (ls_use->opType() != LoadStoreOpType::Set) {
            continue;
          }
          auto consumer_tv = ls_use->out()->as<TensorView>();
          auto vec_it = std::find_if(
              consumer_tv->getLeafDomain().begin(),
              consumer_tv->getLeafDomain().end(),
              [](auto consumer_leaf_id) -> bool {
                return isParallelTypeVectorize(
                    consumer_leaf_id->getParallelType());
              });
          if (vec_it == consumer_tv->getLeafDomain().end()) {
            continue;
          }
          const auto& vec_group =
              id_model.idGraph(IdMappingMode::EXACT).toGroup(*vec_it);
          auto index_it = std::find_if(
              index_domains.begin(),
              index_domains.end(),
              [&](auto index_id) -> bool { return vec_group->has(index_id); });
          if (index_it == index_domains.end() ||
              *index_it == index_domains.back()) {
            continue;
          }

          id_to_move_back = *index_it;
        }
      }

      if (id_to_move_back != nullptr) {
        // reorder the vec domain to the end of the index domains
        std::vector<IterDomain*> reordered_index_domains;
        reordered_index_domains.reserve(index_domains.size());
        for (const auto id : index_domains) {
          if (id != id_to_move_back) {
            reordered_index_domains.push_back(id);
          }
        }
        reordered_index_domains.push_back(id_to_move_back);
        index_domains = reordered_index_domains;
      }
    }
  }

  auto tv_for_promotion = tv;

  // WAR for transpose
  auto transposed_smem_alloc_dom = getAllocationDomainOfTransposedSmemTensor(
      tv, id_model.idGraph(IdMappingMode::EXACT));
  if (transposed_smem_alloc_dom.has_value()) {
    std::cerr
        << "Using consumer domain as the allocation domain of the shared memory producer: "
        << tv->toString() << std::endl;
    index_domains = transposed_smem_alloc_dom.value();
    tv_for_promotion = tv->uses().at(0)->output(0)->as<TensorView>();
  }

  NVF_ERROR(index_domains.size() == contiguity.size());

  std::vector<Val*> strides(index_domains.size(), nullptr);
  Val* cur_contig_stride = tv->fusion()->oneVal();
  for (const auto i : c10::irange(index_domains.size())) {
    auto dim = index_domains.size() - i - 1;
    auto index_domain = index_domains.at(dim);

    if (index_domain->isReduction()) {
      continue;
    }

    if (!contiguity.at(dim).has_value()) {
      NVF_ERROR(index_domains.at(dim)->isBroadcast());
    } else if (isAllocated(index_domain, tv)) {
      if (contiguity.at(dim).value()) {
        strides[dim] = cur_contig_stride;
        cur_contig_stride = SimplifyingIrBuilder::mulExpr(
            index_domains.at(dim)->extent(), cur_contig_stride);
      } else {
        strides[dim] = getAllocationStride(tv, (int64_t)dim);
        cur_contig_stride = SimplifyingIrBuilder::mulExpr(
            index_domains.at(dim)->extent(), strides[dim]);
      }
    }
  }

  std::vector<IterDomain*> actual_index_domains;
  std::vector<Val*> actual_strides;
  for (const auto i : c10::irange(index_domains.size())) {
    auto index_domain = index_domains.at(i);
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

ExprGroups getExprsBetween(
    const std::vector<IterDomain*>& loop_domains,
    const std::vector<IterDomain*>& index_domains,
    const ValGraph& exact_graph,
    const std::unordered_set<Resize*>& resize_paths) {
  const ValGroups loop_domain_groups = exact_graph.toGroups(loop_domains);
  const ValGroups index_domain_groups = exact_graph.toGroups(index_domains);

  VERBOSE() << "getExprsBetween: loop: " << nvfuser::toString(loop_domains)
            << ", index: " << nvfuser::toString(index_domains) << std::endl;

  IndexingTraversal traversal(
      exact_graph,
      {loop_domain_groups.vector().begin(), loop_domain_groups.vector().end()},
      {index_domain_groups.vector().begin(),
       index_domain_groups.vector().end()},
      resize_paths);

  traversal.traverse();

  const ExprGroups exprs = traversal.getShortestExprPath();

  // const ExprGroups exprs = ValGraphBFS::getExprsBetweenVals(
  // exact_graph, loop_domain_groups, index_domain_groups);

  return exprs;
}

ExprGroups getIndexingTraversalPath(
    TensorView* tv,
    const Expr* expr,
    const std::vector<IterDomain*>& from_domains,
    const std::vector<IterDomain*>& to_domains,
    const ValGraph& traversal_graph) {
  auto consumer_tv = ir_utils::getTvOutput(expr);
  std::unordered_set<Resize*> resize_paths;
  if (consumer_tv->hasRFactor()) {
    auto root_to_rf_exprs = StmtSort::getExprsBetween(
        {consumer_tv->getRootDomain().begin(),
         consumer_tv->getRootDomain().end()},
        {consumer_tv->getRFactorDomain().begin(),
         consumer_tv->getRFactorDomain().end()});
    for (Expr* root_to_rf_expr : root_to_rf_exprs) {
      if (auto resize = dynamic_cast<Resize*>(root_to_rf_expr)) {
        resize_paths.insert(resize);
      }
    }
  }

  auto indexing_path =
      getExprsBetween(from_domains, to_domains, traversal_graph, resize_paths);

  VERBOSE() << "Indexing path:\n";
  for (const auto& expr_group : indexing_path) {
    Expr* expr = expr_group->front();
    VERBOSE() << expr->toString();
  }
  VERBOSE() << "--- path done ---\n";

  return indexing_path;
}

class IdGraphIndexCompute : public OptOutDispatch {
 public:
  IdGraphIndexCompute(
      const ValGraph& exact_graph,
      const std::unordered_map<ValGroup, Val*>& initial_index_map)
      : exact_graph_(exact_graph), index_map_(initial_index_map) {}

  void propagate(const ExprGroup& expr_group);

  using OptOutDispatch::handle;

  void handle(Split* split) override;

  void handle(Merge* merge) override;

  void handle(Resize* resize) override;

  bool isForward(Expr* expr) const;

  bool hasIndex(IterDomain* id) const;

  Val* getIndex(IterDomain* id) const;

  void setIndex(IterDomain* id, Val* idx);

  std::unordered_map<ValGroup, Val*> indexMap() const {
    return index_map_;
  }

 private:
  const ValGraph& exact_graph_;
  std::unordered_map<ValGroup, Val*> index_map_;
};

void IdGraphIndexCompute::propagate(const ExprGroup& expr_group) {
  NVF_ERROR(!expr_group->empty());
  dispatch(expr_group->front());
}

bool IdGraphIndexCompute::hasIndex(IterDomain* id) const {
  // If it's a broadcast, its index is always zero.
  if (id->isBroadcast()) {
    return true;
  }
  const ValGroup& id_group = exact_graph_.toGroup(id);
  return index_map_.find(id_group) != index_map_.end();
}

Val* IdGraphIndexCompute::getIndex(IterDomain* id) const {
  // If it's a broadcast, its index is always zero.
  if (id->isBroadcast()) {
    return id->fusion()->zeroVal();
  }
  const ValGroup& id_group = exact_graph_.toGroup(id);
  auto it = index_map_.find(id_group);
  NVF_ERROR(it != index_map_.end(), "Index not found: ", id->toString());
  return it->second;
}

void IdGraphIndexCompute::setIndex(IterDomain* id, Val* idx) {
  std::cerr << "setIndex: " << id->name() << " -> " << idx->toInlineString()
            << std::endl;
  const ValGroup& id_group = exact_graph_.toGroup(id);
  // Due to AlmostExact cycles, can't guarantee the same group is
  // visited only once
#if 1
  index_map_.emplace(id_group, idx);
#else
  NVF_ERROR(
      index_map_.emplace(id_group, idx).second,
      "Index already set: ",
      id->toString(),
      ". Preious: ",
      getIndex(id)->toString(),
      " (",
      getIndex(id)->toInlineString(),
      "). New: ",
      idx->toString(),
      ", (",
      idx->toInlineString(),
      ")");
#endif
}

bool IdGraphIndexCompute::isForward(Expr* expr) const {
  bool ready = true;
  for (const auto inp : ir_utils::filterByType<IterDomain>(expr->inputs())) {
    if (!hasIndex(inp)) {
      std::cerr << "No index for input: " << inp->toString() << std::endl;
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
    auto inner_idx = SimplifyingIrBuilder::modExpr(in_idx, inner_extent);
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

  // TODO: use getMaybeExpandedExtent?
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
    auto inner_idx = SimplifyingIrBuilder::modExpr(out_idx, inner_ext);
    setIndex(merge->inner(), inner_idx);
  }
}

void IdGraphIndexCompute::handle(Resize* resize) {
  const bool is_forward = isForward(resize);

  VERBOSE() << "IdGraphIndexCompute handle (" << (is_forward ? "fwd" : "bwd")
            << "): " << resize->toString();

  auto left_expand = resize->leftExpand();

  auto in_id = is_forward ? resize->in() : resize->out();
  auto out_id = is_forward ? resize->out() : resize->in();

  if (left_expand->isZeroInt()) {
    // Just forward as is
    setIndex(out_id, getIndex(in_id));
    return;
  }

  auto in_idx = getIndex(in_id);
  Val* out_idx = nullptr;

  if (is_forward) {
    out_idx = SimplifyingIrBuilder::addExpr(in_idx, left_expand);
  } else {
    out_idx = SimplifyingIrBuilder::subExpr(in_idx, left_expand);
  }

  setIndex(out_id, out_idx);
}

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

std::vector<IterDomain*> getPredicateDomains(
    TensorView* tv,
    const Expr* expr,
    const IdModel& id_model) {
  // TODO: Contig merged indexing

  // Rfactor domains should be the domains to predicate as they define
  // the logical shape of a tensor. However, in the case of rfactored
  // reductions, rfactor splits may not be divisible, thus root
  // domains need to be predicated. Note that the non-divisible split
  // info does not seem to cover non-divisible reduction rfactor
  // splits.
  std::vector<IterDomain*> predicate_domains =
      tv->hasReduction() ? tv->getRootDomain() : tv->getMaybeRFactorDomain();

  // Broadcast domains should not be predicated
  predicate_domains.erase(
      std::remove_if(
          predicate_domains.begin(),
          predicate_domains.end(),
          [](IterDomain* id) -> bool { return id->isBroadcast(); }),
      predicate_domains.end());

  // If this is an expr initializing a buffer for a reduction, the
  // reduction domains do not need to be predicated. In fact, if it's
  // a Local or Shared memory, no predicate is necessary
  if (lower_utils::isReductionInitExpr(expr)) {
    VERBOSE() << "Reduction init expr: " << expr->toString();
    if (isAllocationBasedOnLeaf(tv)) {
      return {};
    } else {
      predicate_domains.erase(
          std::remove_if(
              predicate_domains.begin(),
              predicate_domains.end(),
              [](IterDomain* id) -> bool { return id->isReduction(); }),
          predicate_domains.end());
    }
  }

  return predicate_domains;
}

kir::ForLoop* getForLoop(
    IterDomain* loop_id,
    const std::vector<kir::ForLoop*>& for_loops,
    const ValGraph& loop_graph) {
  auto it = std::find_if(
      for_loops.begin(), for_loops.end(), [&](kir::ForLoop* for_loop) -> bool {
        IterDomain* for_loop_id = for_loop->iter_domain();
        return loop_graph.disjointValSets().strictAreMapped(
            loop_id, for_loop_id);
      });
  if (it != for_loops.end()) {
    return *it;
  } else {
    return nullptr;
  }
}

} // namespace

TensorIndexer::TensorIndexer(const IdModel& id_model) : id_model_(id_model) {
  buildLoopIndexMap();

  const auto& non_divisible_split_info =
      GpuLower::current()->nonDivisibleSplitInfo();
  for (const auto& [tv, splits] :
       non_divisible_split_info.splitsToPredicate()) {
    std::cerr << "Splits to predicate of tensor: " << tv->toString() << "\n";
    for (const auto split : splits) {
      std::cerr << "\t" << split->toString();
    }
  }
}

void TensorIndexer::buildLoopIndexMap() {
  if (id_model_.idGraph(IdMappingMode::EXACT)
          .disjointValSets()
          .disjointSets()
          .empty()) {
    return;
  }

  std::cerr << "Exact Graph:\n";
  for (const auto& g : id_model_.idGraph(IdMappingMode::EXACT)
                           .disjointValSets()
                           .disjointSets()) {
    std::cerr << nvfuser::toString(g) << std::endl;
  }

  std::cerr << "Almost Exact Graph:\n";
  for (const auto& g : id_model_.idGraph(IdMappingMode::ALMOSTEXACT)
                           .disjointValSets()
                           .disjointSets()) {
    std::cerr << nvfuser::toString(g) << std::endl;
  }

  if (getenv("DOT")) {
    std::ofstream ofs("exact_graph.dot", std::ofstream::trunc);
    auto dot_string =
        ValGraphDotPrinter::getString(id_model_.idGraph(IdMappingMode::EXACT));
    std::cerr << dot_string << std::endl;
    ofs << dot_string;
    ofs.close();
  }

  Fusion* fusion = id_model_.idGraph(IdMappingMode::EXACT)
                       .disjointValSets()
                       .disjointSets()
                       .front()
                       ->front()
                       ->fusion();

  FusionGuard fg(fusion);

  auto shouldUseZeroIndex = [&](const ValGroup& loop_group) -> bool {
    ParallelType ptype = getParallelType(loop_group);
    if (isParallelTypeThread(ptype)) {
      return false;
    }

    // The device paralle type is not included in "isThread". We don't
    // allocate any index variable for device-parallel domains.
    if (isParallelTypeDeviceDim(ptype)) {
      return true;
    }

    if (ptype == ParallelType::Vectorize) {
      return true;
    }

    // All loops in this set are non-parallel, non-concretized broadcast
    //  iterdomains, their "index variable" should be zero.
    if (std::all_of(loop_group->begin(), loop_group->end(), [](Val* val) {
          return val->as<IterDomain>()->isBroadcast();
        })) {
      std::cerr << "All domains are broadcast: "
                << nvfuser::toString(loop_group) << std::endl;
      return true;
    }

    // Trivial loop
    // TODO: consider expanded extent?
    auto leaf_id =
        getLoopPromotion(loop_group->front()->as<IterDomain>(), id_model_);
    if (!leaf_id->maybePartial() &&
        simplifyExpr(leaf_id->extent())->isOneInt()) {
      return true;
    }

    return false;
  };

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

      // TODO: halo loop not considered
      // TODO: double buffering not considered

      Val* loop_index = nullptr;

      // First allocate thread and grid parallel indices:
      //  The validation pass will check that the parallel bindings within the
      //  loop nodes are consistent so all the loops within this disjoint set
      //  will be realized implicitly using parallel index variables.
      ParallelType ptype = getParallelType(loop_group);
      if (isParallelTypeThread(ptype)) {
        loop_index = NamedScalar::getParallelIndex(ptype);
      } else if (shouldUseZeroIndex(loop_group)) {
        std::cerr << "Use zero for " << nvfuser::toString(loop_group)
                  << std::endl;
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
            std::cerr << "Trying to find index val for " << id->toString()
                      << std::endl;
            loop_index = ca_map->getIndexVariable(id);
            break;
          }
          if (loop_index == nullptr) {
            std::cerr << "No existing index found for "
                      << nvfuser::toString(loop_group) << std::endl;
          }
        } else {
          loop_index = IrBuilder::create<Val>(DataType::Index);
        }
      }
      loop_index_map_[loop_group] = loop_index;
      std::cerr << "Loop index map: " << nvfuser::toString(loop_group) << " -> "
                << loop_index->toInlineString() << std::endl;
    }
  }
}

Val* TensorIndexer::getLoopIndex(IterDomain* loop_id) const {
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
    TensorView* tv,
    const Expr* expr,
    const std::optional<std::vector<kir::ForLoop*>>& for_loops,
    const std::vector<IterDomain*>& loop_domains,
    const ValGraph& traversal_graph,
    bool predicate) const {
  // loop_index_map_ is a map on the loop graph. For index
  // propagation, need a map for the exact graph

  const bool as_consumer =
      std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
      expr->outputs().end();

  std::unordered_map<ValGroup, Val*> initial_index_map;

  auto getForLoop = [&](IterDomain* id) -> kir::ForLoop* {
    if (!for_loops.has_value()) {
      return nullptr;
    }

    auto it = std::find_if(
        for_loops->begin(), for_loops->end(), [&](kir::ForLoop* fl) -> bool {
          return id_model_.idGraph(IdMappingMode::LOOP)
              .disjointValSets()
              .strictAreMapped(fl->iter_domain(), id);
        });
    if (it != for_loops->end()) {
      return *it;
    } else {
      return nullptr;
    }
  };

  for (IterDomain* loop_id : loop_domains) {
    Val* loop_index = getLoopIndex(loop_id);
    const auto& exact_group = traversal_graph.toGroup(loop_id);
    VERBOSE() << "Setting initial index. " << loop_id->toString() << ", "
              << nvfuser::toString(exact_group) << ", "
              << loop_index->toInlineString() << std::endl;

    kir::ForLoop* for_loop = getForLoop(loop_id);

    if (for_loops.has_value() && for_loop == nullptr &&
        loop_domains.size() == for_loops->size()) {
      // Why this happen?
      std::cerr << "ForLoop not found for " << loop_id->toString() << std::endl;
      for (const auto fl : *for_loops) {
        std::cerr << "FL: " << fl->iter_domain()->toString() << std::endl;
      }
      NVF_ERROR(false);
    }

    // Even when the iter-domain is not size-1, the actual for-loop
    // can be (e.g., for double buffering)
    if (for_loop != nullptr && for_loop->isTrivial()) {
      std::cerr << "Replacing a loop index with a loop start val: "
                << for_loop->start()->toInlineString()
                << ". Prev: " << loop_index->toInlineString() << std::endl;
      loop_index = for_loop->start();
    }

    // If the for-loop is double-buffered and not prologue, the loop
    // index should be advanced by one except for the double-buffered
    // tensor itself
    if (for_loops.has_value() && GpuLower::hasCurrent() && !as_consumer) {
      loop_index = adjustProducerLoopIndexForDoubleBuffering(
          tv, ir_utils::getTvOutput(expr), for_loop, loop_index);
    }

    if (predicate && for_loop != nullptr && predicateAtEnd(for_loop)) {
    }

    if (initial_index_map.find(exact_group) != initial_index_map.end()) {
      // Initial index already set. This can happen as exact_group is
      // actually an almost-exact group. It should be just size-1
      // domain.
      NVF_ERROR(
          loop_index->isZeroInt(),
          "Unexpected initial index: ",
          loop_index->toInlineString());
      auto existing_index = initial_index_map.at(exact_group);
      NVF_ERROR(
          existing_index->isZeroInt(),
          "Unexpected initial index: ",
          existing_index->toInlineString());
      continue;
    }

    NVF_ERROR(
        initial_index_map.emplace(exact_group, loop_index).second,
        "Initial index already set for ",
        nvfuser::toString(exact_group),
        ". Existing: ",
        initial_index_map.at(exact_group)->toInlineString(),
        ". New: ",
        loop_index->toInlineString());
  }

  return initial_index_map;
}

// 1. Find the loop domains
// 2. Find the index domains
// 3. Find the path from the loop domains to the allocation domains
// 4. Set the initial index vals
// 5. Propagate the initial indices of the loop domains to the index
// domains
Val* TensorIndexer::getIndex(
    TensorView* tv,
    const Expr* expr,
    const std::optional<std::vector<kir::ForLoop*>>& for_loops) {
  const auto& traversal_graph = id_model_.idGraph(IdMappingMode::ALMOSTEXACT);

  bool as_consumer =
      std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
      expr->outputs().end();

  VERBOSE() << "getIndex of " << tv->toString() << " as "
            << (as_consumer ? "consumer" : "producer") << " in "
            << expr->toString() << std::endl;

  const auto [index_domains, strides] = getIndexDomains(tv, expr, id_model_);

  VERBOSE() << "Index domains: " << toDelimitedString(index_domains)
            << std::endl;

  const auto& index_info =
      getIndex(tv, expr, for_loops, index_domains, traversal_graph, false);
  const auto& index_map = index_info.index_map;

  Val* index = tv->fusion()->zeroVal();
  for (const auto i : c10::irange(index_domains.size())) {
    auto index_domain = index_domains.at(i);
    auto idx_it = index_map.find(traversal_graph.toGroup(index_domain));
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        index_domain->toString());
    Val* idx = idx_it->second;
    VERBOSE() << "Index of " << index_domain->toString() << ": "
              << idx->toInlineString() << std::endl;

    index = SimplifyingIrBuilder::addExpr(
        index, SimplifyingIrBuilder::mulExpr(idx, strides.at(i)));
  }

  // Process double buffering when for-loops are given
  if (tv->isDoubleBuffered() && for_loops.has_value() &&
      GpuLower::hasCurrent()) {
    auto adjusted_index =
        adjustIndexToSwitchBuffer(tv, as_consumer, for_loops.value(), index);
    std::cerr << "Adjustment done for switching buffer of " << tv->toString()
              << ": " << adjusted_index->toInlineString()
              << ", before: " << index->toInlineString() << std::endl;
    index = adjusted_index;
  }

  VERBOSE() << "Final index: " << index->toInlineString() << std::endl;

  return index;
}

Val* TensorIndexer::adjustProducerLoopIndexForDoubleBuffering(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const kir::ForLoop* for_loop,
    Val* loop_index) const {
  NVF_ERROR(for_loop != nullptr);

  // Double-buffered tensor itself does not need this adjustment
  if (producer_tv->isDoubleBuffered() &&
      id_model_.idGraph(IdMappingMode::LOOP)
          .disjointValSets()
          .strictAreMapped(
              getDoubleBufferAxis(producer_tv), for_loop->iter_domain())) {
    return loop_index;
  }

  if (for_loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Main &&
      for_loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Epilog) {
    return loop_index;
  }

  if (!consumer_tv->isDoubleBuffered()) {
    return loop_index;
  }

  const auto gpu_lower = GpuLower::current();
  NVF_ERROR(
      gpu_lower != nullptr,
      "Double buffering info of GpuLower is required but GpuLower is missing");

  auto stage_depth =
      (int64_t)GpuLower::current()->doubleBufferInfo().getStageDepthFor(
          for_loop->iter_domain());

  auto adjusted_loop_index = SimplifyingIrBuilder::addExpr(
      loop_index,
      SimplifyingIrBuilder::create<Val>(stage_depth - 1L, DataType::Index));

  std::cerr << "Adjusted loop index: " << adjusted_loop_index->toInlineString()
            << ", Prev: " << loop_index->toInlineString() << std::endl;

  return adjusted_loop_index;
}

Val* TensorIndexer::adjustIndexToSwitchBuffer(
    TensorView* tv,
    bool as_consumer,
    const std::vector<kir::ForLoop*>& for_loops,
    Val* idx) const {
  if (!tv->isDoubleBuffered()) {
    return idx;
  }

  const auto gpu_lower = GpuLower::current();
  NVF_ERROR(
      gpu_lower != nullptr,
      "Double buffering info of GpuLower is required but GpuLower is missing");

  auto db_loop =
      gpu_lower->doubleBufferInfo().getDoubleBufferLoop(tv, for_loops);

  NVF_ERROR(db_loop != nullptr);

  std::cerr << "DB loop of " << tv->toString() << ": "
            << db_loop->iter_domain()->toString() << std::endl;

  // Mostly just copied from getNonGlobalConsumerStridedIndices
  auto stage_depth = (int64_t)gpu_lower->doubleBufferInfo().getStageDepthFor(
      db_loop->iter_domain());
  bool is_circular_buffer_loop = stage_depth > 2;
  bool is_prolog =
      db_loop->doubleBufferLoopStage() == DoubleBufferLoopStage::Prolog;

  NVF_ERROR(!is_circular_buffer_loop, "Circular buffering not supported yet");

  // Prologue doesn't need anything here
  if (is_prolog) {
    return idx;
  }

  if (db_loop->doubleBufferLoopStage() == DoubleBufferLoopStage::Epilog) {
    std::cerr << "Epilog loop:\n"
              << db_loop->toString() << db_loop->start()->toInlineString()
              << " -> " << db_loop->stop()->toInlineString()
              << ", current index: " << idx->toInlineString() << std::endl;
  }

  auto loop_index = getLoopIndex(db_loop->iter_domain());
  if (db_loop->isTrivial()) {
    loop_index = db_loop->start();
  }

  auto db_index_offset = loop_index;
  if (as_consumer) {
    // Read-ahead offset for consumer indexing
    db_index_offset = SimplifyingIrBuilder::addExpr(
        db_index_offset,
        SimplifyingIrBuilder::create<Val>(stage_depth - 1, DataType::Index));
  }

  db_index_offset = SimplifyingIrBuilder::modExpr(
      db_index_offset,
      SimplifyingIrBuilder::create<Val>(stage_depth, DataType::Index));

  auto original_alloc_size =
      gpu_lower->doubleBufferInfo().getOriginalAllocSize(tv);

  auto db_strided_index =
      SimplifyingIrBuilder::mulExpr(db_index_offset, original_alloc_size);

  auto updated_idx = SimplifyingIrBuilder::addExpr(idx, db_strided_index);
  return updated_idx;
}

IndexingInfo TensorIndexer::getIndex(
    TensorView* tv,
    const Expr* expr,
    const std::optional<std::vector<kir::ForLoop*>>& for_loops,
    const std::vector<IterDomain*>& index_domains,
    const ValGraph& traversal_graph,
    bool predicate) const {
  // Step 1: Find loop domains (same as indexing)
  // Step 2: Find rfactor domains
  // Step 3: Find the path from the loop domains to the rfactor
  // domains
  // Step 4: Set the initial indices
  // Step 5: Propagate the indices along the path

  VERBOSE() << "getIndexMap of " << tv->toString() << " in " << expr->toString()
            << std::endl;

  auto loop_domains = getLoopDomains(expr, id_model_);
  VERBOSE() << "Loop domains: " << toDelimitedString(loop_domains) << std::endl;

  VERBOSE() << "Index domains: " << toDelimitedString(index_domains)
            << std::endl;

  auto traversal_path = getIndexingTraversalPath(
      tv, expr, loop_domains, index_domains, traversal_graph);

  const auto initial_index_map = getInitialIndexMap(
      tv, expr, for_loops, loop_domains, traversal_graph, predicate);

  IdGraphIndexCompute index_compute(traversal_graph, initial_index_map);

  for (const auto& expr_group : traversal_path) {
    index_compute.propagate(expr_group);
  }

  IndexingInfo info{traversal_path, index_compute.indexMap()};
  return info;
}

std::unordered_map<Val*, Val*> getPredicateIndexReplacementMap(
    const std::vector<kir::ForLoop*>& for_loops,
    bool is_start_predicate,
    const std::unordered_map<ValGroup, Val*>& index_map,
    const ValGraph& traversal_graph) {
  std::unordered_map<Val*, Val*> replacement_map;

  for (const auto fl : for_loops) {
    auto loop_id = fl->iter_domain();
    NVF_ERROR(
        !loop_id->maybePartial(),
        "Partial loop not supported: ",
        fl->toString());
    auto loop_index_it = index_map.find(traversal_graph.toGroup(loop_id));
    if (loop_index_it == index_map.end()) {
      // The index map is built from the tensor loop domains. There
      // can be for-loops that are not part of the tensor loop
      // domains.
      continue;
    }
    NVF_ERROR(
        loop_index_it != index_map.end(),
        "Index for a loop not found: ",
        loop_id->toString());
    Val* loop_index = loop_index_it->second;
    if (predicateAtEnd(fl)) {
      if (loop_id->isThread()) {
        continue;
      }
      Val* replacement = is_start_predicate
          ? loop_index->fusion()->zeroVal()
          : SimplifyingIrBuilder::subExpr(
                loop_id->extent(), loop_index->fusion()->oneVal());
      auto inserted = replacement_map.emplace(loop_index, replacement).second;
      NVF_ERROR(
          inserted, "Duplicate replacement attempted: ", loop_id->toString());
      VERBOSE() << "Replacing initial index: " << loop_index->toInlineString()
                << " with " << replacement->toInlineString() << std::endl;
    }
  }

  return replacement_map;
}

namespace {

bool isNonDivisibleSplit(const ExprGroup& expr_group) {
  const auto& non_divisible_split_info =
      GpuLower::current()->nonDivisibleSplitInfo();

  std::vector<PredicateDomainInfo> pred_info_vec;

  // non_divisible_split_info should just have a set of all
  // non-divisible splits
  for (const auto& [tv, splits] :
       non_divisible_split_info.splitsToPredicate()) {
    if (std::find_if(splits.begin(), splits.end(), [&](Split* split) {
          return expr_group->has(split);
        }) != splits.end()) {
      return true;
    }
  }

  return false;
}

} // namespace

std::vector<RootPredicateInfo> TensorIndexer::getPredicates(
    TensorView* tv,
    const Expr* expr,
    const std::optional<std::vector<kir::ForLoop*>>& for_loops) {
  VERBOSE() << "getPredicates of " << tv->toString() << " in "
            << expr->toString();

  NVF_ERROR(for_loops.has_value());

  const auto& traversal_graph = id_model_.idGraph(IdMappingMode::ALMOSTEXACT);
  const auto zero_val = tv->fusion()->zeroVal();

  const auto& predicate_domains = getPredicateDomains(tv, expr, id_model_);

  VERBOSE() << "Predicate domains: " << toDelimitedString(predicate_domains)
            << std::endl;

  const auto& index_info =
      getIndex(tv, expr, for_loops, predicate_domains, traversal_graph, true);
  const auto& index_map = index_info.index_map;

  const auto& replacement_map_start = getPredicateIndexReplacementMap(
      for_loops.value(), true, index_map, traversal_graph);

  const auto& replacement_map_stop = getPredicateIndexReplacementMap(
      for_loops.value(), false, index_map, traversal_graph);

  auto non_divisible_splits = getNonDivisibleConsumerDomainsToPredicate(tv);

  std::vector<RootPredicateInfo> info_vec;
  info_vec.reserve(predicate_domains.size() + non_divisible_splits.size());

  for (const auto& predicate_domain : predicate_domains) {
    auto idx_it = index_map.find(traversal_graph.toGroup(predicate_domain));
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        predicate_domain->toString());
    Val* idx = idx_it->second;
    VERBOSE() << "Predicate index of " << predicate_domain->toString() << ": "
              << idx->toInlineString() << std::endl;

    RootPredicateInfo info;
    // For now, just set zero for both start and stop offsets
    info.start_offset_ = zero_val;
    info.stop_offset_ = zero_val;

    // Use the same index for start and stop
    auto start_idx =
        ir_utils::replaceValRecursively(idx, replacement_map_start);
    info.start_predicate_ = SimplifyingIrBuilder::geExpr(
        SimplifyingIrBuilder::addExpr(start_idx, info.start_offset_), zero_val);

    // TODO: predicate elimination
    auto stop_idx = ir_utils::replaceValRecursively(idx, replacement_map_stop);
    info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
        SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
        predicate_domain->extent());

    info.root_ids_ = {predicate_domain};

    info_vec.emplace_back(info);
  }

  // If this is a reduction init expr, then no need to take care of
  // non divisible splits
  if (!lower_utils::isReductionInitExpr(expr)) {
#if 0
    for (const PredicateDomainInfo& non_divisible_split_info :
             non_divisible_splits) {
      IterDomain* non_divisible_domain = non_divisible_split_info.id;
      VERBOSE() << "Non-divisible predicate: " << non_divisible_split_info.id->toString()
                << std::endl;
      RootPredicateInfo info;
      info.start_offset_ = zero_val;
      info.stop_offset_ = zero_val;

      auto idx_it = index_map.find(traversal_graph.toGroup(non_divisible_domain));
      NVF_ERROR(
          idx_it != index_map.end(),
          "Index not found for non-divisible split domain: ",
          non_divisible_domain->toString());

      auto idx = ir_utils::replaceValRecursively(idx_it->second, replacement_map_stop);
      info.stop_predicate_ =
          SimplifyingIrBuilder::ltExpr(idx, non_divisible_domain->extent());
      VERBOSE() << "Precicate: " << info.stop_predicate_->toInlineString()
                << std::endl;
      info.root_ids_ = {non_divisible_domain};
      info_vec.emplace_back(info);
    }
#else
    for (const ExprGroup& eg : index_info.traversal_path) {
      if (!isNonDivisibleSplit(eg)) {
        continue;
      }

      NVF_ERROR(eg->front()->isA<Split>());
      auto split_to_predicate = eg->front()->as<Split>();
      VERBOSE() << "Non-divisible predicate: "
                << split_to_predicate->toString();

      IterDomain* non_divisible_domain = split_to_predicate->in();

      RootPredicateInfo info;
      info.start_offset_ = zero_val;
      info.stop_offset_ = zero_val;

      auto idx_it =
          index_map.find(traversal_graph.toGroup(non_divisible_domain));
      NVF_ERROR(
          idx_it != index_map.end(),
          "Index not found for non-divisible split domain: ",
          non_divisible_domain->toString());

      auto idx =
          ir_utils::replaceValRecursively(idx_it->second, replacement_map_stop);
      info.stop_predicate_ =
          SimplifyingIrBuilder::ltExpr(idx, non_divisible_domain->extent());
      VERBOSE() << "Precicate: " << info.stop_predicate_->toInlineString()
                << std::endl;
      info.root_ids_ = {non_divisible_domain};
      info_vec.emplace_back(info);
    }
#endif
  }

  return info_vec;
}

bool TensorIndexer::isSupported(Fusion* fusion) {
  const auto all_tvs = ir_utils::allTvs(fusion);

  for (const auto& tv : all_tvs) {
    std::stringstream reason;

    if (tv->isCircularBuffered()) {
      reason << "Circular buffering is used: " << tv->toString();
    } else {
      for (const auto& id : ir_utils::allIDsOf(tv)) {
        if (id->getParallelType() == ParallelType::MisalignedVectorize) {
          reason << "MialignedVectorize is used: " << id->toString();
          break;
        } else if (auto swizzle = dynamic_cast<Swizzle*>(id->definition())) {
          reason << "Swizzle not supported: " << swizzle->toString();
          break;
        } else if (
            auto swizzle2d = dynamic_cast<Swizzle2D*>(id->definition())) {
          reason << "Swizzle2D not supported: " << swizzle2d->toString();
          break;
        } else if (ir_utils::isIndexedID(tv, id)) {
          reason << "Index ops such as select not supported: "
                 << tv->toString();
          break;
        }
      }
    }

    if (!reason.str().empty()) {
      std::cerr << "TensorIndexer disabled due to: " << reason.str()
                << std::endl;
      return false;
    }
  }

  return true;
}

} // namespace nvfuser

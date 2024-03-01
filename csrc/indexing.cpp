// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/to_string.h>
#include <indexing.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {

namespace {

IterDomain* getLoopPromotion(IterDomain* id, const IdModel& id_model) {
  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);
  const auto& loop_promotion_map = id_model.loopPromotionMap();
  const auto& loop_group = loop_graph.toGroup(id);

  auto loop_promotion_map_it = loop_promotion_map.find(loop_group);
  NVF_ERROR(
      loop_promotion_map_it != loop_promotion_map.end(),
      "No loop promotion found: ", id->toString());

  return loop_promotion_map_it->second;
}

std::vector<IterDomain*> getLoopDomains(
    TensorView* tv,
    Expr* expr,
    const IdModel& id_model) {
  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);

  std::vector<IterDomain*> loop_domains;

  for (auto leaf_id : tv->getLeafDomain()) {
    NVF_ERROR(loop_graph.hasGroup(leaf_id));

    auto promotion_id = getLoopPromotion(leaf_id, id_model);

    loop_domains.push_back(promotion_id);
  }

  return loop_domains;
}

std::vector<IterDomain*> getIndexDomains(
    TensorView* tv,
    Expr* expr,
    const IdModel& id_model) {

  if (tv->hasAllocation()) {
    return tv->getAllocationDomain();
  }

  // If allocation domain is not set, assume that:
  // Local/Shared: leaf domains to the right of the CA position
  // Global: rfactor domains

  if (tv->getMemoryType() == MemoryType::Global) {
    return tv->getMaybeRFactorDomain();
  } else {
    std::vector<IterDomain*> index_domains;
    index_domains.reserve(tv->nDims() - tv->getComputeAtPosition());
    for (const auto i : c10::irange(tv->getComputeAtPosition(), tv->nDims())) {
      auto leaf_id = tv->axis(i);
      auto promotion_id = getLoopPromotion(leaf_id, id_model);
      index_domains.push_back(promotion_id);
    }
    return index_domains;
  }
}

ExprGroups getExprsBetween(
    const std::vector<IterDomain*>& loop_domains,
    const std::vector<IterDomain*>& index_domains,
    const ValGraph& exact_graph) {

  const ValGroups loop_domain_groups = exact_graph.toGroups(loop_domains);
  const ValGroups index_domain_groups = exact_graph.toGroups(index_domains);

  const ExprGroups exprs = ValGraphBFS::getExprsBetweenVals(
      exact_graph, loop_domain_groups, index_domain_groups);

  return exprs;
}

class IndexCompute : public OptOutDispatch {
 public:
  IndexCompute(
      const ValGraph& exact_graph,
      const std::unordered_map<ValGroup, Val*>& initial_index_map)
      : exact_graph_(exact_graph), index_map_(initial_index_map) {}

  void propagate(const ExprGroup& expr_group);

  using OptOutDispatch::handle;

  void handle(Split* split) override;

  void handle(Merge* merge) override;

  bool isForward(Expr* expr) const;

  Val* getIndex(IterDomain* id) const;

  void setIndex(IterDomain* id, Val* idx);

 private:
  const ValGraph& exact_graph_;
  std::unordered_map<ValGroup, Val*> index_map_;
};

void IndexCompute::propagate(const ExprGroup& expr_group) {
  NVF_ERROR(!expr_group->empty());
  dispatch(expr_group->front());
}

Val* IndexCompute::getIndex(IterDomain* id) const {
  const ValGroup& id_group = exact_graph_.toGroup(id);
  if (auto it = index_map_.find(id_group);
      it != index_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

void IndexCompute::setIndex(IterDomain* id, Val* idx) {
  const ValGroup& id_group = exact_graph_.toGroup(id);
  NVF_ERROR(
      index_map_.emplace(id_group, idx).second,
      "Index already set: ",
      id->toString(),
      ". Preious: ",
      getIndex(id)->toString(),
      ". New: ",
      idx->toString());
}

bool IndexCompute::isForward(Expr* expr) const {
  bool ready = true;
  for (const auto inp : ir_utils::filterByType<IterDomain>(expr->inputs())) {
    if (getIndex(inp) == nullptr) {
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
    NVF_ERROR(
        getIndex(out) != nullptr,
        "Output index not found: ", out->toString());
  }

  return false;
}

void IndexCompute::handle(Split* split) {
  std::cerr << "IndexCompute handle: " << split->toString();

  const bool is_forward = isForward(split);

  if (is_forward) {
    auto in_idx = getIndex(split->in());
    auto outer_idx = SimplifyingIrBuilder::divExpr(in_idx, split->factor());
    setIndex(split->outer(), outer_idx);
    auto inner_idx = SimplifyingIrBuilder::modExpr(in_idx, split->factor());
    setIndex(split->inner(), inner_idx);
  } else {
    auto outer_idx = getIndex(split->outer());
    auto inner_idx = getIndex(split->inner());
    auto in_idx = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(outer_idx, split->factor()),
        inner_idx);
    setIndex(split->in(), in_idx);
  }

}

void IndexCompute::handle(Merge* merge) {
  std::cerr << "IndexCompute handle: " << merge->toString();

  const bool is_forward = isForward(merge);

  // TODO: use getMaybeExpandedExtent?
  auto inner_ext = merge->inner()->extent();

  if (is_forward) {
    auto outer_idx = getIndex(merge->outer());
    auto inner_idx = getIndex(merge->inner());
    auto out_idx = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(outer_idx, inner_ext),
        inner_idx);
    setIndex(merge->out(), out_idx);
  } else {
    auto out_idx = getIndex(merge->out());
    auto outer_idx = SimplifyingIrBuilder::divExpr(out_idx, inner_ext);
    setIndex(merge->outer(), outer_idx);
    auto inner_idx = SimplifyingIrBuilder::modExpr(out_idx, inner_ext);
    setIndex(merge->inner(), inner_idx);
  }
}

} // namespace

Indexing::Indexing(const IdModel& id_model) : id_model_(id_model) {
  buildLoopIndexMap();
}

void Indexing::buildLoopIndexMap() {
  Fusion* fusion = id_model_.idGraph(IdMappingMode::LOOP)
      .disjointValSets()
      .disjointSets().front()->front()->fusion();

  FusionGuard fg(fusion);

  // Run through all disjoint sets registered in loop map,
  //  all lowered kir::ForLoop will correspond to one of the disjoint sets
  //  and we only need one index variable for each set.
  for (const ValGroup& loop_group : id_model_.idGraph(IdMappingMode::LOOP)
                                        .disjointValSets()
                                        .disjointSets()) {
    // TODO: halo loop not considered
    // TODO: double buffering not considered

    // First allocate thread and grid parallel indices:
    //  The validation pass will check that the parallel bindings within the
    //  loop nodes are consistent so all the loops within this disjoint set
    //  will be realized implicitly using parallel index variables.
    if (auto result = std::find_if(
            loop_group->begin(),
            loop_group->end(),
            [](Val* val) {
              return val->as<IterDomain>()->isThread();
            });
        result != loop_group->end()) {
      auto ptype = (*result)->as<IterDomain>()->getParallelType();
      NVF_ERROR(std::all_of(
          loop_group->begin(), loop_group->end(), [ptype](Val* val) {
            auto this_ptype = val->as<IterDomain>()->getParallelType();
            return this_ptype == ParallelType::Serial ||
                this_ptype == ptype;
          }),
                "Inconsistent parallelization detected with loop group of: ",
                nvfuser::toString(loop_group));

      loop_index_map_[loop_group] =
          NamedScalar::getParallelIndex(ptype);
      continue;
    }

    // The device paralle type is not included in "isThread". We don't
    // allocate any index variable for device-parallel domains.
    if (auto result = std::find_if(
            loop_group->begin(),
            loop_group->end(),
            [](Val* val) { return val->as<IterDomain>()->isDeviceDim(); });
        result != loop_group->vector().end()) {
      loop_index_map_[loop_group] = fusion->zeroVal();
      continue;
    }

    // All loops in this set are non-parallel, non-concretized broadcast
    //  iterdomains, their "index variable" should be zero.
    if (std::all_of(
            loop_group->begin(),
            loop_group->end(),
            [](Val* val) { return val->as<IterDomain>()->isBroadcast(); })) {
      loop_index_map_[loop_group] = fusion->zeroVal();
      continue;
    }

    // TODO: Support double-buffered loops. See
    // ComputeAtMap::allocateIndexVariables.

    // Everything now should be serial concrete loops,
    //   we just allocate a loop index integer for each set of loops.
    loop_index_map_[loop_group] =
        IrBuilder::create<Val>(DataType::Index);
  }
}

// 1. Find the loop domains
// 2. Find the index domains
// 3. Find the path from the loop domains to the allocation domains
// 4. Set the initial index vals
// 5. Propagate the initial indices of the loop domains to the index
// domains
Val* Indexing::getIndex(TensorView* tv, Expr* expr) {
  const ValGraph& exact_graph = id_model_.idGraph(IdMappingMode::EXACT);

  bool as_consumer = std::find(
      expr->outputs().begin(),
      expr->outputs().end(),
      tv) != expr->outputs().end();

  std::cerr << "getIndex of "
            << tv->toString()
            << " as " << (as_consumer ? "consumer" : "producer")
            << " in " << expr->toString();

  auto loop_domains = getLoopDomains(tv, expr, id_model_);

  std::cerr << "Loop domains: " << toDelimitedString(loop_domains) << std::endl;

  auto index_domains = getIndexDomains(tv, expr, id_model_);

  std::cerr << "Index domains: " << toDelimitedString(index_domains)
            << std::endl;

  auto indexing_path = getExprsBetween(
      loop_domains, index_domains, exact_graph);

  std::cerr << "Indexing path:\n";
  for (const auto& expr_group : indexing_path) {
    Expr* expr = expr_group->front();
    std::cerr << expr->toString();
  }
  std::cerr << "--- path done ---\n";

  // Map from exact groups to initial index Vals
  std::unordered_map<ValGroup, Val*> initial_index_map;

  // loop_index_map_ is a map on the loop graph. For index
  // propagation, need a map for the exact graph

  for (IterDomain* loop_id : loop_domains) {
    const auto& loop_group =
        id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_id);
    auto loop_index_map_it = loop_index_map_.find(loop_group);
    NVF_ERROR(loop_index_map_it != loop_index_map_.end());
    const auto& exact_group = exact_graph.toGroup(loop_id);
    NVF_ERROR(
        initial_index_map.emplace(exact_group, loop_index_map_it->second).second);
  }

  IndexCompute index_compute(
      id_model_.idGraph(IdMappingMode::EXACT), initial_index_map);

  for (const auto& expr_group : indexing_path) {
    index_compute.propagate(expr_group);
  }

  for (const auto& index_domain : index_domains) {
    auto idx = index_compute.getIndex(index_domain);
    NVF_ERROR(
        idx != nullptr,
        "Index not found for ",
        index_domain->toString());
    std::cerr << "Index of " << index_domain->toString() << ": "
              << idx->toInlineString()
              << std::endl;
  }

  return nullptr;
}

} // namespace nvfuser

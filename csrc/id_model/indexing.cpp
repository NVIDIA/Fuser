// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/indexing.h>
#include <id_model/to_string.h>
#include <id_model/utils.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <val_graph_visitor.h>

#include <fstream>

namespace nvfuser {

namespace {

class IndexingTraversal : public ValGraphBFS {
 public:
  IndexingTraversal(
      const ValGraph& graph,
      std::vector<GroupType> from_groups,
      std::vector<GroupType> to_groups)
      : ValGraphBFS(graph, from_groups, to_groups) {}

  virtual ~IndexingTraversal() = default;

 protected:
  using ValGraphBFS::isVisited;

  bool isVisited(const GroupType& group) const override {
    if (const ValGroup* vg = std::get_if<ValGroup>(&group);
        vg != nullptr && (*vg)->front()->as<IterDomain>()->isBroadcast()) {
      return true;
    }
    return ValGraphBFS::isVisited(group);
  }
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
      id->toString());

  return loop_promotion_map_it->second;
}

std::vector<IterDomain*> getLoopDomains(Expr* expr, const IdModel& id_model) {
  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);

  // Assume consumer-based indexing. Needs to revisit for ops like
  // scatter
  auto loop_domains = ir_utils::getTvOutput(expr)->getLeafDomain();

  for (auto& loop_id : loop_domains) {
    NVF_ERROR(loop_graph.hasGroup(loop_id));

    auto promotion_id = getLoopPromotion(loop_id, id_model);

    loop_id = promotion_id;
  }

  return loop_domains;
}

bool isAllocated(IterDomain* id, TensorView* tv) {
  return ir_utils::isShared(tv->getMemoryType(), id->getParallelType()) &&
      !id->isBroadcast() && !id->isReduction();
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

std::pair<std::vector<IterDomain*>, std::vector<Val*>> getIndexDomains(
    TensorView* tv,
    Expr* expr,
    const IdModel& id_model) {
  // TODO: Contig merged indexing

  std::vector<IterDomain*> index_domains;
  std::vector<std::optional<bool>> contiguity;

  // Ignore allocation of non-global tensors for now
  if (tv->hasAllocation() && tv->getMemoryType() == MemoryType::Global) {
    VERBOSE() << "Tv has allocation of " << tv->toString() << ", "
              << toDelimitedString(tv->getAllocationDomain()) << std::endl;
    index_domains = tv->getAllocationDomain();
    contiguity = tv->domain()->contiguity();
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
    } else if (tv->getMemoryType() == MemoryType::Shared) {
      for (const auto i : c10::irange(tv->nDims())) {
        auto leaf_id = tv->axis(i);
        std::cerr << "Smem leaf domain: " << leaf_id->toString() << " of "
                  << tv->toString() << std::endl;
        auto pt = leaf_id->getParallelType();
        if (isParallelTypeDeviceDim(pt) || isParallelTypeBlockDim(pt)) {
          continue;
        }
        if (i < tv->getComputeAtPosition() && !isParallelTypeThreadDim(pt)) {
          continue;
        }
        index_domains.push_back(leaf_id);
      }
      contiguity = std::vector<std::optional<bool>>(index_domains.size(), true);
    } else {
      index_domains = {
          tv->getLeafDomain().begin() + tv->getComputeAtPosition(),
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
                       tv->getLeafDomain().begin(),
                       tv->getLeafDomain().end(),
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
    const ValGraph& exact_graph) {
  const ValGroups loop_domain_groups = exact_graph.toGroups(loop_domains);
  const ValGroups index_domain_groups = exact_graph.toGroups(index_domains);

  VERBOSE() << "getExprsBetween: loop: " << nvfuser::toString(loop_domains)
            << ", index: " << nvfuser::toString(index_domains) << std::endl;

  IndexingTraversal traversal(
      exact_graph,
      {loop_domain_groups.vector().begin(), loop_domain_groups.vector().end()},
      {index_domain_groups.vector().begin(),
       index_domain_groups.vector().end()});

  traversal.traverse();

  const ExprGroups exprs = traversal.getShortestExprPath();

  // const ExprGroups exprs = ValGraphBFS::getExprsBetweenVals(
  // exact_graph, loop_domain_groups, index_domain_groups);

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

  bool hasIndex(IterDomain* id) const;

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

bool IndexCompute::hasIndex(IterDomain* id) const {
  // If it's a broadcast, its index is always zero.
  if (id->isBroadcast()) {
    return true;
  }
  const ValGroup& id_group = exact_graph_.toGroup(id);
  return index_map_.find(id_group) != index_map_.end();
}

Val* IndexCompute::getIndex(IterDomain* id) const {
  // If it's a broadcast, its index is always zero.
  if (id->isBroadcast()) {
    return id->fusion()->zeroVal();
  }
  const ValGroup& id_group = exact_graph_.toGroup(id);
  auto it = index_map_.find(id_group);
  NVF_ERROR(it != index_map_.end(), "Index not found: ", id->toString());
  return it->second;
}

void IndexCompute::setIndex(IterDomain* id, Val* idx) {
  const ValGroup& id_group = exact_graph_.toGroup(id);
  NVF_ERROR(
      index_map_.emplace(id_group, idx).second,
      "Index already set: ",
      id->toString(),
      ". Preious: ",
      getIndex(id)->toString(), " (", getIndex(id)->toInlineString(),
      "). New: ",
      idx->toString(), ", (", idx->toInlineString(), ")");
}

bool IndexCompute::isForward(Expr* expr) const {
  bool ready = true;
  for (const auto inp : ir_utils::filterByType<IterDomain>(expr->inputs())) {
    if (!hasIndex(inp)) {
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

void IndexCompute::handle(Split* split) {
  VERBOSE() << "IndexCompute handle: " << split->toString();

  const bool is_forward = isForward(split);

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

void IndexCompute::handle(Merge* merge) {
  VERBOSE() << "IndexCompute handle: " << merge->toString();

  const bool is_forward = isForward(merge);

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

TensorIndexer::TensorIndexer(const IdModel& id_model) : id_model_(id_model) {
  buildLoopIndexMap();
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
        GpuLower::current()
            ->commonScalarMap()
            .hoistScalar(leaf_id->extent(), {})
            ->isOneInt()) {
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

// 1. Find the loop domains
// 2. Find the index domains
// 3. Find the path from the loop domains to the allocation domains
// 4. Set the initial index vals
// 5. Propagate the initial indices of the loop domains to the index
// domains
Val* TensorIndexer::getIndex(TensorView* tv, Expr* expr) {
  const ValGraph& exact_graph = id_model_.idGraph(IdMappingMode::EXACT);

  bool as_consumer =
      std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
      expr->outputs().end();

  VERBOSE() << "getIndex of " << tv->toString() << " as "
            << (as_consumer ? "consumer" : "producer") << std::endl;

  auto loop_domains = getLoopDomains(expr, id_model_);

  VERBOSE() << "Loop domains: " << toDelimitedString(loop_domains) << std::endl;

  const auto [index_domains, strides] = getIndexDomains(tv, expr, id_model_);

  VERBOSE() << "Index domains: " << toDelimitedString(index_domains)
            << std::endl;
  std::stringstream ss;
  ss << "Strides:";
  for (const auto& stride : strides) {
    ss << ", " << stride->toInlineString();
  }
  ss << std::endl;
  VERBOSE() << ss.str();

  auto indexing_path =
      getExprsBetween(loop_domains, index_domains, exact_graph);

  VERBOSE() << "Indexing path:\n";
  for (const auto& expr_group : indexing_path) {
    Expr* expr = expr_group->front();
    VERBOSE() << expr->toString();
  }
  VERBOSE() << "--- path done ---\n";

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
    NVF_ERROR(initial_index_map.emplace(exact_group, loop_index_map_it->second)
                  .second);
  }

  IndexCompute index_compute(
      id_model_.idGraph(IdMappingMode::EXACT), initial_index_map);

  for (const auto& expr_group : indexing_path) {
    index_compute.propagate(expr_group);
  }

  Val* index = tv->fusion()->zeroVal();
  for (const auto i : c10::irange(index_domains.size())) {
    auto index_domain = index_domains.at(i);
    auto idx = index_compute.getIndex(index_domain);
    NVF_ERROR(idx != nullptr, "Index not found for ", index_domain->toString());
    VERBOSE() << "Index of " << index_domain->toString() << ": "
              << idx->toInlineString() << std::endl;

    index = SimplifyingIrBuilder::addExpr(
        index, SimplifyingIrBuilder::mulExpr(idx, strides.at(i)));
  }

  VERBOSE() << "Index: " << index->toInlineString() << std::endl;

  return index;
}

bool TensorIndexer::isSupported(Fusion* fusion) {
  const auto all_tvs = ir_utils::allTvs(fusion);

  bool supported = true;
  std::stringstream reason;

  for (const auto& tv : all_tvs) {
    for (const auto& id : ir_utils::allIDsOf(tv)) {
      if (id->getParallelType() == ParallelType::MisalignedVectorize) {
        supported = false;
        reason << "MialignedVectorize is used: " << id->toString();
        break;
      } else if (auto resize = dynamic_cast<Resize*>(id->definition())) {
        supported = false;
        reason << "Resize not supported: " << resize->toString();
        break;
      } else if (auto swizzle = dynamic_cast<Swizzle*>(id->definition())) {
        supported = false;
        reason << "Swizzle not supported: " << swizzle->toString();
        break;
      } else if (auto swizzle2d = dynamic_cast<Swizzle2D*>(id->definition())) {
        supported = false;
        reason << "Swizzle2D not supported: " << swizzle2d->toString();
        break;
      }
    }

    // Transpose not working. Disable it for now.
    if (tv->hasRFactor() && tv->getRootDomain() != tv->getRFactorDomain()) {
      std::unordered_set<IterDomain*> root_set(
          tv->getRootDomain().begin(), tv->getRootDomain().end());
      std::unordered_set<IterDomain*> rf_set(
          tv->getRFactorDomain().begin(), tv->getRFactorDomain().end());
      if (root_set == rf_set) {
        // Root and rfactor are just reordered, meaning this tv is transposed
        supported = false;
        reason << "Transpose not supported: " << tv->definition()->toString();
        break;
      }
    }
  }

  if (!supported) {
    std::cerr << "TensorIndexer disabled due to: " << reason.str() << std::endl;
    return false;
  } else {
    return true;
  }
}

} // namespace nvfuser

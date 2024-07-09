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
#include <id_model/contiguity.h>
#include <id_model/indexing.h>
#include <id_model/to_string.h>
#include <id_model/utils.h>
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
bool isPartitionedLoop(const TensorView* tv, IterDomain* id) {
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
bool mayRequireAllocation(const TensorView* tv, IterDomain* id) {
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

std::vector<ForLoop*> getMaxPathLoops(const std::vector<ForLoop*>& for_loops) {
  std::vector<ForLoop*> unswitched_domains;

  bool within_unswitch = false;

  for (const auto fl : for_loops) {
    auto parallel_type = fl->iter_domain()->getParallelType();

    if (parallel_type == ParallelType::Unswitch ||
        parallel_type == ParallelType::Unroll) {
      within_unswitch = true;
    }

    // Don't unswitch threaded loops even when unswitched
    if (fl->iter_domain()->isThread() ||
        (fl->iter_domain()->getParallelType() != ParallelType::Vectorize &&
         !within_unswitch && !predicateAtEnd(fl))) {
      continue;
    } else {
      unswitched_domains.push_back(fl);
    }
  }

  return unswitched_domains;
}

// TODO: Use this from getPredicateIndexReplacementMap
std::unordered_set<ValGroup> getMaxPathLoopDomains(
    TensorView* consumer_tv,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph,
    const ValGraph& traversal_graph) {
  auto unswitched_loops = getMaxPathLoops(for_loops);
  std::unordered_set<ValGroup> max_path_loop_domains;

  for (auto loop_domain : consumer_tv->getLoopDomain()) {
    const auto& loop_group = loop_graph.toGroup(loop_domain);
    auto it = std::find_if(
        unswitched_loops.begin(),
        unswitched_loops.end(),
        [&loop_group](ForLoop* fl) -> bool {
          return loop_group->has(fl->iter_domain());
        });
    if (it != unswitched_loops.end()) {
      max_path_loop_domains.emplace(traversal_graph.toGroup(loop_domain));
    }
  }

  return max_path_loop_domains;
}

// BFS traversal for indexing. The only difference with the default
// ValGraphBFS is that for indexing there must be a special care taken
// when resize is involved since there can be multiple paths and
// there's only one correct path. Specifically, any resize expr group
// node must appear in the root-logical path of the consumer
// tensor. Otherwise, resize nodes should be ignored. See
// IndexingTest.ResizePath for a concret example.
class IndexingTraversal : public ValGraphBFS {
 public:
  IndexingTraversal(
      const Expr* expr,
      const ValGraph& graph,
      std::vector<GroupType> from_groups,
      std::vector<GroupType> to_groups)
      : ValGraphBFS(graph, from_groups, to_groups) {
    auto consumer_tv = ir_utils::getTvOutput(expr);
    NVF_ERROR(consumer_tv != nullptr);
    if (consumer_tv->hasRoot()) {
      // Remember the resize exprs appearing in the consumer
      // tensro. These resize exprs are the only ones that should be
      // valid to visit when indexing the inputs and outputs of the expr
      auto root_to_logical_exprs = StmtSort::getExprsBetween(
          {consumer_tv->getRootDomain().begin(),
           consumer_tv->getRootDomain().end()},
          {consumer_tv->getLogicalDomain().begin(),
           consumer_tv->getLogicalDomain().end()});
      for (Expr* root_to_logical_expr : root_to_logical_exprs) {
        if (auto resize = dynamic_cast<Resize*>(root_to_logical_expr)) {
          resize_paths_.insert(resize);
        }
      }
    }
  }

  virtual ~IndexingTraversal() = default;

  static ExprPath getExprsBetween(
      const Expr* expr,
      const ValGraph& graph,
      const ValGroups& from_groups,
      const ValGroups& to_groups) {
    IndexingTraversal traversal(
        expr,
        graph,
        {from_groups.vector().begin(), from_groups.vector().end()},
        {to_groups.vector().begin(), to_groups.vector().end()});
    traversal.traverse();
    return traversal.getShortestExprPath();
  }

  using ValGraphBFS::isVisited;

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
        // This resize node should never be traversed for indexing of
        // the given expr
        return true;
      }
    }
    return false;
  }

 private:
  std::unordered_set<Resize*> resize_paths_;
};

// Currently it's only Shared or Local but Global can be the case
// too.
bool isAllocationBasedOnLeaf(TensorView* tv) {
  return tv->getMemoryType() == MemoryType::Shared ||
      tv->getMemoryType() == MemoryType::Local;
}

// Similar to IndexCompute but adapted for the graph-based indexing
class IdGraphIndexCompute : public OptOutDispatch {
 public:
  IdGraphIndexCompute(
      const ValGraph& traversal_graph,
      std::unordered_map<ValGroup, Val*> initial_index_map,
      const std::unordered_set<ValGroup>& max_path_domains)
      : traversal_graph_(traversal_graph),
        index_map_(std::move(initial_index_map)),
        max_path_domains_(max_path_domains) {}

  // Propagate the index map through a given expr of a specified
  // direction.
  void propagate(const ExprGroup& expr_group, Direction direction) {
    NVF_ERROR(!expr_group->empty());
    // Propagate max path domains
    propagateMaxPathDomains(expr_group, direction);
    // This looks a little ugly but the dispatch interface doesn't
    // have a way to pass arguments
    current_direction_ = direction;
    dispatch(expr_group->front());
    current_direction_ = Direction::Undefined;
  }

  const std::unordered_map<ValGroup, Val*>& indexMap() const {
    return index_map_;
  }

 private:
  using OptOutDispatch::handle;

  void handle(Split* split) override;

  void handle(Merge* merge) override;

  void handle(Swizzle* swizzle) override;

  void handle(Resize* resize) override;

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

  bool isInMaxPath(IterDomain* id) const {
    const auto& id_group = traversal_graph_.toGroup(id);
    return max_path_domains_.find(id_group) != max_path_domains_.end();
  }

  void propagateMaxPathDomains(
      const ExprGroup& expr_group,
      Direction direction) {
    const auto inputs = direction == Direction::Forward
        ? traversal_graph_.inputGroups(expr_group)
        : traversal_graph_.outputGroups(expr_group);

    if (std::any_of(
            inputs.begin(), inputs.end(), [&](const ValGroup& input) -> bool {
              return max_path_domains_.find(input) != max_path_domains_.end();
            })) {
      const auto outputs = direction == Direction::Forward
          ? traversal_graph_.outputGroups(expr_group)
          : traversal_graph_.inputGroups(expr_group);
      max_path_domains_.insert(outputs.begin(), outputs.end());
    }
  }

 private:
  const ValGraph& traversal_graph_;
  std::unordered_map<ValGroup, Val*> index_map_;
  Direction current_direction_ = Direction::Undefined;
  std::unordered_set<ValGroup> max_path_domains_;
};

bool IdGraphIndexCompute::isForward(Expr* expr) const {
  return current_direction_ == Direction::Forward;
}

void IdGraphIndexCompute::handle(Split* split) {
  const bool is_forward = isForward(split);

  VERBOSE() << "IdGraphIndexCompute handle (" << (is_forward ? "fwd" : "bwd")
            << "): " << split->toString();

  auto inner_extent = split->inner()->extent();

  if (is_forward) {
    auto in_idx = getIndex(split->in());
    auto outer_idx = SimplifyingIrBuilder::divExpr(in_idx, inner_extent);
    Val* inner_idx = nullptr;
    if (isInMaxPath(split->in())) {
      inner_idx = SimplifyingIrBuilder::subExpr(
          inner_extent, in_idx->fusion()->oneVal());
    } else {
      inner_idx = SimplifyingIrBuilder::modExpr(in_idx, inner_extent);
    }
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

  VERBOSE() << "IdGraphIndexCompute handle (" << (is_forward ? "fwd" : "bwd")
            << "): " << merge->toString();

  // TODO: use getMaybeExpandedExtent?
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
    Val* inner_idx = nullptr;
    // TODO: This is a safe but conservative workaround. See the old
    // IndexCompute for optimization
    // Leave it for now. Revisit after contig indexing. See if how
    // it's impacting. Maybe just fine to look at if all subsequent
    // depedent splits are divisible. That shoud be the case of the
    // transpose.
    if (isInMaxPath(merge->out())) {
      VERBOSE() << "Taking max path: " << merge->toString();
      inner_idx = SimplifyingIrBuilder::subExpr(
          inner_ext, inner_ext->fusion()->oneVal());
    } else {
      inner_idx = SimplifyingIrBuilder::modExpr(out_idx, inner_ext);
    }
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

} // namespace

TensorIndexer::TensorIndexer(IdModel& id_model)
    : id_model_(id_model), concrete_info_(id_model_.fusion()) {
  buildLoopIndexMap();

  if (getenv("DOT")) {
    std::ofstream ofs("exact_graph.dot", std::ofstream::trunc);
    auto dot_string = ValGraphDotPrinter::getString(
        id_model_.idGraph(IdMappingMode::ALMOSTEXACT));
    std::cerr << dot_string << std::endl;
    ofs << dot_string;
    ofs.close();
  }

  const auto& non_divisible_split_info =
      GpuLower::current()->nonDivisibleSplitInfo();
  for (const auto& [tv, splits] :
       non_divisible_split_info.splitsToPredicate()) {
    VERBOSE() << "Splits to predicate of tensor: " << tv->toString() << "\n";
    for (const auto split : splits) {
      VERBOSE() << "\t" << split->toString();
    }
  }
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
      tv->hasReduction() ? tv->getMaybeRootDomain() : tv->getLogicalDomain();

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

ForLoop* getForLoop(
    IterDomain* loop_id,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph) {
  auto it = std::find_if(
      for_loops.begin(), for_loops.end(), [&](ForLoop* for_loop) -> bool {
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

void TensorIndexer::buildLoopIndexMap() {
  if (id_model_.empty()) {
    return;
  }

  Fusion* fusion = id_model_.fusion();
  FusionGuard fg(fusion);

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
            VERBOSE() << "Trying to find index val for " << id->toString()
                      << std::endl;
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
      VERBOSE() << "Loop index map: " << nvfuser::toString(loop_group) << " -> "
                << loop_index->toInlineString() << std::endl;
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
    const std::vector<IterDomain*>& loop_domains,
    const std::vector<ForLoop*>& for_loops) const {
  std::unordered_map<ValGroup, Val*> initial_index_map;

  // For a given list of the loop domains, assign its corresponding
  // index Val.
  for (IterDomain* loop_id : loop_domains) {
    Val* loop_index = getLoopIndex(loop_id);
    const auto& almost_exact_group = traversalGraph().toGroup(loop_id);
    VERBOSE() << "Setting initial index. " << loop_id->toString() << ", "
              << nvfuser::toString(almost_exact_group) << ", "
              << loop_index->toInlineString() << std::endl;

    if (initial_index_map.find(almost_exact_group) != initial_index_map.end()) {
      // Initial index already set. This can happen as this is an
      // almost-exact group. It should be just size-1 domain.
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

    // This is a WAR for circular buffering. The loop graph is
    // designed to represent each loop and each loop group is supposed
    // to have a one-to-one relationship with each loop. However, for
    // circular buffering, this assumption is broken as we are using
    // the same iter domain for the prologue, main and epilogue
    // loops. Those loops should have distinctive loop groups, but for
    // now, here's a workaround to assign a correct loop index
    {
      const IdModel& id_model = GpuLower::current()->idModel();
      const ValGroup& loop_group =
          id_model.idGraph(IdMappingMode::LOOP).toGroup(loop_id);
      auto loop_it =
          std::find_if(for_loops.begin(), for_loops.end(), [&](auto fl) {
            return loop_group->has(fl->iter_domain());
          });
      // It's possible that there's no corresponding ForLoop, i.e,
      // when this loop ID corresponds to a reduction
      // domain and we are building a map for the expression to
      // initialize the reduction buffer. For such a case, this WAR is
      // irrelevant.
      if (loop_it != for_loops.end()) {
        ForLoop* fl = *loop_it;
        if (fl->circularBufferLoopStage() == CircularBufferLoopStage::Prolog ||
            fl->circularBufferLoopStage() == CircularBufferLoopStage::Epilog) {
          loop_index = fl->indexOrStartIfTrivial();
        }
      }
    }

    initial_index_map.emplace(almost_exact_group, loop_index);
  }

  return initial_index_map;
}

std::vector<Val*> TensorIndexer::getIndexFor(
    const Expr* expr,
    bool as_consumer,
    const std::vector<ForLoop*>& for_loops,
    const ValGroups& index_groups) const {
  const auto& info = computeIndex(expr, for_loops, index_groups, false, false);
  const auto& replacement_map = getIndexReplacementMap(
      expr, as_consumer, info.loop_domains, for_loops, info.index_map);

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

std::pair<std::deque<ValGroup>, std::deque<Val*>> TensorIndexer::
    getContigDomainsAndStrides(
        const std::vector<IterDomain*>& allocation_domains,
        const std::vector<Val*>& strides,
        const std::vector<bool>& contiguity,
        const ExprPath& traversal_path) const {
  const std::unordered_map<IterDomain*, ValGroup>& contig_domains =
      getContigDomains(
          allocation_domains,
          contiguity,
          reverse(traversal_path),
          traversalGraph(),
          concrete_info_,
          false);

  // Find contiguous domains to index
  std::unordered_set<ValGroup> already_indexed_domains;
  std::deque<ValGroup> contig_alloc_groups;
  std::deque<Val*> contig_strides;
  for (const auto i : c10::irange(allocation_domains.size())) {
    // Traverse back from the innermost domains so that the right
    // stride val is picked up for each contiguous domain
    auto i1 = allocation_domains.size() - 1 - i;
    IterDomain* allocation_domain = allocation_domains.at(i1);
    auto contig_domains_it = contig_domains.find(allocation_domain);
    NVF_ERROR(
        contig_domains_it != contig_domains.end(),
        "No contig domain mapping found for ",
        allocation_domain->toString());

    const ValGroup& contig_domain_group = contig_domains_it->second;
    if (already_indexed_domains.find(contig_domain_group) !=
        already_indexed_domains.end()) {
      VERBOSE() << "Already indexed: " << allocation_domain->toString()
                << std::endl;
      continue;
    }
    already_indexed_domains.emplace(contig_domain_group);

    if (!contig_domain_group->has(allocation_domain)) {
      VERBOSE() << "Contig indexing: "
                << contig_domain_group->front()->toString() << " instead of "
                << allocation_domain->toString() << std::endl;
    } else {
      VERBOSE() << "Non contig indexing: " << allocation_domain->toString()
                << std::endl;
    }

    VERBOSE() << "Stride: " << strides.at(i1)->toInlineString() << std::endl;

    contig_alloc_groups.push_front(contig_domain_group);
    contig_strides.push_front(strides.at(i1));
  }

  return {contig_alloc_groups, contig_strides};
}

Val* TensorIndexer::getLinearIndex(
    TensorView* tv,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops) const {
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

  bool as_consumer =
      std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
      expr->outputs().end();

  VERBOSE() << "getLinearIndex of " << tv->toString() << " as "
            << (as_consumer ? "consumer" : "producer") << " in "
            << expr->toString() << std::endl;

  const auto alloc_info = getIndexingAllocationInfo(tv);

  VERBOSE() << "Allocation domains: " << toDelimitedString(alloc_info.domains)
            << std::endl;

  const auto& index_info = computeIndex(
      expr,
      for_loops,
      traversalGraph().toGroups(alloc_info.domains),
      false,
      false);
  const auto& index_map = index_info.index_map;

  // ValGroups may not be suitable here. It should be fine currently,
  // but if we ever want to support self-mapped allocation domains, we
  // should not deduplicate the domain groups. Use deque as that's
  // convenient for getContigDomainsAndStrides.
  std::deque<ValGroup> contig_alloc_groups;
  std::deque<Val*> contig_strides;

  if (enableContigIndexing()) {
    VERBOSE() << "Contig indexing enabled\n";
    const auto& contig_alloc_strides = getContigDomainsAndStrides(
        alloc_info.domains,
        alloc_info.strides,
        alloc_info.contiguity,
        index_info.traversal_path);
    contig_alloc_groups = contig_alloc_strides.first;
    contig_strides = contig_alloc_strides.second;
  } else {
    VERBOSE() << "Contig indexing disabled\n";
    std::transform(
        alloc_info.domains.begin(),
        alloc_info.domains.end(),
        std::back_inserter(contig_alloc_groups),
        [&](IterDomain* allocation_domain) {
          return traversalGraph().toGroup(allocation_domain);
        });
    contig_strides = {alloc_info.strides.begin(), alloc_info.strides.end()};
  }
  const auto& replacement_map = getIndexReplacementMap(
      expr, as_consumer, index_info.loop_domains, for_loops, index_map);

  // Linearize the indices with strides.
  Val* linear_index = tv->fusion()->zeroVal();
  for (const auto i : c10::irange(contig_alloc_groups.size())) {
    const auto& contig_domain_group = contig_alloc_groups.at(i);
    auto idx_it = index_map.find(contig_domain_group);
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        contig_domain_group->front()->toString());
    Val* idx = idx_it->second;
    VERBOSE() << "Index of " << contig_domain_group->front()->toString() << ": "
              << idx->toInlineString() << std::endl;
    Val* replaced_idx = ir_utils::replaceValRecursively(idx, replacement_map);

    linear_index = SimplifyingIrBuilder::addExpr(
        linear_index,
        SimplifyingIrBuilder::mulExpr(replaced_idx, contig_strides.at(i)));
  }

  // If a tensor is circular buffered, it also requires indexing of
  // the circular buffer itself
  if (tv->isCircularBuffered() && GpuLower::hasCurrent()) {
    auto adjusted_index =
        adjustIndexToSwitchBuffer(tv, as_consumer, for_loops, linear_index);
    linear_index = adjusted_index;
  }

  VERBOSE() << "Final index: " << linear_index->toInlineString() << std::endl;
  return linear_index;
}

// Get the loop domains of a given expr, which are (potentially
// promoted) loop domains of the consumer tensor.
std::vector<IterDomain*> TensorIndexer::getLoopDomains(const Expr* expr) const {
  // Assume consumer-based indexing. Needs to revisit for ops like
  // scatter
  auto loop_domains = ir_utils::getTvOutput(expr)->getLoopDomain();

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
    loop_id = getLoopPromotion(loop_id, id_model_);
  }

  return loop_domains;
}

IndexingInfo TensorIndexer::computeIndex(
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops,
    const ValGroups& index_groups,
    bool is_predicate,
    bool is_unswitch) const {
  VERBOSE() << "computeIndex of " << expr->toString() << std::endl;

  const auto loop_domains = getLoopDomains(expr);
  VERBOSE() << "Loop domains: " << toDelimitedString(loop_domains) << std::endl;

  const ValGroups loop_groups = traversalGraph().toGroups(loop_domains);

  auto traversal_path = IndexingTraversal::getExprsBetween(
      expr, traversalGraph(), loop_groups, index_groups);

  VERBOSE() << "Indexing path:\n";
  for (const auto& [expr_group, direction] : traversal_path) {
    Expr* expr = expr_group->front();
    VERBOSE() << direction << " " << expr->toString();
  }
  VERBOSE() << "--- path done ---\n";

  const std::unordered_map<ValGroup, Val*> initial_index_map =
      getInitialIndexMap(loop_domains, for_loops);

  const std::unordered_set<ValGroup> max_path_loop_domains = is_unswitch
      ? getMaxPathLoopDomains(
            ir_utils::getTvOutput(expr),
            for_loops,
            id_model_.idGraph(IdMappingMode::LOOP),
            traversalGraph())
      : std::unordered_set<ValGroup>{};

  IdGraphIndexCompute index_compute(
      traversalGraph(), initial_index_map, max_path_loop_domains);

  for (const auto& [expr_group, direction] : traversal_path) {
    index_compute.propagate(expr_group, direction);
  }

  IndexingInfo info{loop_domains, traversal_path, index_compute.indexMap()};
  return info;
}

std::unordered_map<Val*, Val*> TensorIndexer::getIndexReplacementMap(
    const Expr* expr,
    bool as_consumer,
    const std::vector<IterDomain*>& loop_domains,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<ValGroup, Val*>& index_map) const {
  std::unordered_map<Val*, Val*> replacement_map;

  for (const auto loop_id : loop_domains) {
    const ValGroup& loop_group = traversalGraph().toGroup(loop_id);
    auto index_it = index_map.find(loop_group);
    NVF_ERROR(index_it != index_map.end());
    Val* cur_index = index_it->second;

    Val* replacement_index = nullptr;
    // Replace the index of a vectorized/bulk domain with zero. Note that
    // vectorized domains may need to use N-1, where N is the extent
    // of the domain, for predication, so the replacement is not
    // always done with zero.
    if (loop_id->getParallelType() == ParallelType::Vectorize ||
        loop_id->getParallelType() == ParallelType::Bulk) {
      replacement_index = loop_id->fusion()->zeroVal();
    } else {
      ForLoop* for_loop = getForLoop(
          loop_id, for_loops, id_model_.idGraph(IdMappingMode::LOOP));

      // Even when the iter-domain is not size-1, the actual for-loop
      // can be (e.g., for double buffering).
      if (for_loop != nullptr) {
        if (for_loop->isTrivial()) {
          VERBOSE() << "Replacing a loop index with a loop start val: "
                    << for_loop->start()->toInlineString()
                    << ", loop_id: " << loop_id->toString() << std::endl;
          replacement_index = for_loop->start();
        }

        // TODO: Make it mandatory to have a GpuLower
        if (GpuLower::hasCurrent() && !as_consumer) {
          replacement_index = adjustProducerLoopIndexForCircularBuffering(
              expr,
              for_loop,
              replacement_index != nullptr ? replacement_index : cur_index);
        }
      }
    }

    if (replacement_index == nullptr || cur_index == nullptr) {
      continue;
    }

    replacement_map.emplace(cur_index, replacement_index);
  }

  return replacement_map;
}

// TODO: Drop the tv parameter. It's only for double buffering, which
// I believe should be done as a separate step after indexing
std::vector<Val*> TensorIndexer::getPerDimIndex(
    TensorView* tv,
    const std::vector<IterDomain*>& index_domains,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops) {
  const auto& traversal_graph = id_model_.idGraph(IdMappingMode::ALMOSTEXACT);

  VERBOSE() << "getPerDimIndex of " << toDelimitedString(index_domains)
            << " in " << expr->toString() << std::endl;

  const auto& index_info = computeIndex(
      expr, for_loops, traversalGraph().toGroups(index_domains), false, false);

  const auto& index_map = index_info.index_map;

  std::vector<Val*> indices;
  indices.reserve(index_domains.size());

  for (const auto i : c10::irange(index_domains.size())) {
    auto index_domain = index_domains.at(i);

    if (index_domain->isBroadcast() || index_domain->isReduction()) {
      indices.push_back(index_domain->fusion()->zeroVal());
      continue;
    }

    auto idx_it = index_map.find(traversal_graph.toGroup(index_domain));
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        index_domain->toString());
    Val* idx = idx_it->second;
    VERBOSE() << "Index of " << index_domain->toString() << ": "
              << idx->toInlineString() << std::endl;

    indices.push_back(idx);
  }

  return indices;
}

// If the for-loop is double-buffered and not prologue, the loop
// index should be advanced by one except for the double-buffered
// tensor itself
Val* TensorIndexer::adjustProducerLoopIndexForCircularBuffering(
    const Expr* expr,
    const ForLoop* for_loop,
    Val* loop_index) const {
  NVF_ERROR(for_loop != nullptr);

  auto consumer_tv = ir_utils::getTvOutput(expr);

  if (!consumer_tv->isCircularBuffered()) {
    return loop_index;
  }

  NVF_ERROR(expr->inputs().size() == 1);

  auto producer_tv = expr->input(0)->as<TensorView>();

  // Double-buffered tensor itself does not need this adjustment
  if (producer_tv->isCircularBuffered() &&
      id_model_.idGraph(IdMappingMode::LOOP)
          .disjointValSets()
          .strictAreMapped(
              getCircularBufferAxis(producer_tv), for_loop->iter_domain())) {
    return loop_index;
  }

  if (for_loop->circularBufferLoopStage() != CircularBufferLoopStage::Main &&
      for_loop->circularBufferLoopStage() != CircularBufferLoopStage::Epilog) {
    return loop_index;
  }

  const auto gpu_lower = GpuLower::current();
  NVF_ERROR(
      gpu_lower != nullptr,
      "Double buffering info of GpuLower is required but GpuLower is missing");

  auto stage_depth =
      (int64_t)GpuLower::current()->circularBufferInfo().getStageDepthFor(
          for_loop->iter_domain());

  auto adjusted_loop_index = SimplifyingIrBuilder::addExpr(
      loop_index,
      SimplifyingIrBuilder::create<Val>(stage_depth - 1L, DataType::Index));

  VERBOSE() << "Adjusted initial producer index: "
            << adjusted_loop_index->toInlineString() << std::endl;
  VERBOSE() << expr->toString();

  return adjusted_loop_index;
}

Val* TensorIndexer::adjustIndexToSwitchBuffer(
    TensorView* tv,
    bool as_consumer,
    const std::vector<ForLoop*>& for_loops,
    Val* idx) const {
  if (!tv->isCircularBuffered()) {
    return idx;
  }

  const auto gpu_lower = GpuLower::current();
  NVF_ERROR(
      gpu_lower != nullptr,
      "Double buffering info of GpuLower is required but GpuLower is missing");

  auto db_loop =
      gpu_lower->circularBufferInfo().getCircularBufferLoop(tv, for_loops);

  NVF_ERROR(db_loop != nullptr);

  // Mostly just copied from getNonGlobalConsumerStridedIndices

  bool is_epilogue =
      db_loop->circularBufferLoopStage() == CircularBufferLoopStage::Prolog;

  auto loop_index = db_loop->indexOrStartIfTrivial();

  const auto stage_depth =
      (int64_t)gpu_lower->circularBufferInfo().getStageDepthFor(
          db_loop->iter_domain());

  auto db_index_offset = loop_index;
  if (as_consumer && !is_epilogue) {
    // Read-ahead offset for consumer indexing
    db_index_offset = SimplifyingIrBuilder::addExpr(
        db_index_offset,
        SimplifyingIrBuilder::create<Val>(stage_depth - 1, DataType::Index));
  }

  // % `num_stages` not necessary in epilogue
  if (!is_epilogue) {
    db_index_offset = SimplifyingIrBuilder::modExpr(
        db_index_offset,
        SimplifyingIrBuilder::create<Val>(stage_depth, DataType::Index));
  }

  auto original_alloc_size =
      gpu_lower->circularBufferInfo().getOriginalAllocSize(tv);

  auto db_strided_index =
      SimplifyingIrBuilder::mulExpr(db_index_offset, original_alloc_size);

  auto updated_idx = SimplifyingIrBuilder::addExpr(idx, db_strided_index);
  return updated_idx;
}

std::unordered_map<Val*, Val*> TensorIndexer::getPredicateIndexReplacementMap(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    bool is_start_predicate,
    bool is_unswitch,
    const std::unordered_map<ValGroup, Val*>& index_map,
    const ValGraph& traversal_graph) const {
  std::unordered_map<Val*, Val*> replacement_map;

  auto replace_for_unswitch =
      [&](ForLoop* fl, IterDomain* loop_id, bool within_unswitch) -> Val* {
    // Don't replace thread indices even when unswitched
    if (fl->iter_domain()->isThread() ||
        (fl->iter_domain()->getParallelType() != ParallelType::Vectorize &&
         !within_unswitch && !predicateAtEnd(fl))) {
      return nullptr;
    } else {
      return is_start_predicate
          ? fl->fusion()->zeroVal()
          : SimplifyingIrBuilder::subExpr(
                fl->simplifiedStop(), fl->fusion()->oneVal());
    }
  };

  auto replace_for_double_buffering = [&](ForLoop* fl,
                                          Val* original_index) -> Val* {
    auto db_axis =
        GpuLower::current()->circularBufferInfo().getCircularBufferAxis(tv);
    if (db_axis == nullptr ||
        !id_model_.idGraph(IdMappingMode::LOOP)
             .disjointValSets()
             .strictAreMapped(fl->iter_domain(), db_axis)) {
      return nullptr;
    }

    // The prologue loop does not need to be changed
    if (fl->circularBufferLoopStage() == CircularBufferLoopStage::Prolog) {
      return nullptr;
    }

    auto stage_depth =
        (int64_t)GpuLower::current()->circularBufferInfo().getStageDepthFor(
            fl->iter_domain());
    return SimplifyingIrBuilder::addExpr(
        original_index,
        SimplifyingIrBuilder::create<Val>(stage_depth - 1L, DataType::Index));
  };

  bool within_unswitch = false;

  for (const auto fl : for_loops) {
    auto parallel_type = fl->iter_domain()->getParallelType();

    if (parallel_type == ParallelType::Unswitch ||
        parallel_type == ParallelType::Unroll) {
      within_unswitch = is_unswitch;
    }

    auto loop_id = getLoopPromotion(fl->iter_domain(), id_model_);
    NVF_ERROR(
        !loop_id->maybePartial(),
        "Partial loop not supported: ",
        fl->toString());
    auto loop_index_it = index_map.find(traversal_graph.toGroup(loop_id));
    if (loop_index_it == index_map.end()) {
      // The index map is built from the tensor loop domains. There
      // can be for-loops that are not part of this tensor, e.g, a
      // tensor inlined into a higher dimensional tensor.
      continue;
    }
    Val* loop_index = loop_index_it->second;

    // If it's already const scalar, no replacment should be necessary
    if (loop_index->isConst()) {
      continue;
    }

    Val* replacement = loop_index;

    // Trivial loop. Note that not all trivial loops should just use
    // the start index for predication. For example, a vectorized loop
    // is trivial, but its predicate should use `vec_factor - 1` as
    // its index. This is taken care after this.
    if (fl->isTrivial()) {
      replacement = fl->start();
    }

    auto unswitched_index = replace_for_unswitch(fl, loop_id, within_unswitch);
    if (unswitched_index != nullptr) {
      replacement = unswitched_index;
    }

    // Adjustment for double buffering
    auto db_index = replace_for_double_buffering(fl, replacement);
    if (db_index != nullptr) {
      replacement = db_index;
    }

    if (replacement != loop_index) {
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

std::optional<CircularBufferLoopStage> getCircularBufferLoopStage(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph) {
  auto db_axis =
      GpuLower::current()->circularBufferInfo().getCircularBufferAxis(tv);
  if (db_axis == nullptr) {
    return std::nullopt;
  }

  for (const auto fl : for_loops) {
    if (loop_graph.disjointValSets().strictAreMapped(
            fl->iter_domain(), db_axis)) {
      return fl->circularBufferLoopStage();
    }
  }

  NVF_ERROR(false, "Double-buffered loop not found for ", tv->toString());
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
        // expr should give us correct allocation info.
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
    const IdModel& id_model = GpuLower::current()->idModel();

    std::vector<IterDomain*> allocation_domains;
    std::vector<std::optional<bool>> contiguity;

    bool use_set_allocatin_domain = false;

    // Set allocation domains are not necessrily truth. Ideally, we
    // should make sure the set domain is always valid, but for now do
    // some inference if it's likely valid
    if (tv->hasAllocation()) {
      // If the tensor is global, it's likely valid
      if (tv->getMemoryType() == MemoryType::Global) {
        use_set_allocatin_domain = true;
      }
      // If the tensor is shared and swizzle or TMA is used, it should
      // be the valid domain. Also, if the allocation domain is just a
      // permutation of the loop domain, use the set allocation
      // domain. This seems to happen only with
      // AllocationDomainTest.TransposedIntermediate.
      if (tv->getMemoryType() == MemoryType::Shared &&
          (std::any_of(
               tv->getAllocationDomain().begin(),
               tv->getAllocationDomain().end(),
               [](IterDomain* allocation_domain) {
                 return dynamic_cast<Swizzle*>(
                            allocation_domain->definition()) != nullptr ||
                     allocation_domain->getParallelType() == ParallelType::Bulk;
               }) ||
           std::is_permutation(
               tv->getLoopDomain().begin(),
               tv->getLoopDomain().end(),
               tv->getAllocationDomain().begin()))) {
        if (std::none_of(
                tv->getAllocationDomain().begin(),
                tv->getAllocationDomain().end(),
                [](IterDomain* allocation_domain) {
                  return dynamic_cast<Swizzle*>(
                             allocation_domain->definition()) != nullptr ||
                      allocation_domain->getParallelType() ==
                      ParallelType::Bulk;
                })) {
          std::cerr << "Permutation allocation domain\n";
        }
        use_set_allocatin_domain = true;
      }
    }

    if (use_set_allocatin_domain) {
      allocation_domains = tv->getAllocationDomain();
      contiguity = tv->domain()->contiguity();
      NVF_ERROR(!tv->isCircularBuffered());
    } else {
      // If allocation domain is not set, assume that:
      // - Global: logical domains
      // - Local/Shared: loop domains to the right of the CA position
      if (tv->getMemoryType() == MemoryType::Global) {
        VERBOSE() << "Tv does not have allocation of " << tv->toString() << ", "
                  << toDelimitedString(tv->getMaybeAllocationDomain())
                  << std::endl;
        allocation_domains = tv->getLogicalDomain();
        contiguity = tv->domain()->contiguity();
        NVF_ERROR(!tv->isCircularBuffered());
      } else {
        int64_t allocation_pos =
            lower_utils::getAllocInformation(tv, for_loops).alloc_pos;

        if (tv->isCircularBuffered()) {
          allocation_pos = getCircularBufferAxisPosition(tv) + 1;
        }

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
        VERBOSE() << "Allocation domain reorderred: " << tv->toString()
                  << ". Oriignal: " << toDelimitedString(allocation_domains)
                  << ", reordered: "
                  << toDelimitedString(reordered_domains.value()) << std::endl;
        allocation_domains = reordered_domains.value();
        NVF_ERROR(std::all_of(contiguity.begin(), contiguity.end(), [](auto b) {
          return b.has_value() && b.value();
        }));
      }
      // WAR for transpose
      if (auto transposed_smem_alloc_dom =
              patchAllocationOfTransposedSmemTensor(
                  tv,
                  allocation_domains,
                  id_model.idGraph(IdMappingMode::EXACT));
          transposed_smem_alloc_dom.has_value()) {
        VERBOSE()
            << "Using consumer domain as the allocation domain of the shared memory producer: "
            << tv->toString() << std::endl;
        allocation_domains = transposed_smem_alloc_dom.value();
        NVF_ERROR(std::all_of(contiguity.begin(), contiguity.end(), [](auto b) {
          return b.has_value() && b.value();
        }));
        contiguity =
            std::vector<std::optional<bool>>(allocation_domains.size(), true);
      }
    }

    NVF_ERROR(allocation_domains.size() == contiguity.size());

    return {allocation_domains, contiguity};
  }

  // Get allocation info used for indexing. Loop promotion is
  // considered. Strides are also calculated.
  IndexingAllocationInfo getIndexingAllocationInfo(
      TensorView* tv,
      std::vector<IterDomain*> allocation_domains,
      std::vector<std::optional<bool>> contiguity) {
    const IdModel& id_model = GpuLower::current()->idModel();

    auto allocation_tv = tv;

    std::vector<IterDomain*> promoted_allocation_domains;
    promoted_allocation_domains.reserve(allocation_domains.size());

    // Loop promotion may affect allocations. Promotions of intermediate
    // domains may not be defined correctly. Only consider loop domains
    // for now.
    for (const auto& allocation_domain : allocation_domains) {
      bool is_loop =
          std::find(
              allocation_tv->getLoopDomain().begin(),
              allocation_tv->getLoopDomain().end(),
              allocation_domain) != allocation_tv->getLoopDomain().end();
      auto promotion_id = allocation_domain;
      // If the allocation domain is still a broadcast domain, i.e., not
      // merged with a non-broadcast domain, it should
      // not be necessary to use the promotion domain.
      // TODO: Add tests
      if (is_loop && !allocation_domain->isBroadcast()) {
        promotion_id = getLoopPromotion(allocation_domain, id_model);
      }
      promoted_allocation_domains.push_back(promotion_id);
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
    std::vector<bool> actual_contiguity;
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
      auto contig = contiguity.at(i);
      NVF_ERROR(contig.has_value());
      actual_contiguity.push_back(contig.value());
    }

    NVF_ERROR(actual_allocation_domains.size() == actual_strides.size());

    return IndexingAllocationInfo{
        actual_allocation_domains, actual_strides, actual_contiguity};
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

    if (tv->name() == 6) {
      VERBOSE() << "Ordered domains: " << toDelimitedString(ordered_domains)
                << std::endl;
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

  // Transpose with shared memory may need to change the ordering of
  // allocation domains when shared memory is used as an input to
  // vectorized stores. The transpose scheduler stages data to shared
  // memory for vectorized stores to global memory. The layout of the
  // shared memory staging buffer needs to be compatible with the
  // vectorized stores. More specifically, here's a typical pattern of
  // the transpose scheduler:
  //
  // t0_g: [I0, I1]
  // t1_l = transpose(0, 1); // [I1, I0]
  // t2_s = t1_l; // [I1, I0]
  // t3_g = t2_s; // [I1, I0]
  //
  // t0, t1, t2:
  //   split I0 by 32 -> I/32a, 32a
  //   split I1 by 32 -> I/32b, 32b
  //   merge 32a and 32b -> 32a*32b
  //   split 32a*32b by 4 -> 32a*32b/4, 4
  //  -> loop domain: [I0/32a, I1/32b, 32a*32b/4, 4]
  // t3:
  //   split I0 by 32 -> I/32a, 32a
  //   split I1 by 32 -> I/32b, 32b
  //   merge 32b and 32a -> 32b*32a
  //   split 32*32 by 4 -> 32b*32a/4, 4
  //  -> loop domain: [I0/32a, I1/32b, 32b*32a/4, 4]
  //
  // Notice that t2 has 32a*32b, whereas t3 has 32b*32a. When the innermost
  // domain of t3 is vectorized, this means that 32a must be the
  // innermost in the allocation domain of t2. However, the inferred
  // allocation domain has [..., 32a*32b/4, 4], so 32a is not the
  // innermost.
  //
  // When a given tensor is found to have this pattern, allocation
  // domains as ordered in the same way as the vectorized global
  // memory tensor are returned. In the case of the above example,
  // [32b, 32a] is returned.
  std::optional<std::vector<IterDomain*>> patchAllocationOfTransposedSmemTensor(
      const TensorView* tv,
      const std::vector<IterDomain*>& allocation_domains,
      const ValGraph& exact_graph) const {
    // First, do pattern matching to see if this tensor is a shared
    // memory tensor transpose. Pattern matching conditions include:
    //
    // - Shared memory tensor
    // - Ignore if swizzle is used as it should have the correct
    //   allocation domain
    // - BID/DID should not be used with allocation domains
    // - Consumer tensor must be a global memory tensor with vectorization
    // - There must be a merge op whose two outputs are the dominating
    //   domains of the allocation domains
    // - The consumer tensor also has a merge but with the inner and
    //   outer reversed

    if (allocation_domains.empty()) {
      return std::nullopt;
    }

    if (tv->getMemoryType() != MemoryType::Shared) {
      return std::nullopt;
    }

    // No BID/DID parallel type should be used
    if (std::any_of(
            allocation_domains.begin(),
            allocation_domains.end(),
            [](IterDomain* id) -> bool {
              return isParallelTypeDeviceDim(id->getParallelType()) ||
                  isParallelTypeBlockDim(id->getParallelType());
            })) {
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

    IterDomain* consumer_vectorized_domain = nullptr;
    if (auto it = std::find_if(
            consumer->getLoopDomain().begin(),
            consumer->getLoopDomain().end(),
            [](IterDomain* loop_id) {
              return loop_id->getParallelType() == ParallelType::Vectorize;
            });
        it != consumer->getLoopDomain().end()) {
      consumer_vectorized_domain = *it;
    } else {
      return std::nullopt;
    }

    // May be naive, but assume a simple pattern that all allocation
    // domains are derived from a merge.

    // First, find the closest merge
    auto getOriginatingMerge = [](IterDomain* id) -> Merge* {
      while (id != nullptr) {
        auto def = id->definition();
        if (auto merge = dynamic_cast<Merge*>(def)) {
          return merge;
        } else if (auto split = dynamic_cast<Split*>(def)) {
          id = split->in();
        } else {
          // Unsupported op
          return nullptr;
        }
      }
      return nullptr;
    };

    Merge* producer_common_merge =
        getOriginatingMerge(allocation_domains.front());
    if (producer_common_merge == nullptr) {
      return std::nullopt;
    }

    // Test if all allocation domains and the merge output are
    // equivalent
    auto producer_merge_dep_exprs = DependencyCheck::getAllExprsBetween(
        {producer_common_merge->out()},
        {allocation_domains.begin(), allocation_domains.end()});

    std::unordered_set<IterDomain*> equiv_domain_set(
        allocation_domains.begin(), allocation_domains.end());

    // Traverse back from the allocation domains to the merge output
    // and see if they are equivalent
    for (auto it = producer_merge_dep_exprs.rbegin();
         it != producer_merge_dep_exprs.rend();
         ++it) {
      Expr* expr = *it;
      for (auto out : expr->outputs()) {
        auto it = equiv_domain_set.find(out->as<IterDomain>());
        if (it == equiv_domain_set.end() &&
            mayRequireAllocation(tv, out->as<IterDomain>())) {
          // missing dependency
          return std::nullopt;
        }
        if (it != equiv_domain_set.end()) {
          equiv_domain_set.erase(it);
        }
      }
      for (auto input : expr->inputs()) {
        equiv_domain_set.insert(input->as<IterDomain>());
      }
    }

    // If they are equivalent, the merge output should be the only
    // remaining domain
    if (!(equiv_domain_set.size() == 1 &&
          *(equiv_domain_set.begin()) == producer_common_merge->out())) {
      // Not all allocation domains are used, meaning the merge output
      // is not equivalent to the allocation domains
      return std::nullopt;
    }

    // Look for a reverse merge in the consumer that uses the same
    // inputs but outer and inner are reversed

    IterDomain* merge_outer = producer_common_merge->outer();
    const ValGroup& merge_outer_group = exact_graph.toGroup(merge_outer);
    IterDomain* merge_inner = producer_common_merge->inner();
    const ValGroup& merge_inner_group = exact_graph.toGroup(merge_inner);

    const ExprGroups& merge_outer_uses = exact_graph.getUses(merge_outer_group);
    ExprGroup reverse_merge;
    for (const auto& merge_outer_use : merge_outer_uses) {
      Merge* merge = dynamic_cast<Merge*>(merge_outer_use->front());
      if (merge == nullptr) {
        continue;
      }
      if (exact_graph.toGroup(merge->outer()) == merge_inner_group &&
          exact_graph.toGroup(merge->inner()) == merge_outer_group) {
        reverse_merge = merge_outer_use;
        break;
      }
    }

    if (reverse_merge.get() == nullptr) {
      return std::nullopt;
    }

    ValGroup reverse_merge_output =
        exact_graph.outputGroups(reverse_merge).at(0);

    // Look for a matching merge in the consumer
    const auto consumer_all_ids = ir_utils::allIDsOf(consumer);
    IterDomain* consumer_merge_out = nullptr;
    for (auto consumer_id : consumer_all_ids) {
      if (reverse_merge_output->has(consumer_id)) {
        consumer_merge_out = consumer_id;
        break;
      }
    }

    if (consumer_merge_out == nullptr) {
      return std::nullopt;
    }

    // If there's a loop id that depends on consumer_merge_output, the
    // producer tensor needs to use the memory layout that works for
    // the vectorized store of the consumer tensor.
    if (!DependencyCheck::isDependencyOf(
            consumer_merge_out, consumer_vectorized_domain)) {
      return std::nullopt;
    }

    std::vector<IterDomain*> patched_allocation_domains{
        merge_inner, merge_outer};

    VERBOSE() << "Patching smem allocation for transpose: " << tv->toString()
              << ". "
              << "Original: " << toDelimitedString(allocation_domains)
              << ". Patched: " << toDelimitedString(patched_allocation_domains)
              << std::endl;

    return patched_allocation_domains;
  }

  std::unordered_map<TensorView*, IndexingAllocationInfo> tv_alloc_info_map;
  std::unordered_set<TensorView*> used_as_producer;
};

} // namespace

std::vector<RootPredicateInfo> TensorIndexer::getPredicates(
    TensorView* tv,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops,
    bool is_unswitch) {
  if (is_unswitch) {
    VERBOSE() << "get unswitch predicates of " << tv->toString() << " in "
              << expr->toString();
  } else {
    VERBOSE() << "get inline predicates of " << tv->toString() << " in "
              << expr->toString();
  }

  // For a double buffered tensor, use the predicate from the main
  // loop only. The prologue loop is only for the first element, so it
  // should be ignored. Double buffering may or may not create the
  // eplogue loop, but irrespective of that we can just use the
  // predicate of the main loop.
  if (is_unswitch) {
    if (auto loop_stage = getCircularBufferLoopStage(
            tv, for_loops, id_model_.idGraph(IdMappingMode::LOOP));
        loop_stage.has_value() &&
        loop_stage.value() != CircularBufferLoopStage::Main) {
      return {};
    }
  }

  const auto& traversal_graph = id_model_.idGraph(IdMappingMode::ALMOSTEXACT);
  const auto zero_val = tv->fusion()->zeroVal();

  const auto& predicate_domains = getPredicateDomains(tv, expr, id_model_);

  VERBOSE() << "Predicate domains: " << toDelimitedString(predicate_domains)
            << std::endl;

  const auto& index_info = computeIndex(
      expr,
      for_loops,
      traversalGraph().toGroups(predicate_domains),
      true,
      is_unswitch);
  const auto& index_map = index_info.index_map;

  auto replacement_map_start = getPredicateIndexReplacementMap(
      tv, for_loops, true, is_unswitch, index_map, traversal_graph);

  auto replacement_map_stop = getPredicateIndexReplacementMap(
      tv, for_loops, false, is_unswitch, index_map, traversal_graph);

  auto non_divisible_splits = getNonDivisibleConsumerDomainsToPredicate(tv);

  const std::unordered_map<IterDomain*, ValGroup>& contig_domains =
      getContigDomains(
          predicate_domains,
          std::vector<bool>(predicate_domains.size(), true),
          reverse(index_info.traversal_path),
          traversal_graph,
          concrete_info_,
          true);

  auto getCoveredPredicatedDomains =
      [&predicate_domains, &contig_domains](const ValGroup& contig_group) {
        std::unordered_set<IterDomain*> covered_domains;
        for (const auto& predicate_domain : predicate_domains) {
          auto contig_domains_it = contig_domains.find(predicate_domain);
          NVF_ERROR(contig_domains_it != contig_domains.end());
          if (contig_group == contig_domains_it->second) {
            covered_domains.emplace(predicate_domain);
          }
        }
        return covered_domains;
      };

  std::vector<RootPredicateInfo> info_vec;
  info_vec.reserve(predicate_domains.size() + non_divisible_splits.size());
  std::unordered_set<ValGroup> already_indexed_domains;

  for (const auto& predicate_domain : predicate_domains) {
    auto contig_domains_it = contig_domains.find(predicate_domain);
    NVF_ERROR(
        contig_domains_it != contig_domains.end(),
        "No contig domain mapping found for ",
        predicate_domain->toString());
    const ValGroup& contig_domain_group = contig_domains_it->second;

    VERBOSE() << "Predicate domain: " << predicate_domain->toString()
              << ", contig domain: " << contig_domain_group->front()->toString()
              << std::endl;

    auto idx_it = index_map.find(traversal_graph.toGroup(predicate_domain));
    if (!getenv("DISABLE_CONTIG_INDEXING")) {
      if (already_indexed_domains.find(contig_domain_group) !=
          already_indexed_domains.end()) {
        VERBOSE() << "Already indexed: " << predicate_domain->toString()
                  << std::endl;
        continue;
      }
      already_indexed_domains.emplace(contig_domain_group);

      if (!contig_domain_group->has(predicate_domain)) {
        VERBOSE() << "Contig predication: "
                  << contig_domain_group->front()->toString() << " instead of "
                  << predicate_domain->toString()
                  << ". Tensor: " << tv->toString() << std::endl;
      }

      // auto idx_it =
      // index_map.find(traversal_graph.toGroup(predicate_domain));
      idx_it = index_map.find(contig_domain_group);
    }
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        contig_domain_group->front()->toString());
    Val* idx = idx_it->second;
    VERBOSE() << "Predicate index of " << predicate_domain->toString() << ": "
              << idx->toInlineString() << ", unswitch? : " << is_unswitch
              << std::endl;

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
    VERBOSE() << "Before replacement: " << idx->toInlineString()
              << " after: " << stop_idx->toInlineString() << std::endl;

    if (getenv("DISABLE_CONTIG_INDEXING")) {
      info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
          SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
          predicate_domain->extent());
      info.root_ids_ = {predicate_domain};
    } else {
      info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
          SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
          contig_domain_group->front()->as<IterDomain>()->extent());
      info.root_ids_ = getCoveredPredicatedDomains(contig_domain_group);
      VERBOSE() << "Contig covered root: " << toDelimitedString(info.root_ids_)
                << std::endl;
    }

    info_vec.emplace_back(info);
  }

  // If this is a reduction init expr, then no need to take care of
  // non divisible splits
  if (!lower_utils::isReductionInitExpr(expr)) {
    for (const auto& [eg, direction] : index_info.traversal_path) {
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
      info.start_predicate_ = non_divisible_domain->fusion()->trueVal();
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
  }

  return info_vec;
}

void TensorIndexer::setupAllocationDomains(const std::vector<Expr*>& exprs) {
  AllocationDomainSetup alloc_setup;
  alloc_setup.setup(exprs);
  alloc_info_ = std::move(alloc_setup.tv_alloc_info_map);
}

bool TensorIndexer::isSupported(Fusion* fusion) {
  const auto all_tvs = ir_utils::allTvs(fusion);

  auto printReason = [](const std::string& reason) -> void {
    VERBOSE() << "TensorIndexer disabled due to: " << reason << std::endl;
  };

  if (fusion->hasManaged("loop_rotation")) {
    printReason("loop rotation is not supported");
    return false;
  }

  for (const auto& tv : all_tvs) {
    std::stringstream reason;

    if (auto loadstore = dynamic_cast<LoadStoreOp*>(tv->definition());
        loadstore != nullptr &&
        (loadstore->opType() == LoadStoreOpType::LdMatrix)) {
      // loadstore->opType() == LoadStoreOpType::CpAsync ||
      // loadstore->opType() == LoadStoreOpType::CpAsyncBulkTensorTile)) {
      reason << "LoadStoreOp not supported: " << loadstore->toString();
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
      printReason(reason.str());
      return false;
    }
  }

  return true;
}

} // namespace nvfuser

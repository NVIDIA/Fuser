// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <predicate_compute.h>

#include <device_lower/lower2device.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <id_model/indexing_traversal.h>
#include <id_model/predicate_indexing.h>
#include <id_model/utils.h>
#include <index_compute.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <transform_iter.h>

#include <device_lower/utils.h>

namespace nvfuser {

namespace {

bool isTensorIndexOp(Expr* expr) {
  const auto& outputs = expr->outputs();
  return !outputs.empty() && outputs[0]->isA<kir::TensorIndex>();
}

bool isOutputLocal(const Expr* expr) {
  return std::all_of(
      expr->outputs().begin(), expr->outputs().end(), [](const Val* output) {
        return !output->isA<TensorView>() ||
            output->as<TensorView>()->getMemoryType() == MemoryType::Local;
      });
}

} // namespace

bool ParallelizedDomainPredicate::PredicateInfo::addDomain(IterDomain* id) {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::EXACT);
  if (std::find(ids_.begin(), ids_.end(), concrete_id) == ids_.end()) {
    ids_.push_back(concrete_id);
    return true;
  } else {
    return false;
  }
}

Val* ParallelizedDomainPredicate::PredicateInfo::getPredicate() const {
  Val* pred = nullptr;

  auto index = SimplifyingIrBuilder::create<NamedScalar>(
      stringifyThread(pt_), DataType::Int);

  for (const auto& pred_id : ids()) {
    // Just sanity check that pred_id is concrete
    NVF_ERROR(
        pred_id ==
        GpuLower::current()->caMap()->getConcreteMappedID(
            pred_id, IdMappingMode::EXACT));
    auto new_pred = SimplifyingIrBuilder::ltExpr(index, pred_id->extent());
    pred = SimplifyingIrBuilder::logicalAndExpr(pred, new_pred);
  }

  return pred;
}

namespace {

// For a given loop nest represented by a vector of ForLoops, returns
// all unswitched parallel loop IDs that do not require parallel type
// predicates. An ID is considered fully unswitched when all of its
// dependent loop IDs are unswitched. Similarly, a loop is fully
// unswitched when all of its dependent predicated IDs are fully
// unswitched. This information is used to determine if it's safe to
// omit the predicate for a parallel type.
std::vector<IterDomain*> getUnswitchProtectedParallelLoopIds(
    const Expr* expr,
    const std::vector<ForLoop*>& loops,
    ForLoop* unswitched_loop) {
  if (unswitched_loop == nullptr) {
    return {};
  }

  const auto& id_model = GpuLower::current()->idModel();
  const auto& indexing_graph =
      id_model.idGraph(TensorIndexer::traversalGraphType());

  auto out_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(out_tv != nullptr);

  std::vector<IterDomain*> loop_ids;
  loop_ids.reserve(loops.size());
  std::transform(
      loops.begin(),
      loops.end(),
      std::back_inserter(loop_ids),
      [&](ForLoop* loop) {
        return getLoopPromotion(loop->iter_domain(), id_model);
      });

  const auto predicate_ids = getPredicateDomains(out_tv, expr);

  const IndexingTraversal::ExprPath predicate_path =
      IndexingTraversal::getExprsBetween(
          expr, indexing_graph, loop_ids, predicate_ids);

  // All loops that are right of unswitched_loop are also unswitched,
  // except when they are parallelized. We don't assign maximum possible
  // index values to unswitched parallel loops (e.g., threadIdx.x, not
  // blockDim.x - 1), so parallelized loops are not considered
  // unswitched for the sake of this analysis.
  ValGroups non_unswitch_dep_ids;
  bool unswitch_found = false;
  for (const auto loop : loops) {
    if (loop == unswitched_loop) {
      unswitch_found = true;
    }
    if (!unswitch_found ||
        isParallelTypeThread(loop->iter_domain()->getParallelType())) {
      non_unswitch_dep_ids.pushBack(
          indexing_graph.toGroup(loop->iter_domain()));
    }
  }

  // Find all IDs along the predicate indexing path that depend on the
  // non unswitched loop IDs.
  for (const auto& [expr_g, dir] : predicate_path) {
    const auto inputs = getInputsOfExprGroup(indexing_graph, expr_g, dir);
    const auto outputs = getOutputsOfExprGroup(indexing_graph, expr_g, dir);
    if (std::any_of(inputs.begin(), inputs.end(), [&](const ValGroup& input) {
          return non_unswitch_dep_ids.has(input);
        })) {
      // Depends on non-unswitched ids
      non_unswitch_dep_ids.pushBack(outputs);
    }
  }

  std::vector<IterDomain*> unswitch_protected_loop_ids;
  unswitch_found = false;
  for (const auto loop : loops) {
    if (loop == unswitched_loop) {
      unswitch_found = true;
    }

    if (!unswitch_found) {
      continue;
    }

    const auto unswitched_loop_id = loop->iter_domain();
    const ParallelType pt = unswitched_loop_id->getParallelType();

    // Don't care serial loops
    if (!isParallelTypeThread(pt)) {
      continue;
    }

    // Traverse the predicate indexing path from this unswitched loop
    // ID. If any expr along the path also uses any of the non
    // unswitched IDs or their dependent IDs, this loop ID is not
    // considered fully unswitched. Also, even if unswitched,
    // parallelized loop IDs do not use the maximum possible value as
    // their indices (e.g., not (blockDim.x - 1) but threadIdx.x), so
    // there must be no use of any of other parallel types than this
    // parallel type.

    // Keep track of IDs that have dependencies with unswitched_loop_id
    ValGroups unswitch_dep_ids;
    unswitch_dep_ids.pushBack(indexing_graph.toGroup(unswitched_loop_id));

    bool protected_by_unswitch = true;

    for (const auto& [expr_g, dir] : predicate_path) {
      const auto inputs = getInputsOfExprGroup(indexing_graph, expr_g, dir);
      const auto outputs = getOutputsOfExprGroup(indexing_graph, expr_g, dir);

      // If none of the inputs depends on unswitched_loop_id and its
      // dependents, this expr should not matter.
      if (std::none_of(
              inputs.begin(), inputs.end(), [&](const ValGroup& input) {
                return unswitch_dep_ids.has(input);
              })) {
        continue;
      }

      // If any of the non unswitched IDs is used, this is not
      // protected. Note that non_unswitch_dep_ids contains all
      // parallelized unswitched IDs and their dependents, including
      // unswitched_loop_id itself. Use of unswitched_loop_id and its
      // dependents should not make unswitched_loop_id not fully
      // unswitched.
      if (std::any_of(inputs.begin(), inputs.end(), [&](const ValGroup& input) {
            return non_unswitch_dep_ids.has(input) &&
                !unswitch_dep_ids.has(input);
          })) {
        protected_by_unswitch = false;
        break;
      }

      // Continue to keep track of the dependencies from unswitched_loop_id
      unswitch_dep_ids.pushBack(outputs);
    }

    if (protected_by_unswitch) {
      unswitch_protected_loop_ids.push_back(unswitched_loop_id);
    }
  }

  return unswitch_protected_loop_ids;
}

} // namespace

std::unordered_map<ParallelType, ParallelizedDomainPredicate::PredicateInfo>
ParallelizedDomainPredicate::getPredicateMap(
    const Expr* expr,
    const std::vector<ForLoop*>& loops,
    ForLoop* unswitched_loop) {
  const auto gpu_lower = GpuLower::current();
  auto output_tvs = ir_utils::getTvs(expr->outputs());

  if (output_tvs.empty()) {
    return {};
  }

  // Initialize a map with empty predicate info
  std::unordered_map<ParallelType, PredicateInfo> map;
  for (auto pt : kParallelTypeThreads) {
    map.insert({pt, PredicateInfo(pt)});
  }

  // For each loop, check if it's parallelized by an non-exact
  // threading dimension. If yes and it's used in the given expr, the
  // domain needs to be protected by a predicate on the thread/block
  // index.

  bool within_unswitch = false;
  std::unordered_set<Val*> non_unswitched_root_domains;

  auto unswitch_protected_loop_ids =
      getUnswitchProtectedParallelLoopIds(expr, loops, unswitched_loop);

  for (const auto i : arange(loops.size())) {
    auto loop = loops[i];

    // Parallel dimensions need not be predicated if fully unswitched.
    if (loop == unswitched_loop) {
      within_unswitch = true;
    }

    auto loop_id = loop->iter_domain();
    auto loop_ptype = loop_id->getParallelType();

    // Not necessary to add a predicate if the paralle type is exact
    if (!isParallelTypeThread(loop_ptype) ||
        lower_utils::isExtentEqualToMaxParallelTypeExtent(loop_id)) {
      continue;
    }
    auto parallel_dim = gpu_lower->parallelDimensionMap().getRaw(loop_ptype);

    // If protected by unswitch, the unswitch predicate is enough without
    // predicating the parallel type. For example, suppose a logical
    // ID is inner split by a factor of K and both of the two outputs
    // are unswitched. Also suppose the inner output IDs is
    // parallelized with TIDx but the other output is not. The logical
    // ID would be predicated by something like:
    //
    //   threadIdx.x + (ceilDiv(N, K) - 1) * K < N
    //
    // where N is the extent of the logical ID. As you can see, since
    // the other output is assigned with the maximum index, this
    // predicate is sufficient even when blockDim.x > K.
    if (within_unswitch &&
        std::find(
            unswitch_protected_loop_ids.begin(),
            unswitch_protected_loop_ids.end(),
            loop_id) != unswitch_protected_loop_ids.end()) {
      continue;
    }

    for (auto tv : output_tvs) {
      // Check if the loop domain is used by the output tensor
      auto it = std::find_if(
          tv->getLoopDomain().begin(),
          tv->getLoopDomain().end(),
          [&](auto tv_id) {
            return gpu_lower->caMap()->areMapped(
                loop_id, tv_id, IdMappingMode::EXACT);
          });
      if (it == tv->getLoopDomain().end()) {
        continue;
      }

      IterDomain* tv_id = *it;

      // If the corresponding domain is a broadcast, it's not really used.
      if (tv_id->isBroadcast()) {
        continue;
      }

      // If it's a root domain, it should be covered by the root
      // predicates, so no extra predicate is required.
      if (std::find(
              tv->getMaybeRootDomain().begin(),
              tv->getMaybeRootDomain().end(),
              tv_id) != tv->getMaybeRootDomain().end()) {
        continue;
      }

      // loop_ptype not being exact does not mean the predicate is not trivial.
      // For example, if I have T1[blockIdx.x{3}] and T2[blockIdx.x{5}], then
      // blockIdx.x will not be exact. However, the predicate blockIdx.x < 5 is
      // still trivial.
      if (tv_id->extent()->sameAs(parallel_dim)) {
        continue;
      }

      // tv_id needs to be predicated. Adds it to the PredicateInfo map.
      auto& info = map.at(loop_ptype);
      info.addDomain(tv_id);
    }
  }

  return map;
}

Val* ParallelizedDomainPredicate::getPredicate(
    const Expr* expr,
    const std::vector<ForLoop*>& loops) {
  DEBUG_PRINT_SCOPE_NAME(
      "ParallelizedDomainPredicate::getPredicate", "expr = ", expr);
  auto pred_map = getPredicateMap(expr, loops);

  Val* pred = GpuLower::current()->kernel()->trueVal();

  for (auto pt : kParallelTypeThreads) {
    auto pred_info_it = pred_map.find(pt);
    if (pred_info_it != pred_map.end()) {
      const auto& pred_info = pred_info_it->second;
      auto tid_pred = pred_info.getPredicate();
      pred = SimplifyingIrBuilder::logicalAndExpr(pred, tid_pred);
    }
  }

  NVF_ERROR(pred != nullptr);
  RECORD_AND_RETURN(pred);
}

UnswitchPredicateKey::UnswitchPredicateKey()
    : predicated_concrete_id_(nullptr) {
  for (auto pt : kParallelTypeThreads) {
    parallel_concrete_ids_.insert({pt, nullptr});
  }
}

// For a predicated concrete domain, id, find which thread parallel
// types are used. For each used parallel type, find the concrete
// domain that the paralllel type is associated with. The parallelized
// concrete domains are used to uniquely collect all necessary
// unswitch predicates.
UnswitchPredicateKey::UnswitchPredicateKey(
    IterDomain* predicated_consumer_id,
    TensorView* consumer_tv,
    IterDomain* predicated_concrete_id,
    std::unordered_set<IterDomain*> loop_ids)
    : predicated_concrete_id_(predicated_concrete_id),
      loop_ids_(std::move(loop_ids)) {
  // Initialize the parallelized domain map
  // TODO: Add DID
  for (auto pt : kParallelTypeThreads) {
    parallel_concrete_ids_.insert({pt, nullptr});
  }

  std::vector<Val*> all_parallelized_consumer_loop_ids;

  // Identify which parallel type is used for which loop domain for
  // this index. This information is obtained here for the legacy method by
  // looking at the dependency between the predicated domain and the
  // loop domains of the tensor. However, in the case of the new
  // indexer, that information is not readily available since the
  // indexing graph needs to be traversed. Instead, the loop
  // dependency information is given to this constructor in the case
  // of the new indexer.
  //
  // TODO: Clean up once the migration to the new indexer is
  // completed.
  //
  // When loop_ids_ is not empty, the correct loop domains are already
  // given to this class. That's the case when using the new
  // indexer. When given, use them to figure out which parallel type
  // is used for which loop domain.
  if (!loop_ids_.empty()) {
    for (auto loop_id : loop_ids_) {
      auto pt = loop_id->getParallelType();
      // DID is ignored.
      // TODO: support DID
      if (isParallelTypeThread(pt)) {
        // This map is supposed to contain CA conrete IDs but as long
        // as they are uniquely mapped to some representative domains,
        // it should work.
        parallel_concrete_ids_.at(pt) = GpuLower::current()
                                            ->tensorIndexer()
                                            .traversalGraph()
                                            .toGroup(loop_id)
                                            ->front()
                                            ->as<IterDomain>();
      }
    }
    return;
  }

  std::copy_if(
      consumer_tv->getLoopDomain().begin(),
      consumer_tv->getLoopDomain().end(),
      std::back_inserter(all_parallelized_consumer_loop_ids),
      [](IterDomain* x) { return isParallelTypeThread(x->getParallelType()); });

  // If the consumer domais are not parallelized at all, no need to
  // differentiate keys based on how the predicated id is parallelized
  if (all_parallelized_consumer_loop_ids.empty()) {
    return;
  }

  // All domains that are parallelized descendants of predicated_consumer_id
  auto all_parallelized_consumer_ids = DependencyCheck::getAllValsBetween(
      {predicated_consumer_id}, all_parallelized_consumer_loop_ids);
  // Just pick loop domains
  std::vector<IterDomain*> parallelized_consumer_loop_ids;
  std::copy_if(
      consumer_tv->getLoopDomain().begin(),
      consumer_tv->getLoopDomain().end(),
      std::back_inserter(parallelized_consumer_loop_ids),
      [&](IterDomain* x) {
        return std::find(
                   all_parallelized_consumer_ids.begin(),
                   all_parallelized_consumer_ids.end(),
                   x) != all_parallelized_consumer_ids.end();
      });

  if (parallelized_consumer_loop_ids.empty()) {
    // None of the parallelized loop domains are derived from
    // predicated_consumer_id
    return;
  }

  // Find the corresponding concrete id for each parallel type
  for (auto consumer_loop : parallelized_consumer_loop_ids) {
    auto pt = consumer_loop->getParallelType();
    auto concrete_loop = GpuLower::current()->caMap()->getConcreteMappedID(
        consumer_loop, IdMappingMode::EXACT);
    parallel_concrete_ids_.at(pt) = concrete_loop;
  }
}

std::string UnswitchPredicateKey::toString() const {
  std::stringstream ss;
  ss << "Predicated domain: ";
  if (predicatedId() != nullptr) {
    ss << predicatedId();
  } else {
    ss << "null";
  }
  for (auto pt : kParallelTypeThreads) {
    auto pid = parallelId(pt);
    ss << ", " << pt << ": ";
    if (pid) {
      ss << pid;
    } else {
      ss << "null";
    }
  }
  return ss.str();
}

std::size_t UnswitchPredicateKeyHash::operator()(
    const UnswitchPredicateKey& key) const {
  auto h = std::hash<const IterDomain*>{}(key.predicatedId());
  for (auto pt : kParallelTypeThreads) {
    h = h ^ std::hash<const IterDomain*>{}(key.parallelId(pt));
  }
  return h;
};

namespace {

// Create elect-sync to pick a thread
Val* createElectSyncExpr() {
  Val* full_mask_val = IrBuilder::create<Val>(0xFFFFFFFF, PrimDataType::UInt32);
  Val* elect_sync_val = IrBuilder::create<Val>(PrimDataType::Bool);
  IrBuilder::create<UnaryOp>(
      UnaryOpType::ElectSync, elect_sync_val, full_mask_val);
  return elect_sync_val;
}

// Select first warp of threads along TIDx axis and use ptx::elect_sync if not
// warp collective.
// TODO If TIDx is known at compile-time, generate custom mask.
Val* selectFirstWarpElectSyncPredicate(bool is_warp_collective) {
  Val* warp_size = IrBuilder::create<Val>(32L, PrimDataType::UInt64);
  Val* select_first_warp = IrBuilder::ltExpr(
      NamedScalar::getParallelIndex(ParallelType::TIDx), warp_size);

  // Short-Circuit: TMA Store is a warp-collective, so ElectSync is not
  // necessary.
  if (is_warp_collective) {
    return select_first_warp;
  }

  return SimplifyingIrBuilder::logicalAndExpr(
      createElectSyncExpr(), select_first_warp);
}

// Get linear index for AsyncWarp Group. Then, select first warp. Finally, use
// ptx::elect_sync if not warp collective.
// TODO If TIDx is known at compile-time, generate custom mask.
Val* createElectSyncPredicateAsync() {
  Val* zero = IrBuilder::create<Val>(0L, PrimDataType::UInt64);
  Val* warp_size = IrBuilder::create<Val>(32L, PrimDataType::UInt64);

  const ParallelDimensionMap& pdim_map =
      GpuLower::current()->parallelDimensionMap();
  Val* async_warp_thread_index = pdim_map.getLinearThreadIndexAsync();
  Val* warp_id =
      SimplifyingIrBuilder::divExpr(async_warp_thread_index, warp_size);
  // TODO Only select first warp now
  Val* select_warp = SimplifyingIrBuilder::eqExpr(warp_id, zero);

  // Use elect-sync if available
  if (pdim_map.canUseElectSyncInAsyncWarp()) {
    return SimplifyingIrBuilder::logicalAndExpr(
        select_warp, createElectSyncExpr());
  }

  // Warp Specialized ParallelType is ThreadIdx.x and it contains less than 32
  // threads, so manually select first thread in warp.
  Val* thread_id =
      SimplifyingIrBuilder::modExpr(async_warp_thread_index, warp_size);
  Val* select_thread = SimplifyingIrBuilder::eqExpr(thread_id, zero);
  return SimplifyingIrBuilder::logicalAndExpr(select_warp, select_thread);
}

Val* createElectSyncPredicate(kir::Predicate* pred, bool is_async_warp) {
  NVF_ERROR(pred != nullptr);
  NVF_ERROR(pred->expr() != nullptr);

  TensorView* out_tv = ir_utils::getTvOutput(pred->expr());
  NVF_ERROR(out_tv != nullptr, "Missing TensorView output");

  bool is_tv_tidx_parallelized = std::any_of(
      out_tv->domain()->loop().begin(),
      out_tv->domain()->loop().end(),
      [](IterDomain* id) {
        return id->getParallelType() == ParallelType::TIDx;
      });

  // short-circuit: out_tv uses ParallelType::TIDx
  if (is_tv_tidx_parallelized) {
    return pred->fusion()->trueVal();
  }

  Val* tidx_paralleltype_dim =
      GpuLower::current()->parallelDimensionMap().get(ParallelType::TIDx);

  // short-circuit: ParallelType::TIDx is not used in cuda kernel.
  if (tidx_paralleltype_dim == nullptr) {
    return pred->fusion()->trueVal();
  }

  // short-circuit: Expect ParallelType::TIDx to have at least one warp.
  bool is_tma_store = ir_utils::isCpAsyncBulkStore(pred->expr());
  if (tidx_paralleltype_dim->isConstScalar() &&
      tidx_paralleltype_dim->evaluate().as<int64_t>() < 32) {
    if (is_tma_store) {
      return pred->fusion()->trueVal();
    } else {
      Val* zero = IrBuilder::create<Val>(0L, PrimDataType::UInt64);
      return IrBuilder::eqExpr(
          NamedScalar::getParallelIndex(ParallelType::TIDx), zero);
    }
  }

  NVF_ERROR(!(is_tma_store && is_async_warp));
  if (is_async_warp) {
    return createElectSyncPredicateAsync();
  }
  return selectFirstWarpElectSyncPredicate(is_tma_store);
}

Val* createSingleExpressionElectSync(
    kir::Predicate* pred,
    const std::vector<ForLoop*>& loops) {
  NVF_ERROR(pred->expr() != nullptr);
  NVF_ERROR(
      ir_utils::isCpAsyncBulk(pred->expr()) ||
          (pred->expr()->isA<MmaOp>() &&
           pred->expr()->as<MmaOp>()->isBlackwell()),
      "Limited to TMA/Blackwell MMA expressions");

  TensorView* out_tv = ir_utils::getTvOutput(pred->expr());
  Val* zero = IrBuilder::create<Val>(0L, PrimDataType::UInt64);

  const ParallelDimensionMap& pdim_map =
      GpuLower::current()->parallelDimensionMap();
  auto pred_map =
      ParallelizedDomainPredicate::getPredicateMap(pred->expr(), loops);

  bool is_async_warp = std::any_of(loops.begin(), loops.end(), [](ForLoop* fl) {
    return fl->circularBufferLoopStage() == CircularBufferLoopStage::AsyncWarp;
  });

  Val* parallel_dom_pred = GpuLower::current()->kernel()->trueVal();
  for (auto pt : {ParallelType::TIDx, ParallelType::TIDy, ParallelType::TIDz}) {
    // short-circuit: parallelDim is not used by CTA
    if (!pdim_map.has(pt)) {
      continue;
    }

    // Case 1: TMA/Blackwell MMA expression uses ParallelDim to launch multiple
    // operations simultaneously. Use parallel domain predicate if it
    // exists.
    auto pred_info_it = pred_map.find(pt);
    if (pred_info_it != pred_map.end()) {
      const ParallelizedDomainPredicate::PredicateInfo& pred_info =
          pred_info_it->second;
      parallel_dom_pred = SimplifyingIrBuilder::logicalAndExpr(
          parallel_dom_pred, pred_info.getPredicate());
    }

    // Case 2: ParallelDim is used by CTA but not the TMA/Blackwell MMA
    // expression. Select a single thread along ParallelDim.
    bool is_tv_tid_parallelized = std::any_of(
        out_tv->domain()->loop().begin(),
        out_tv->domain()->loop().end(),
        [&](IterDomain* id) { return id->getParallelType() == pt; });
    if (!is_tv_tid_parallelized) {
      if (pt == ParallelType::TIDx) {
        // Use createElectSyncPredicate for ParallelDim::TIDx.
        parallel_dom_pred = SimplifyingIrBuilder::logicalAndExpr(
            parallel_dom_pred, createElectSyncPredicate(pred, is_async_warp));
      } else {
        // Select first element of dimension for ParallelDim::TIDy and
        // ParallelDim::TIDz.
        Val* paralleltype_dim =
            GpuLower::current()->parallelDimensionMap().get(pt);
        if (paralleltype_dim == nullptr || !paralleltype_dim->isOneInt()) {
          parallel_dom_pred = SimplifyingIrBuilder::logicalAndExpr(
              parallel_dom_pred,
              IrBuilder::eqExpr(NamedScalar::getParallelIndex(pt), zero));
        }
      }
    }
  }
  NVF_ERROR(parallel_dom_pred != nullptr);
  return parallel_dom_pred;
}

// Multiple expressions exist in a for-loop. The common usage of this function
// is to initialize mbarriers, invalidate mbarriers, or issue multiple TMA load
// operations in a circular buffer for-loop.
//
// Assumptions required for this elect-sync predicate:
//  1. ParallelType::TIDx >= 32 threads.
//  2. TMA expression does not use ParallelType::TIDy or ParallelType::TIDz.
Val* createMultipleExpressionElectSync(
    kir::Predicate* pred,
    const std::vector<ForLoop*>& loops) {
  NVF_ERROR(pred->expr() == nullptr);

  Val* zero = IrBuilder::create<Val>(0L, PrimDataType::UInt64);
  const ParallelDimensionMap& pdim_map =
      GpuLower::current()->parallelDimensionMap();

  // Determine if warp specialized tma load expression.
  ParallelType async_warp_on = ParallelType::Serial;
  auto async_warp_loop_it =
      std::find_if(loops.begin(), loops.end(), [](ForLoop* fl) {
        return fl->circularBufferLoopStage() ==
            CircularBufferLoopStage::AsyncWarp;
      });
  if (async_warp_loop_it != loops.end()) {
    auto circular_buffer_type = std::get<WarpSpecialized>(
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor((*async_warp_loop_it)->iter_domain())
            .type);
    async_warp_on = circular_buffer_type.on;
  }

  // Short-circuit: If we are in a async warp, then the warp-dispatching
  // IfThenElse already selects on `async_warp_on`, so we should not
  // generate predicates for it here.
  if (async_warp_loop_it == loops.end()) {
    Val* conditional = async_warp_on == ParallelType::TIDx
        ? pred->fusion()->trueVal()
        : selectFirstWarpElectSyncPredicate(/*is_warp_collective=*/false);
    for (ParallelType pt : {ParallelType::TIDy, ParallelType::TIDz}) {
      if (pdim_map.has(pt) && async_warp_on != pt) {
        conditional = SimplifyingIrBuilder::logicalAndExpr(
            conditional,
            IrBuilder::eqExpr(NamedScalar::getParallelIndex(pt), zero));
      }
    }
    return conditional;
  }

  return createElectSyncPredicateAsync();
}

} // namespace

// predicate value for 1D TMA load and expect arrive bytes, it combines
// ElectSync and Inline predicate.
OneDimTmaPredicateInfo PredicateCompute::OneDimTmaLoadExpectArrive(
    kir::Predicate* pred,
    const std::vector<ForLoop*>& current_loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::OneDimTmaLoadExpectArrive");
  auto expr = pred->expr();
  NVF_ERROR(expr != nullptr);
  OneDimTmaPredicateInfo one_dim_tma_pred_info;
  auto pval_elect_sync = createElectSyncPredicate(pred, true);
  auto pval_inline = getInlinePredicate(
      expr,
      current_loops,
      /*rotated_loop_=*/std::unordered_set<ForLoop*>{},
      /*thread_pred=*/nullptr,
      PredicateType::Inline);
  // We want to merge [pval_inline] with [pval_elect_sync].
  // However, the loop indices nested in [ IF ElectSync] are no longer
  // accessible when predicates are combined. Therefore, we visit all the
  // for-loops after the one contains elect sync and replace loop index with
  // zero.
  std::unordered_map<Val*, Val*> replace_map;
  for (auto fl : pred->tma1dLoadLoops()) {
    // save circular buffer loop index, will be replaced when generating
    // predicate for MBarrierWaitParity in computation branch.
    if (fl->circularBufferLoopStage() == CircularBufferLoopStage::AsyncWarp) {
      one_dim_tma_pred_info.circular_loop_index = fl->index();
      continue;
    }
    // tma1dLoadLoops() returns all the loops above the actual tma load expr.
    // skip the loops that are already in the current loop nest since their
    // indices are accessible.
    if (std::any_of(
            current_loops.begin(), current_loops.end(), [&](ForLoop* loop) {
              return loop->iter_domain() == fl->iter_domain();
            })) {
      continue;
    }
    // Replace indicies of other forloops to 0.
    // Replace the loop index with zero removes the corresponding predicate
    // to this loop-domain, we should ensure the split generating this
    // domain is divisible.
    replace_map[fl->index()] = GpuLower::current()->kernel()->zeroVal();
    auto id_def = fl->iter_domain()->definition();
    if (!id_def) {
      continue;
    }
    if (auto split = dynamic_cast<Split*>(id_def)) {
      GpuLower::current()->validate(
          split->isDivisible(),
          "Loop domains between circular buffer and 1D TMA load requires "
          "divisible split, got: ",
          split->toString());
    }
  }
  pval_inline = ir_utils::replaceValRecursively(pval_inline, replace_map);
  one_dim_tma_pred_info.inline_pred_val = pval_inline;
  one_dim_tma_pred_info.combined_pred_val =
      SimplifyingIrBuilder::logicalAndExpr(pval_elect_sync, pval_inline);
  return one_dim_tma_pred_info;
}

// predicates MBarrierWaitParity for 1d tma load
Val* PredicateCompute::OneDimTmaWaitParity(
    kir::Predicate* pred,
    const std::vector<ForLoop*>& current_loops,
    const OneDimTmaPredicateInfo& one_dim_tma_pred_info) {
  FUSER_PERF_SCOPE("GpuLower::Lower::OneDimTmaWaitParity");
  auto expr = pred->expr();
  NVF_ERROR(expr != nullptr);
  // Since MBarrierWaitParity has no output tensor, its predicate value
  // cannot be computed directly. Instead, we reuse [inline_pred_1d_tma], but
  // replace the circular buffer load loop index with that of the circular
  // buffer compute loop.
  NVF_ERROR(expr->isA<kir::MBarrierWaitParity>())
  auto inline_pred_1d_tma = one_dim_tma_pred_info.inline_pred_val;
  auto circular_loop_index = one_dim_tma_pred_info.circular_loop_index;
  auto fl = current_loops.back();
  std::unordered_map<Val*, Val*> replace_map;
  replace_map[circular_loop_index] = fl->index();
  auto pred_val =
      ir_utils::replaceValRecursively(inline_pred_1d_tma, replace_map);
  return pred_val;
}

Val* PredicateCompute::getElectSyncPredicate(
    kir::Predicate* pred,
    const std::vector<ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getElectSyncPredicate");

  // Short-Circuit: A single expression is associated with the predicate.
  if (pred->expr() != nullptr) {
    return createSingleExpressionElectSync(pred, loops);
  }

  return createMultipleExpressionElectSync(pred, loops);
}

Val* PredicateCompute::getInlinePredicate(
    const Expr* expr,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    Val* thread_pred,
    PredicateType pred_type) {
  DEBUG_PRINT_SCOPE(
      "expr = ",
      expr,
      "thread_pred = ",
      thread_pred,
      "pred_type = ",
      pred_type);
  FUSER_PERF_SCOPE("GpuLower::Lower::getInlinePredicate");

  const auto gpu_lower = GpuLower::current();

  // If outputs are registers, no need to predicate for threads
  if (isOutputLocal(expr)) {
    thread_pred = gpu_lower->kernel()->trueVal();
    // If it is a initilization op, return immediately.
    if (ir_utils::isTensorScalarFillOp(expr)) {
      RECORD_AND_RETURN(thread_pred);
    }
  }

  if (loops.empty()) {
    NVF_ERROR(thread_pred != nullptr);
    RECORD_AND_RETURN(thread_pred);
  }

  auto out_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(out_tv != nullptr, "Missing TensorView output");

  if (gpu_lower->predicateElimination().canOmitPredicate(expr)) {
    RECORD_AND_RETURN(thread_pred);
  }

  auto parallel_dom_pred =
      ParallelizedDomainPredicate::getPredicate(expr, loops);
  NVF_ERROR(parallel_dom_pred != nullptr);

  // TMA handles out-of-bounds accesses in hardware, so parallel_dom_pred
  // itself is sufficient to predicate the accesses.
  // TMem ld/st accesses TMem in a very specific pattern and can not be
  // predicated like accesses to general memory types, we do not have a good
  // way to predicate the accesses yet, so we just skip the predicate for now.
  if (ir_utils::isCpAsyncBulkTensorTile(expr) || ir_utils::isLdStTMem(expr)) {
    RECORD_AND_RETURN(parallel_dom_pred);
  }

  std::vector<PredicateInfo> pred_info_vec;
  if (!ir_utils::hasRootToLoopLinearTransformations(out_tv) ||
      GpuLower::current()->idModelOptions().inlinePredicate()) {
    pred_info_vec =
        gpu_lower->tensorIndexer().getPredicates(out_tv, expr, loops);
  } else {
    pred_info_vec = Index::getReferenceRootPredicates(
        out_tv, loops, rotated_loops, nullptr);
  }

  std::vector<Val*> preds;

  // When pred_type is ReductionWrite, filter out predicates for
  // reduction axes. For blockReduce, this is necessary when reduction
  // axes start at non-zero offsets and parallelized with TID since
  // blockReduce returns a valid output only at offset-zero
  // threads. Similarly, for gridReduce, the last block to store the
  // output may be predicated out with the read predicate, so the
  // write predicate needs to ignore the reduction axes.
  bool non_zero_start_found = false;
  for (const auto& pred_info : pred_info_vec) {
    if (pred_type == PredicateType::ReductionWrite) {
      const auto& consumer_ids = pred_info.predicatedDomains();
      bool pred_for_reduction_axis = false;
      for (auto consumer_id : consumer_ids) {
        if (consumer_id->isReduction()) {
          if (!consumer_id->start()->isZeroInt()) {
            non_zero_start_found = true;
          }
          pred_for_reduction_axis = true;
          break;
        }
      }
      // Don't add the predicate if it corresponds to a reduction axis
      if (pred_for_reduction_axis) {
        continue;
      }
    }
    preds.push_back(pred_info.startPredicate());
    preds.push_back(pred_info.stopPredicate());
  }

  // When generating a predicate for blockReduce writes and not for
  // gridReduce, if all reduction axes start with zero, we can just
  // use the same predicate for reads. nullptr is returned then.
  if (pred_type == PredicateType::ReductionWrite && !non_zero_start_found &&
      !out_tv->domain()->hasGridReduction()) {
    RECORD_AND_RETURN(nullptr);
  }

  preds.push_back(parallel_dom_pred);

  // Don't need thread predicate for 1D TMA load with circular buffer, it is
  // already predicated with ElectSync.
  if (thread_pred &&
      !(ir_utils::isCpAsyncBulk1D(expr) &&
        gpu_lower->circularBufferInfo().getCircularBufferAxis(out_tv))) {
    preds.push_back(thread_pred);
  }

  if (preds.empty()) {
    RECORD_AND_RETURN(GpuLower::current()->kernel()->trueVal());
  }

  Val* cond = preds[0];
  for (const auto i : arange(1, preds.size())) {
    cond = SimplifyingIrBuilder::logicalAndExpr(cond, preds[i]);
  }

  RECORD_AND_RETURN(cond);
}

Val* UnswitchPredicate::get(
    const std::vector<ForLoop*>& outer_loops,
    ForLoop* unrolled_loop) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::get");

  UnswitchPredicate up(outer_loops, unrolled_loop);

  Val* unswitch_pred = GpuLower::current()->kernel()->trueVal();
  for (auto pred : up.predicates_) {
    unswitch_pred = SimplifyingIrBuilder::logicalAndExpr(unswitch_pred, pred);
  }

  return unswitch_pred;
}

void UnswitchPredicate::predicateOn(Expr* tv_expr) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::predicateOn");

  if (for_loops_.empty()) {
    return;
  }

  const auto gpu_lower = GpuLower::current();

  if (gpu_lower->predicateElimination().canOmitPredicate(tv_expr)) {
    return;
  }

  auto out_tv = ir_utils::getTvOutput(tv_expr);
  NVF_ERROR(out_tv != nullptr, "Missing TensorView output");

  std::vector<PredicateInfo> ref_pred_info;

  if (!ir_utils::hasRootToLoopLinearTransformations(out_tv) ||
      GpuLower::current()->idModelOptions().unswitchPredicate()) {
    ref_pred_info = gpu_lower->tensorIndexer().getPredicates(
        out_tv, tv_expr, for_loops_, unrolled_loop_);
  } else {
    ref_pred_info = Index::getReferenceRootPredicates(
        out_tv, for_loops_, rotated_loop_, unrolled_loop_);
  }

  // If RootPredicateInfo has a static predicate that is more
  // restrictive than the current one, replace the current with the
  // new one. If it has a dynamic predicate, add it to the dynamic
  // predicate list. Since the final static predicate can't be
  // determined until all expressions are analyzed, predicates are
  // temporarily placed in the predicated_keys map and the final
  // predicates are generated in the finalize function.

  for (const auto& pred_info : ref_pred_info) {
    NVF_ERROR(pred_info.startPredicate() != nullptr);
    NVF_ERROR(pred_info.stopPredicate() != nullptr);

    const auto& root_ids = pred_info.predicatedDomains();

    bool add_pred = false;

    // Used to find a matching existing MergedPredicates
    UnswitchPredicateKey first_key;
    bool first_key_set = false;

    for (auto root_id : root_ids) {
      auto concrete_root_id = gpu_lower->caMap()->getConcreteMappedID(
          root_id, IdMappingMode::EXACT);

      if (root_id->isBroadcast()) {
        continue;
      }

      UnswitchPredicateKey key(
          root_id, out_tv, concrete_root_id, pred_info.loopDomains());
      auto inserted = predicated_keys_.insert(key).second;
      add_pred = add_pred || inserted;

      if (!first_key_set) {
        first_key = key;
        first_key_set = true;
      }
    }

    if (!first_key_set) {
      // No predicate generated
      continue;
    }

    // The start and stop offsets may need to be merged to avoid
    // redundant predicates. When these offsets are zero, nothing is
    // done. When non-zero, find the corresponding MergedPredicates
    // and merge both the start and stop offsets. Note that the
    // offsets are non-zero, the predicates must be generated at a
    // root domain, so root_ids.size() must be one. That unique root
    // domain is used as a key to find the corresponding
    // MergedPredicate.

    // Initialize with an invalid iterator to signal no corresponding
    // MergedPredicates is found yet.
    auto merged_pred_it = pending_predicates_.end();

    if (add_pred) {
      // This is a new predicate for the root domain. Initialize a new
      // MergedPredicates and add it to the pending list.
      UnswitchPredicate::MergedPredicates merged_pred;

      // To look up this MergedPredicates for other predicates
      // generated for the same predicate key
      // TODO: This seems to assume the merge logic is only necessary
      // for shift predicates. Now that it's removed, it should be
      // possible to clean this up.
      // TODO: This doesn't seem to work if circular buffer predicates
      // are involved with contig indexing.
      if (root_ids.size() == 1) {
        merged_pred.predicate_key = first_key;
      }

      pending_predicates_.push_back(merged_pred);

      merged_pred_it =
          pending_predicates_.begin() + (int64_t)pending_predicates_.size() - 1;
    } else if (root_ids.size() == 1) {
      // If not new, try to find a corresponding MergedPredicates.
      merged_pred_it = std::find_if(
          pending_predicates_.begin(),
          pending_predicates_.end(),
          [&first_key](const auto& merged_predicates) {
            return merged_predicates.predicate_key == first_key;
          });
      // Note: It is possible that no matching merged predicate info
      // is found. Since add_pred is false here, the root domain is
      // already predicated. It must mean that the root domain
      // is included in a contiguous merged domain, which means there
      // must be no halo-extended domain involved.
    }

    // If a corresponding MergedPredicates is found, merge both the
    // start and stop offsets.
    if (merged_pred_it != pending_predicates_.end()) {
      mergeUnswitchPredicates(
          pred_info.startPredicate(),
          pred_info.startOffset(),
          pred_info.loopStage(),
          merged_pred_it->start,
          true);

      mergeUnswitchPredicates(
          pred_info.stopPredicate(),
          pred_info.stopOffset(),
          pred_info.loopStage(),
          merged_pred_it->stop,
          false);
    }
  }

  addParallelizedDomainPredicates(tv_expr);
}

void UnswitchPredicate::addParallelizedDomainPredicates(Expr* tv_expr) {
  auto pred_map = ParallelizedDomainPredicate::getPredicateMap(
      tv_expr, for_loops_, unrolled_loop_);
  for (auto pt : kParallelTypeThreads) {
    auto pred_info_it = pred_map.find(pt);
    if (pred_info_it == pred_map.end()) {
      continue;
    }
    const auto& new_info = pred_info_it->second;
    auto& predicated =
        parallelized_dom_predicates_
            .insert({pt, ParallelizedDomainPredicate::PredicateInfo{pt}})
            .first->second;
    for (auto id : new_info.ids()) {
      if (predicated.addDomain(id)) {
        predicates_.push_back(new_info.getPredicate());
      }
    }
  }
}

void UnswitchPredicate::openLoop(ForLoop* fl) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::openLoop");

  for_loops_.push_back(fl);

  for (auto expr : fl->body().exprs()) {
    if (ir_utils::isTvOp(expr) || isTensorIndexOp(expr)) {
      predicateOn(expr);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      openIte(ite);
    } else if (auto for_loop = dynamic_cast<ForLoop*>(expr)) {
      openLoop(for_loop);
    }
  }

  for_loops_.pop_back();
}

void UnswitchPredicate::openIte(kir::IfThenElse* ite) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::openIte");

  // Loop rotation transform loops like
  //  for i ...
  //    statement1(i)
  //    statement2(i)
  //    statement3(i)
  //    statement4(i)
  // into
  //  statement1(0)
  //  statement2(0)
  //  for i ...
  //    statement3(i)
  //    statement4(i)
  //    if LoopRotation:
  //      statement1(i+1)
  //      statement2(i+1)
  // So when we see an `if LoopRotation` during visiting, the last loop is
  // rotated, and we need to use `i+1` instead of `i` as loop index.
  if (ite->predicate()->predicate_type() == PredicateType::LoopRotation) {
    rotated_loop_.insert(for_loops_.back());
  }

  // only expand the ite thenBody
  for (auto expr : ite->thenBody().exprs()) {
    if (ir_utils::isTvOp(expr) || isTensorIndexOp(expr)) {
      predicateOn(expr);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      openIte(ite);
    } else if (auto for_loop = dynamic_cast<ForLoop*>(expr)) {
      openLoop(for_loop);
    }
  }

  if (ite->predicate()->predicate_type() == PredicateType::LoopRotation) {
    rotated_loop_.erase(for_loops_.back());
  }
}

void UnswitchPredicate::finalize() {
  for (const auto& merged_pred : pending_predicates_) {
    const auto& start_info = merged_pred.start;
    if (start_info.static_pred) {
      predicates_.push_back(start_info.static_pred);
    }
    for (auto dynamic_pred : start_info.dynamic_preds) {
      predicates_.push_back(dynamic_pred);
    }
    const auto& stop_info = merged_pred.stop;
    if (stop_info.static_pred) {
      predicates_.push_back(stop_info.static_pred);
    }
    for (auto dynamic_pred : stop_info.dynamic_preds) {
      predicates_.push_back(dynamic_pred);
    }
  }
}

void UnswitchPredicate::mergeUnswitchPredicates(
    Val* predicate,
    Val* offset,
    CircularBufferLoopStage loop_stage,
    MergedPredicates::Info& merged_predicate_info,
    bool is_start) {
  auto is_more_restrictive_static_offset = [&is_start](
                                               auto new_val, auto current_val) {
    if (is_start) {
      return new_val < current_val;
    } else {
      return new_val > current_val;
    }
  };

  // This feels like a hacky WAR but when we have predicates generated
  // from certain circular buffer loops, they should have more
  // restrictive conditions. Only the most restrictive one should be
  // used. If this is not done, we could end up having an unswitch
  // predicate like: idx(i) < N && idx(i + 1) < N, which obvously has
  // redundancy. This check is meant to keep only the second term,
  // i.e., idx(i + 1) < N.
  auto is_more_restrictive_loop_stage =
      [&is_start](
          CircularBufferLoopStage new_stage,
          CircularBufferLoopStage existing_stage) -> bool {
    NVF_ERROR(
        existing_stage == CircularBufferLoopStage::Prolog ||
            existing_stage == CircularBufferLoopStage::Main ||
            existing_stage == CircularBufferLoopStage::Epilog ||
            existing_stage == CircularBufferLoopStage::NotApplicable,
        "Unknown stage: ",
        existing_stage);
    NVF_ERROR(
        new_stage == CircularBufferLoopStage::Prolog ||
            new_stage == CircularBufferLoopStage::Main ||
            new_stage == CircularBufferLoopStage::Epilog ||
            new_stage == CircularBufferLoopStage::NotApplicable,
        "Unknown stage: ",
        new_stage);

    // For the start predicate, prologue and main are more restrictive
    // than main and epilogue, respectively.
    // If non circular buffer predicacate exists,
    // that should just work too
    //
    // For the stop predicate, epilogue should be more restrictive
    // than main. If the current stage is prologue or non circular
    // buffer, main or epilogue should be more restrictive.
    if (is_start) {
      return (existing_stage == CircularBufferLoopStage::Main &&
              (new_stage == CircularBufferLoopStage::NotApplicable ||
               new_stage == CircularBufferLoopStage::Prolog)) ||
          (existing_stage == CircularBufferLoopStage::Epilog &&
           (new_stage != CircularBufferLoopStage::Epilog));
    } else {
      return (existing_stage == CircularBufferLoopStage::Main &&
              (new_stage == CircularBufferLoopStage::Epilog)) ||
          (existing_stage == CircularBufferLoopStage::Prolog &&
           (new_stage == CircularBufferLoopStage::Main ||
            new_stage == CircularBufferLoopStage::Epilog)) ||
          (existing_stage == CircularBufferLoopStage::NotApplicable &&
           (new_stage == CircularBufferLoopStage::Main ||
            new_stage == CircularBufferLoopStage::Epilog));
    }
  };

  if (merged_predicate_info.loop_stage !=
          CircularBufferLoopStage::NotApplicable ||
      loop_stage != CircularBufferLoopStage::NotApplicable) {
    NVF_ERROR(
        merged_predicate_info.dynamic_preds.empty(),
        "Dynamic predicates not supported with circular buffering");
    NVF_ERROR(
        merged_predicate_info.static_offset == 0,
        "Non-zero static ofset not supported with circular buffering: ",
        merged_predicate_info.static_offset);
    NVF_ERROR(
        offset->isZero(),
        "Non-zero static ofset not supported with circular buffering: ",
        offset->toInlineString());
    // If merged_predicate_info.static_pred is nullptr, nothing is
    // set yet.
    if (merged_predicate_info.static_pred == nullptr ||
        is_more_restrictive_loop_stage(
            loop_stage, merged_predicate_info.loop_stage)) {
      merged_predicate_info.static_pred = predicate;
      merged_predicate_info.loop_stage = loop_stage;
    }
    return;
  }

  auto offset_int = dynamic_cast<Val*>(offset);
  // If it's a static predicate, replace the current one if it's
  // more restrictive. If it's dynamic, just adds it to the dynamic
  // predicate list.
  if (offset_int && offset_int->isConst()) {
    auto offset_const = offset_int->value();
    auto& static_pred = merged_predicate_info.static_pred;
    auto& static_offset = merged_predicate_info.static_offset;
    if (static_pred == nullptr ||
        is_more_restrictive_static_offset(offset_const, static_offset)) {
      static_pred = predicate;
      static_offset = offset_const;
    }
  } else {
    merged_predicate_info.dynamic_preds.push_back(predicate);
  }
}

UnswitchPredicate::UnswitchPredicate(
    std::vector<ForLoop*> outer_loops,
    ForLoop* unrolled_loop)
    : for_loops_(std::move(outer_loops)), unrolled_loop_(unrolled_loop) {
  openLoop(unrolled_loop);
  finalize();
}

} // namespace nvfuser

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
#include <id_model/utils.h>
#include <index_compute.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <transform_iter.h>

#include <c10/util/irange.h>
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

std::unordered_set<Val*> getNonUnswitchedRootDomains(
    const std::vector<ForLoop*>& loops,
    size_t unswitched_loop_index) {
  std::vector<Val*> non_unswited_loop_domains;
  std::transform(
      loops.begin(),
      loops.begin() + (int64_t)unswitched_loop_index,
      std::back_inserter(non_unswited_loop_domains),
      [&](ForLoop* loop) { return loop->iter_domain(); });

  auto non_unswitched_inputs =
      IterVisitor::getInputsTo(non_unswited_loop_domains);

  auto non_unswitched_root_doms =
      ir_utils::filterByType<IterDomain>(non_unswitched_inputs);

  std::unordered_set<Val*> non_unswitched_concrete_root_domains;

  std::transform(
      non_unswitched_root_doms.begin(),
      non_unswitched_root_doms.end(),
      std::inserter(
          non_unswitched_concrete_root_domains,
          non_unswitched_concrete_root_domains.end()),
      [&](auto root_dom) {
        return GpuLower::current()->caMap()->getConcreteMappedID(
            root_dom, IdMappingMode::EXACT);
      });

  return non_unswitched_concrete_root_domains;
}

bool isFullyUnswitched(
    IterDomain* loop_id,
    const std::unordered_set<Val*>& non_unswitched_root_domains) {
  auto root_vals = IterVisitor::getInputsTo({loop_id});

  auto root_domains = ir_utils::filterByType<IterDomain>(root_vals);

  return std::none_of(
      root_domains.begin(), root_domains.end(), [&](auto root_dom) {
        auto concrete_root_dom =
            GpuLower::current()->caMap()->getConcreteMappedID(
                root_dom, IdMappingMode::EXACT);
        return non_unswitched_root_domains.count(concrete_root_dom) > 0;
      });
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

  for (const auto i : c10::irange(loops.size())) {
    auto loop = loops[i];

    // Parallel dimensions need not be predicated if fully unswitched.
    if (loop == unswitched_loop) {
      within_unswitch = true;
      non_unswitched_root_domains = getNonUnswitchedRootDomains(loops, i);
    }

    auto loop_id = loop->iter_domain();
    auto loop_ptype = loop_id->getParallelType();

    // Not necessary to add a predicate if the paralle type is exact
    if (!isParallelTypeThread(loop_ptype) ||
        lower_utils::isExtentEqualToMaxParallelTypeExtent(loop_id)) {
      continue;
    }
    auto parallel_dim = gpu_lower->parallelDimensionMap().getRaw(loop_ptype);

    // Parallel dimensions need not be predicated if fully unswitched.
    if (within_unswitch &&
        isFullyUnswitched(loop_id, non_unswitched_root_domains)) {
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
  if (ir_utils::isCpAsyncBulk(expr)) {
    RECORD_AND_RETURN(parallel_dom_pred);
  }

  std::vector<PredicateInfo> pred_info_vec;
  if (!ir_utils::hasRootToLoopLinearTransformations(out_tv) ||
      (isIdModelOptionEnabled(IdModelEnableOption::InlinePredicate) &&
       GpuLower::current()->isTensorIndexerEnabled())) {
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

  if (thread_pred != nullptr) {
    std::cerr << "Thread predicate: " << thread_pred->toInlineString() << "\n";
    preds.push_back(thread_pred);
  }

  if (preds.empty()) {
    RECORD_AND_RETURN(GpuLower::current()->kernel()->trueVal());
  }

  Val* cond = preds[0];
  for (const auto i : c10::irange(1, preds.size())) {
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
      (isIdModelOptionEnabled(IdModelEnableOption::UnswitchPredicate) &&
       GpuLower::current()->isTensorIndexerEnabled())) {
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
        std::cerr << "Parallel domain predicate: "
                  << new_info.getPredicate()->toInlineString() << "\n";
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

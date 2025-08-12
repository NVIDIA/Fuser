// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <id_model/indexing.h>
#include <id_model/utils.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <transform_iter.h>

#include <device_lower/analysis/sync_information.h>

namespace nvfuser {

namespace {

// Validate parallelization of a single tensor
void validateParallelizationOfTensor(TensorView* tv) {
  // Each ParallelType can be used only once.
  ParallelTypeBitmap pt_map;
  for (auto i : arange(tv->nDims())) {
    auto axis = tv->axis(i);
    auto ptype = axis->getParallelType();
    if (!isParallelTypeThread(ptype)) {
      continue;
    }

    // It doesn't matter if this axis is a non-concretized broadcast
    // TODO: merging broadcast and non-broadcast
    if (axis->isBroadcast() &&
        !GpuLower::current()->concretizedBroadcastDomains()->isConcretized(
            axis)) {
      continue;
    }

    NVF_ERROR(
        !pt_map.get(ptype),
        "Multiple use of ",
        ptype,
        " in tensor t",
        tv->name(),
        ": ",
        tv);
    pt_map.set(ptype);
  }

  // If this tensor is predicated by a paralel type, it should not be
  // used to parallelize any domain of this tensor

  const auto thread_pred =
      GpuLower::current()->threadPredMap().getPredicateInfo(tv);

  auto predicated_parallel_types = pt_map & thread_pred.limited_types;

  NVF_ERROR(
      predicated_parallel_types.none(),
      "Invalid parallelization of tensor t",
      tv->name(),
      ". The tensor is parallelized with ",
      predicated_parallel_types.toString(),
      ", but it's invalid to use the types as the tensor is also predicated "
      "with them.",
      ", thread pred: ",
      thread_pred.limited_types.toString());
}

// Return true when consumer_id of consumer_tv can accommodate
// incoherent data dependencies.
bool allowIncoherentDependency(
    TensorView* consumer_tv,
    IterDomain* consumer_id) {
  auto def = consumer_tv->definition();
  NVF_ERROR(def != nullptr);

  // In the case of topk, the dependency of the topk IDs are taken
  // care by the topk operation itself.
  if (auto topk = dynamic_cast<TopKOp*>(def)) {
    auto topk_loop_ids = ir_utils::getReachableIds(
        consumer_tv->getLoopDomain(),
        {consumer_tv->getLogicalDomain().at(topk->dim())});
    if (std::ranges::find(topk_loop_ids, consumer_id) != topk_loop_ids.end()) {
      return true;
    }
  }

  return false;
}

// Check if an iter domain of a tensor is a subject of a scatter
// op. Specifically, if the given expr is a scatter op using the given
// tensor as its input, returns true if the given iter domain is
// derived from the scattered logical iter domain.
bool isConsumedByScatter(TensorView* tv, IterDomain* id, Expr* consumer_expr) {
  auto scatter = dynamic_cast<ScatterOp*>(consumer_expr);
  if (scatter == nullptr || scatter->in() != tv) {
    return false;
  }

  auto logical_scatter_dim =
      TensorDomain::noReductions(tv->getLogicalDomain()).at(scatter->dim());
  return DependencyCheck::isDependencyOf(logical_scatter_dim, id);
}

// Check if an iter domain of a tensor is an output of a scatter
// op. All non-scattered IDs should be derived from the non-scattered
// logical IDs. If the given ID is not found in the non-scattered ID
// set, it must be produced by the scatter. Note that we can't just do
// isDependencyOf like isConsumedByScatter since the given ID has no
// dependency with any of the logical IDs of the given tensor since
// the loop domain is set by the index tensor.
bool isProducedByScatter(TensorView* tv, IterDomain* id) {
  auto scatter = dynamic_cast<ScatterOp*>(tv->definition());
  if (scatter == nullptr) {
    return false;
  }

  auto logical_scatter_dim =
      TensorDomain::noReductions(tv->getLogicalDomain()).at(scatter->dim());

  std::unordered_set<Val*> non_scatter_logical_ids;
  std::ranges::copy_if(
      tv->getLogicalDomain(),
      std::inserter(non_scatter_logical_ids, non_scatter_logical_ids.end()),
      [&](IterDomain* logical_id) {
        return logical_id != logical_scatter_dim;
      });

  auto all_non_scatter_ids = DependencyCheck::getAllValsBetween(
      non_scatter_logical_ids,
      {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});

  return std::ranges::find(all_non_scatter_ids, id) ==
      all_non_scatter_ids.end();
}

} // namespace

SyncMap::SyncMap(Fusion* fusion) {
  FUSER_PERF_SCOPE("SyncMap::SyncMap");
  FusionGuard fg(fusion);

  NVF_ERROR(GpuLower::current()->hasIdModel());

  const auto& ca_map = GpuLower::current()->caMap();
  const auto& pred_map = GpuLower::current()->threadPredMap();

  auto exprs = StmtSort::getExprs(fusion);

  // Run through expressions and check for communication across threads/blocks
  // occuring from producer to consumer of the expression
  for (auto expr : exprs) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    // Validate parallelization of each consumer by itself
    for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
      validateParallelizationOfTensor(consumer);
    }

    // It's probably enough to just check all producers to one consumer as
    // multi-consumers are guaranteed to be transformed/parallelized the same,
    // but to be conservative for now checking every producer <-> consumer
    // relationship.
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // Parallelization on input tensors have no effect.
      if (producer->isFusionInput()) {
        continue;
      }

      ParallelTypeBitmap raw_dims;

      const auto parallel_bcast_doms =
          pred_map.getParallelBroadcastDomains(producer);

      // Stash information about parallelized producer iteration domains
      std::vector<IterDomain*> producer_parallel_ids(
          ParallelTypeBitmap::kNumParallelTypes, nullptr);

      // Get the parallel types that producer will be predicated off in producer
      // writes.
      //  In this case we need a sync whether the producer-consumer axes are
      //  mapped or not since the predicate pass will generate pattern like
      //  below to eliminate redundant writes: if(threadIdx.x == 0)
      //    shared[threadIdx.x + i] = ...
      // We will need a raw sync after this pattern for correctness.
      auto producer_redundant_types = GpuLower::current()
                                          ->threadPredMap()
                                          .getPredicateInfo(producer)
                                          .redundant_types;
      // Get the parallel types that are inactive in consumer's use chains.
      auto producer_redundant_use_types = GpuLower::current()
                                              ->threadPredMap()
                                              .getPredicateInfo(producer)
                                              .redundant_use_types;

      // In sync info pass we only consider the parallel types in
      //  producer that are redundantly produced but not redundantly consumed.
      producer_redundant_types =
          producer_redundant_types & (~producer_redundant_use_types);

      for (const auto producer_i : arange(producer->nDims())) {
        auto producer_axis = producer->getLoopDomain().at(producer_i);
        auto producer_ptype =
            ca_map->getConcreteMappedID(producer_axis, IdMappingMode::LOOP)
                ->getParallelType();

        if (!isParallelTypeThread(producer_ptype)) {
          continue;
        }

        // Producer reductions shouldn't map to consumers
        if (producer_axis->isReduction()) {
          continue;
        }

        producer_parallel_ids[getParallelTypeBitMapOffset(producer_ptype)] =
            producer_axis;
      }

      for (auto consumer :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        // Stash information about parallelized consumer iteration domains
        std::vector<IterDomain*> consumer_parallel_ids(
            ParallelTypeBitmap::kNumParallelTypes, nullptr);
        for (const auto consumer_i : arange(consumer->nDims())) {
          auto consumer_axis = consumer->getLoopDomain().at(consumer_i);
          auto consumer_ptype =
              ca_map->getConcreteMappedID(consumer_axis, IdMappingMode::LOOP)
                  ->getParallelType();

          if (!isParallelTypeThread(consumer_ptype)) {
            continue;
          }

          // When the consumer axis is a broadcast, it is not really
          // parallelized unless thread-predicated and eventually concretized
          if (consumer_axis->isBroadcast() &&
              (!parallel_bcast_doms.get(consumer_ptype) ||
               !GpuLower::current()
                    ->concretizedBroadcastDomains()
                    ->isConcretized(consumer_axis))) {
            continue;
          }

          consumer_parallel_ids[getParallelTypeBitMapOffset(consumer_ptype)] =
              consumer_axis;
        }

        // P2C map is required when using the IdModel-based analysis
        const std::unordered_map<IterDomain*, IterDomain*>
            p2c_map_no_forwarding =
                BestEffortReplay(
                    consumer->getLoopDomain(),
                    producer->getLoopDomain(),
                    PairwiseLogicalDomainMap(producer, consumer)
                        .mapProducerToConsumer(),
                    /*replay_forward_id_map=*/{},
                    /*target_forward_id_map=*/{},
                    /*skip_replay_swizzle=*/false,
                    /*skip_target_swizzle=*/false,
                    /*skip_resize=*/false,
                    /*error_on_failure=*/false)
                    .getReplay();

        // At this point each parallel type that's present in the consumer or
        // the producer will be present in their corresponding `_parallel_ids`
        // map going from parallel index type (only size 6 for grid/block dims)
        // to the iteration domain of that parallel type.
        for (auto parallel_type : kParallelTypeThreads) {
          // TIDx is reserved for lane_id in the case of mma ops.
          //  It is swizzled and handled separately in validateMma.
          if (parallel_type == ParallelType::TIDx && expr->isA<MmaOp>()) {
            continue;
          }

          // In the case when the parallel id's are mapped by ca map,
          //   will additionally need to consider if the producer is
          //   a redundant write. The raw dim can be skipped only if
          //   consumer use chains only contain redundant uses.
          //  TODO:
          //    still losing a bit precision here for expr ordering
          //  sensitive cases, but we could wait until that becomes
          //  a perf limiter to fix.
          if (producer_redundant_types.get(parallel_type)) {
            raw_dims.set(parallel_type);
            continue;
          }

          auto parallel_type_i = getParallelTypeBitMapOffset(parallel_type);

          auto p_id = producer_parallel_ids[parallel_type_i];
          auto c_id = consumer_parallel_ids[parallel_type_i];

          if (p_id == nullptr && c_id == nullptr) {
            continue;
          } else if (p_id != nullptr && c_id == nullptr) {
            auto it = std::find_if(
                consumer->getLoopDomain().begin(),
                consumer->getLoopDomain().end(),
                [&](IterDomain* c_id) {
                  return GpuLower::current()->caMap()->areMapped(
                      p_id, c_id, IdMappingMode::PERMISSIVE);
                });

            // If there isn't a mapping from producer to a consumer domain,
            // need to assume there's communication across this parallel
            // dimension.
            c_id = it == consumer->getLoopDomain().end() ? nullptr : *it;
            // i.e. if producer is parallelized across threadIdx.x in a
            // certain split, if the consumer doesn't map to this split,
            // then we need to assume it has to be in smem with proper
            // syncs.
          } else if (p_id == nullptr && c_id != nullptr) {
            auto it = std::find_if(
                producer->getLoopDomain().begin(),
                producer->getLoopDomain().end(),
                [&](IterDomain* p_id) {
                  return GpuLower::current()->caMap()->areMapped(
                      p_id, c_id, IdMappingMode::PERMISSIVE);
                });
            if (it == producer->getLoopDomain().end()) {
              // Can't infer anything if producer doesn't have a matching axis
              // to parallel consumer dim.
              continue;
            }
            p_id = *it;
          }

          // Comm pattern options (when parallel types don't have matching
          // axes) and required memory, Chart is producer parallel type,
          // consumer parallel type Parallel types are Serial(S),
          // threadIdx(T), blockIdx(B), Memory required for the producer is
          // Local(L), Shared(S), Global(G), Sync is None (N/A), blockSync(B),
          // grid_sync(G)
          //
          // P    C   Mem Req   Sync Type
          // S    S      L          N/A
          // S    T      L          N/A
          // S    B      L          N/A
          // T    S      S           B
          // T    T      S           B
          // T    B      S           B
          // B    S      G           G
          // B    T      G           G
          // B    B      G           G

          auto producer_ptype =
              ca_map->getConcreteMappedID(p_id, IdMappingMode::LOOP)
                  ->getParallelType();
          auto consumer_ptype = c_id == nullptr
              ? ParallelType::Serial
              : ca_map->getConcreteMappedID(c_id, IdMappingMode::LOOP)
                    ->getParallelType();

          auto producer_parallel_bcast = p_id->isBroadcast() &&
              isParallelTypeThread(producer_ptype) &&
              parallel_bcast_doms.get(producer_ptype) &&
              GpuLower::current()->concretizedBroadcastDomains()->isConcretized(
                  p_id);

          auto producer_parallelized = isParallelTypeThread(producer_ptype) &&
              (!p_id->isBroadcast() || producer_parallel_bcast);

          // Handle special cases first

          // If any loop id of producer is block or grid parallel and is
          // involved
          //  in any swizzle pattern, track this parallel dim as a communication
          //  dimension that requires the corresponding synchronization and
          //  memory type.
          if (isParallelTypeThread(producer_ptype) &&
              producer->hasSwizzleOp()) {
            if (!ir_utils::getAllSwizzlesBetween(
                     producer->getLogicalDomain(), {p_id})
                     .empty()) {
              raw_dims.set(producer_ptype);
              continue;
            }
          }

          // When the producer axis is not parallelized, no sync is
          // necessary
          if (!producer_parallelized) {
            continue;
          }

          // Certain operations resolve data dependencies by
          // themselves, thus not requiring a RAW sync
          if (allowIncoherentDependency(consumer, c_id)) {
            continue;
          }

          if (producer_ptype == consumer_ptype) {
            // Case 1:
            // Producer loop ID: non-broadcast
            // Consumer loop ID: non-broadcast
            // -> No sync if they are exactly mapped. This case is covered by
            // the promotion check.
            //
            // Case 2:
            // Producer loop ID: broadcast (which may be produced by
            // merging multiple broadcast domains)
            // Consumer loop ID: non-broadcast
            // -> They are not exactly mapped but sync is not necessary as
            // discussed below.
            //
            // Case 3:
            // Producer loop ID: non-broadcast
            // Consumer loop ID: non-broadcast
            // -> Sync required if they are not exactly mapped, even when they
            // are mapped by the best effort replay. (See
            // NVFuserTest.RAWSync for a concrete repro).

            // Case 1. Note that indexing through scatter needs to be
            // excluded due to its indirect indexing.
            const auto& id_model = GpuLower::current()->idModel();
            auto producer_loop_id = getLoopPromotion(p_id, id_model);
            auto consumer_loop_id = getLoopPromotion(c_id, id_model);
            const auto& indexing_traveral_graph =
                id_model.idGraph(TensorIndexer::traversalGraphType());
            if (indexing_traveral_graph.disjointValSets().strictAreMapped(
                    producer_loop_id, consumer_loop_id) &&
                !isConsumedByScatter(producer, p_id, expr) &&
                !isProducedByScatter(producer, p_id)) {
              continue;
            }

            // Case 2
            // If the producer ID is a broadcast, it does not
            // require synchronization even when the producer and
            // consumer domains are not promoted to the same
            // group. For example,
            //
            // tv0: [i0]
            // tv1: [b1]
            // tv2 = tv1
            // tv3 = tv0 + tv2
            //
            // tv2->axis(0)->parallelize(ParallelType::TIDx);
            // tv3->axis(0)->parallelize(ParallelType::TIDx);
            //
            // Assume that there's no inlining. Since it isn't
            // inlined, the loop domain of tv2 is not mapped with
            // that of tv3, thus the avove condition won't
            // hit. Still, since tv2 will be executed by all TIDx
            // threads independently, there's no need of
            // synchronization.
            //
            // Consider a similar case like below:
            //
            // tv0: [i0, i1]
            // tv1: [i2, b3]
            // tv2 = tv1
            // tv3 = tv0 + tv2
            //
            // tv2->merge(0, 1);
            // tv3->merge(0, 1);
            // tv2->axis(0)->parallelize(ParallelType::TIDx);
            // tv3->axis(0)->parallelize(ParallelType::TIDx);
            //
            // This case does require a synchronization since for
            // tv2, TIDx will be used to parallelize the outer
            // domain only, whereas for tv3 it is mapped to the
            // merged domain of the outer and inner domains. In
            // other words, if a broadcast becomes non-broadcast
            // by getting merged with a non-broadcast domain, it
            // requires a synchronization.
            if (p_id->isBroadcast()) {
              if (auto it = p2c_map_no_forwarding.find(p_id);
                  it != p2c_map_no_forwarding.end() && it->second == c_id) {
                continue;
              }
            }
          }

          raw_dims.set(producer_ptype);
        } // end for ptypes

        if (raw_dims.hasBID()) {
          NVF_ERROR(
              producer->getMemoryType() == MemoryType::Global,
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer->toString(),
              ") and TV",
              consumer->name(),
              "(",
              consumer->toString(),
              "). Producer is required to be in Global Memory based on "
              "parallelization strategy.",
              " RAW flags: ",
              raw_dims.toString());
        } else if (raw_dims.hasTID()) {
          NVF_ERROR(
              ir_utils::isLdMatrixOp(producer->definition()) ||
                  ir_utils::isStMatrixOp(consumer->definition()) ||
                  producer->getMemoryType() == MemoryType::Global ||
                  producer->getMemoryType() == MemoryType::Shared ||
                  producer->getMemoryType() == MemoryType::Tensor,
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer->toString(),
              ") and TV",
              consumer->name(),
              "(",
              consumer->toString(),
              "). Producer is required to be in Global, Shared or Tensor "
              "Memory based on parallelization strategy.",
              " RAW flags: ",
              raw_dims.toString());
        }

      } // end for consumers

      if (raw_dims.any()) {
        needs_raw_sync_[producer] |= raw_dims;
      }
    } // end producer
  }
}

std::string SyncMap::toString() const {
  std::stringstream ss;
  ss << "SyncMap:";
  std::vector<TensorView*> sorted_tvs;
  std::transform(
      needs_raw_sync_.begin(),
      needs_raw_sync_.end(),
      std::back_inserter(sorted_tvs),
      [](auto kv) { return kv.first; });
  std::sort(
      sorted_tvs.begin(),
      sorted_tvs.end(),
      [](TensorView* tv1, TensorView* tv2) {
        return tv1->name() < tv2->name();
      });
  bool is_first = true;
  for (auto tv : sorted_tvs) {
    if (!is_first) {
      ss << ",";
    }
    ss << " " << tv->toString() << " -> " << needs_raw_sync_.at(tv).toString();
    is_first = false;
  }
  return ss.str();
}

} // namespace nvfuser

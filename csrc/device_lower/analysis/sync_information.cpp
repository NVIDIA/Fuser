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
#include <id_model/indexing_utils.h>
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
  for (auto i : c10::irange(tv->nDims())) {
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
      ", but it's invalid to use the types as the tensor is also predicated with them.",
      ", thread pred: ",
      thread_pred.limited_types.toString());
}

//! Properties used in useSameIndex that only depends on the producer and
//! consumer tensors and can be reused for validating different pairs
//! of their loop IDs. Works as caching as some properties can be
//! expensive to compute.
struct ProducerConsumerIndexingInfoCache {
 public:
  ProducerConsumerIndexingInfoCache(
      TensorView* producer_tv,
      TensorView* consumer_tv)
      : producer_tv_(producer_tv), consumer_tv_(consumer_tv) {}

  const std::vector<IterDomain*>& getConsumerLeafIDsSharedWithProducer() {
    if (!consumer_loop_ids_shared_with_producer_.has_value()) {
      const auto& ca_map = *(GpuLower::current()->caMap());
      std::vector<IterDomain*> consumer_loop_ids_shared_with_producer;
      std::copy_if(
          consumer_tv_->getLoopDomain().begin(),
          consumer_tv_->getLoopDomain().end(),
          std::back_inserter(consumer_loop_ids_shared_with_producer),
          [&](auto consumer_loop_id) {
            return std::find_if(
                       producer_tv_->getLoopDomain().begin(),
                       producer_tv_->getLoopDomain().end(),
                       [&](auto producer_loop_id) {
                         return ca_map.areMapped(
                             producer_loop_id,
                             consumer_loop_id,
                             IdMappingMode::LOOP);
                       }) != producer_tv_->getLoopDomain().end();
          });
      consumer_loop_ids_shared_with_producer_ =
          std::move(consumer_loop_ids_shared_with_producer);
    }
    return *consumer_loop_ids_shared_with_producer_;
  }

  const std::vector<Val*>& getConsumerRootIDsSharedWithProducer() {
    if (!consumer_root_ids_shared_with_producer_.has_value()) {
      const auto& consumer_loop_ids_shared_with_producer =
          getConsumerLeafIDsSharedWithProducer();
      consumer_root_ids_shared_with_producer_ = InputsOf::outputs(
          {consumer_loop_ids_shared_with_producer.begin(),
           consumer_loop_ids_shared_with_producer.end()});
    }
    return *consumer_root_ids_shared_with_producer_;
  }

  const std::vector<IterDomain*>& getConsumerOnlyPermissiveLeafIds() {
    // When a given ID is the factor of 1 of a split, return the other
    // output. Return nullptr otherwise.
    auto get_split1_other_out = [](IterDomain* id) -> IterDomain* {
      if (id->extent()->isOneInt() && id->definition() != nullptr &&
          id->definition()->isA<Split>()) {
        auto split = id->definition()->as<Split>();
        if (split->innerSplit() && split->inner() == id) {
          return split->outer();
        } else if (!split->innerSplit() && split->outer() == id) {
          return split->inner();
        }
      }
      return nullptr;
    };

    if (!consumer_only_permissive_loop_ids_.has_value()) {
      // consumer_only_permissive_loop_ids_ = {};
      std::vector<IterDomain*> consumer_only_permissive_loop_ids;
      const auto& ca_map = *(GpuLower::current()->caMap());
      std::copy_if(
          consumer_tv_->getLoopDomain().begin(),
          consumer_tv_->getLoopDomain().end(),
          std::back_inserter(consumer_only_permissive_loop_ids),
          [&](IterDomain* consumer_loop_id) {
            const auto& consumer_loop_ids_shared_with_producer =
                getConsumerLeafIDsSharedWithProducer();
            if (std::find(
                    consumer_loop_ids_shared_with_producer.begin(),
                    consumer_loop_ids_shared_with_producer.end(),
                    consumer_loop_id) !=
                consumer_loop_ids_shared_with_producer.end()) {
              return false;
            }

            auto loop_concrete_id = ca_map.getConcreteMappedID(
                consumer_loop_id, IdMappingMode::LOOP);

            // If the loop concrete ID has the same info as the
            // consumer loop ID, indexing shouldn't be affected by the
            // loop concrete ID
            if (ca_map.areMapped(
                    consumer_loop_id,
                    loop_concrete_id,
                    IdMappingMode::ALMOSTEXACT)) {
              return false;
            }

            // Note that the factor output domain of split-by-one is
            // not mapped in the almost exact map. As long as the
            // other domains are almost-exactly mapped, this shouldn't
            // affect the indexing neither.
            auto consumer_split1_other = get_split1_other_out(consumer_loop_id);
            auto loop_concrete_split1_other =
                get_split1_other_out(loop_concrete_id);

            if (consumer_split1_other != nullptr &&
                loop_concrete_split1_other != nullptr &&
                ca_map.areMapped(
                    consumer_split1_other,
                    loop_concrete_split1_other,
                    IdMappingMode::ALMOSTEXACT)) {
              return false;
            }

            return true;
          });
      consumer_only_permissive_loop_ids_ =
          std::move(consumer_only_permissive_loop_ids);
    }
    return *consumer_only_permissive_loop_ids_;
  }

  const VectorOfUniqueEntries<IterDomain*>& getConsumerLoopIndexingIDs() {
    if (!consumer_loop_indexing_ids_.has_value()) {
      consumer_loop_indexing_ids_ =
          LoopIndexingAnalysis::getReplayableConcreteIDs(
              getConsumerOnlyPermissiveLeafIds(), consumer_tv_);
    }
    return *consumer_loop_indexing_ids_;
  }

 private:
  TensorView* producer_tv_ = nullptr;
  TensorView* consumer_tv_ = nullptr;
  // Consumer loop IDs that are also used to index the producer, i.e.,
  // those that are loop-mapped with the producer loop IDs
  std::optional<std::vector<IterDomain*>>
      consumer_loop_ids_shared_with_producer_;
  // Root IDs of the shared loop IDs
  std::optional<std::vector<Val*>> consumer_root_ids_shared_with_producer_;
  // Consumer CA loop IDs that are not shared with producer and
  // permissively mapped with consumers of the consumer
  std::optional<std::vector<IterDomain*>> consumer_only_permissive_loop_ids_;
  // IDs whose index depends on consumer_only_permissive_loop_ids_
  std::optional<VectorOfUniqueEntries<IterDomain*>> consumer_loop_indexing_ids_;
};

// For a given pair of a producer and consumer loop ID, check if the
// root domains that have dependencies with them are guaranteed to
// have the same index.
//
// The algorithm first sees if the root domains reachable from the
// consumer domain are all exactly mapped with the root domains
// reachable from the producer domain. This is to detect merged
// broadcast domains that only show up in the consumer. If such a
// consumer-only root domain is found, it can mean the producer and
// consumer are indexed differently, but not always. If there's a
// consumer loop ID that is shared with the producer through
// computeAt, and if there's a dependency from the loop ID to the
// consumer-only root ID, the producer indexing also uses the shared
// consumer ID and the indexing traversal reach at the consumer-only
// broadcast root domain, generating the same index as that of the
// consumer.
//
// It is also necessary to check non-CA-shared consumer loop IDs that
// are permissively mapped with its consumers. See inline comments
// below.
bool useSameIndex(
    TensorView* producer_tv,
    IterDomain* producer_id,
    TensorView* consumer_tv,
    IterDomain* consumer_id,
    ProducerConsumerIndexingInfoCache& indexing_info) {
  const auto& ca_map = *(GpuLower::current()->caMap());

  // At least, they must be mapped exactly or permissively
  if (!ca_map.areMapped(producer_id, consumer_id, IdMappingMode::EXACT) &&
      !ca_map.areMapped(producer_id, consumer_id, IdMappingMode::PERMISSIVE)) {
    return false;
  }

  // If the producer ID is mapped with any of the consumer IDs, the
  // indexing is done with the corresponding consumer ID
  if (std::any_of(
          consumer_tv->getLoopDomain().begin(),
          consumer_tv->getLoopDomain().end(),
          [&](IterDomain* consumer_loop_id) {
            return ca_map.areMapped(
                consumer_loop_id, producer_id, IdMappingMode::LOOP);
          })) {
    return true;
  }

  // Grab all consumer root IDs that have the threading index of
  // consumer_id. The goal of the analysis below is to find out if all
  // of the root IDs are indexed in the same way between the producer
  // and consumer tensors.
  auto consumer_root_ids = InputsOf::output(consumer_id);

  auto producer_logical_vals = StmtSort::getStmtsBetween(
      {producer_tv->getLogicalDomain().begin(),
       producer_tv->getLogicalDomain().end()},
      {producer_id});
  auto producer_logical_ids =
      ir_utils::filterByType<IterDomain>(producer_logical_vals);

  // For each of the root IDs that consumer_id is dependent on, check
  // if the producer uses the same indexing as the consumer. This
  // requires that the producer has a root ID that is exactly mapped with
  // the consumer root ID. Another case is when the consumer root ID
  // has a dependency with any of the loop consumer IDs that are
  // shared with the producer. In that case, the producer uses those
  // shared consumer loop IDs to index the root ID and thus uses the same index
  if (!std::all_of(
          ir_utils::filterByType<IterDomain>(consumer_root_ids).begin(),
          ir_utils::filterByType<IterDomain>(consumer_root_ids).end(),
          [&](IterDomain* consumer_root_id) {
            return std::find_if(
                       producer_logical_ids.begin(),
                       producer_logical_ids.end(),
                       [&](IterDomain* producer_root_id) {
                         return ca_map.areMapped(
                             producer_root_id,
                             consumer_root_id,
                             IdMappingMode::EXACT);
                       }) != producer_logical_ids.end() ||
                std::find(
                    indexing_info.getConsumerRootIDsSharedWithProducer()
                        .begin(),
                    indexing_info.getConsumerRootIDsSharedWithProducer().end(),
                    consumer_root_id) !=
                indexing_info.getConsumerRootIDsSharedWithProducer().end();
          })) {
    return false;
  }

  // At this point, consumer_root_ids is the set of root IDs that
  // commonly have dependencies with producer_id and consumer_id.
  //
  // It is also necessary to look at consumer loop IDs that are
  // computed-at its consumers, which means the consumer is indexed
  // using its consumer domains. Unless such IDs are also shared with the
  // producer, the consumer may have a different index as that of the
  // producer.

  // Example:
  // t0: [I0], t1: [I0, I1]
  // t2 = t0
  // t3 = broadcast(t2, {true, false})
  // t4 = t3 + t1
  //
  // t0: [I0]
  // t1: [I0, I1]
  // t2: [I0]
  // t3: [I0, B0]
  // t4: [I0, I1]
  //
  // t4->merge(0)->split(0, 4)
  // propagate t4 transformations
  // parallelize axis(-1) with tidx
  //
  // t0: [I0/4, tidx(4)]
  // t1: [I0*I1/4, tidx(4)]
  // t2: [I0/4, tidx(4)]
  // t3: [I0*B0/4, tidx(4)]
  // t4: [I0*I1/4, tidx(4)]
  //
  // t2->computeAt(t4, 1)
  //
  // t0: [I0/4, tidx(4)]
  // t1: [I0*I1/4, tidx(4)]
  // t2: [I0/4, tidx(4)] ca(1)
  // t3: [I0*B0/4, tidx(4)] ca(1)
  // t4: [I0*I1/4, tidx(4)] produce(1)
  //
  // The interesting part here is t0 and t2. They are completely
  // exactly mapped, but the CA of t2 makes it indexed based on its
  // consumer, t4. Specifically, the code would look like:
  //
  // for (i: I0/4)
  //   t0[i * bdimx + tidx] = ...
  // for (i: I0*I1/4)
  //   t2[(i * bdimx + tidx) % bdimx] = t0[...]
  //   t3[(i * bdimx + tidx) % bdimx] = t2[...]
  //   t4[i * bdimx + tidx] = t3[...] + t1[...]
  //
  // t2->axis(0) is an example of consumer-only loop IDs that are
  // permissively mapped with consumers of consumers. Since it's
  // effectively replaced with t4->axis(0) when indexing t2, whereas
  // t0 is independently indexed, t0 must be placed on shared memory
  // (or global memory) with a RAW sync. See See FusionValidateParallelize10.
  //
  // For the same original fusion, consider this transformation:
  //
  // t4->merge(0)->split(0, 4)->split->(0, 2)
  // propagate t4 transformations
  // parallelize axis(-1) with tidx
  //
  // t0: [I0/4/2, 2, tidx(4)]
  // t1: [I0*I1/4/2, 2, tidx(4)]
  // t2: [I0/4/2, 2, tidx(4)]
  // t3: [I0*B0/4/2, 2, tidx(4)]
  // t4: [I0*I1/4/2, 2, tidx(4)]
  //
  // t0->computeAt(t4, 1)
  //
  // t0: [I0/4/2, 2, tidx(4)] ca(1)
  // t1: [I0*I1/4/2, 2, tidx(4)]
  // t2: [I0/4/2, 2, tidx(4)] ca(1)
  // t3: [I0*B0/4/2, 2, tidx(4)]
  // t4: [I0*I1/4/2, 2, tidx(4)] produce(1)
  //
  // For t1 and t2, t2->axis(1) is again a consumer-only loop ID
  // permissively mapped with its consumer. However, in this case, t0
  // also shares the first loop ID with t2 and t4, making it indexed
  // using t4.
  //
  // for (i: I0*I1/4/2)
  //   for (j: 2)
  //     t0[((i * 2 + j) * bdimx + tidx) % bdimx] = ...
  //   for (j: 2)
  //     t2[((i * 2 + j) * bdimx + tidx) % bdimx] = t0[...]
  //   for (j: 2)
  //     t3[((i * 2 + j) * bdimx + tidx) % bdimx] = t2[...]
  //   for (j: 2)
  //     t4[(i * 2 + j) * bdimx + tidx] = t3[...] + t1[...]
  //
  // All of the tensors are indexed consistently, so no RAW sync is
  // required in this case. See FusionValidateParallelize11.

  // If there's no consumer-only loop ID that is permissively mapped
  // with its consumers, this pair of producer and consumer indices
  // should be used in the same way
  if (indexing_info.getConsumerOnlyPermissiveLeafIds().empty()) {
    return true;
  }

  return std::all_of(
      ir_utils::filterByType<IterDomain>(consumer_root_ids).begin(),
      ir_utils::filterByType<IterDomain>(consumer_root_ids).end(),
      [&](IterDomain* consumer_root_id) {
        // If the consumer root ID is part of the shared root IDs
        // with the producer, it is guaranteed to be indexed in
        // the same way. See the second example above.
        if (std::find(
                indexing_info.getConsumerRootIDsSharedWithProducer().begin(),
                indexing_info.getConsumerRootIDsSharedWithProducer().end(),
                consumer_root_id) !=
            indexing_info.getConsumerRootIDsSharedWithProducer().end()) {
          return true;
        }

        // Check if the consumer root ID has a dependency with any
        // of the consumer-only loop IDs. If so, its index may be
        // different from the producer. The dependency here means
        // the indexing traversal from the LOOP concrete domains of the
        // loop IDs. It's not just enough to do normal backward
        // travesal from the concrete domains as they may come from
        // post-view tensors.
        return !indexing_info.getConsumerLoopIndexingIDs().has(
            ca_map.getConcreteMappedID(consumer_root_id, IdMappingMode::EXACT));
      });
}

} // namespace

SyncMap::SyncMap(Fusion* fusion) {
  FUSER_PERF_SCOPE("SyncMap::SyncMap");
  FusionGuard fg(fusion);

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

      for (const auto producer_i : c10::irange(producer->nDims())) {
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
        for (const auto consumer_i : c10::irange(consumer->nDims())) {
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

        ProducerConsumerIndexingInfoCache indexing_info(producer, consumer);

        // P2C map is required when using the IdModel-based analysis
        const std::unordered_map<IterDomain*, IterDomain*>
            p2c_map_no_forwarding = GpuLower::current()->hasIdModel()
            ? BestEffortReplay(
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
                  .getReplay()
            : std::unordered_map<IterDomain*, IterDomain*>{};

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

          // Use the IdModel loop promotion when available. This is
          // required for tensors with non-trivial loop domains
          if (GpuLower::current()->hasIdModel()) {
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

              // Case 1
              const auto& id_model = GpuLower::current()->idModel();
              auto producer_loop_id =
                  indexing_utils::getLoopPromotion(p_id, id_model);
              auto consumer_loop_id =
                  indexing_utils::getLoopPromotion(c_id, id_model);
              const auto& indexing_traveral_graph =
                  id_model.idGraph(TensorIndexer::traversalGraphType());
              if (indexing_traveral_graph.disjointValSets().strictAreMapped(
                      producer_loop_id, consumer_loop_id)) {
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
          } else {
            // When the producer is parallelized, the producer and the
            // consumer must use the same index with the same parallel
            // type. Otherwise, a sync is required. This is not the case
            // when this op is a parallel broadcast.
            if (producer_parallel_bcast) {
              // As long as they are permissively mapped using the same
              // parallel type, no communication is required
              if (producer_ptype == consumer_ptype &&
                  ca_map->areMapped(p_id, c_id, IdMappingMode::PERMISSIVE)) {
                continue;
              }
              // Can this happen?
              NVF_THROW(
                  "Unexpected case. Producer: ",
                  producer->toString(),
                  ", consumer: ",
                  consumer->toString());
            }
            if (producer_ptype == consumer_ptype) {
              if (useSameIndex(producer, p_id, consumer, c_id, indexing_info)) {
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
              "). Producer is required to be in Global Memory based on parallelization strategy.",
              " RAW flags: ",
              raw_dims.toString());
        } else if (raw_dims.hasTID()) {
          NVF_ERROR(
              ir_utils::isLdMatrixOp(producer->definition()) ||
                  ir_utils::isStMatrixOp(consumer->definition()) ||
                  producer->getMemoryType() == MemoryType::Global ||
                  producer->getMemoryType() == MemoryType::Shared,
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer->toString(),
              ") and TV",
              consumer->name(),
              "(",
              consumer->toString(),
              "). Producer is required to be in Global or Shared Memory based on parallelization strategy.",
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

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/utils.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>
#include <scheduler/tools/inlining.h>
#include <transform_iter.h>

#include <utility>

namespace nvfuser {

MaxPosCalculator::MaxPosCalculator(
    std::unordered_set<IterDomain*> uninlinable_ids,
    bool compute_at_only)
    : uninlinable_ids_(std::move(uninlinable_ids)) {
  buildUnmappableDims(compute_at_only);
}

void MaxPosCalculator::buildUnmappableDims(bool compute_at_only) {
  // When used for computeAt only, i.e., without inlining storeAt
  // positions, the restriction below does not apply
  if (compute_at_only) {
    return;
  }
  ComputeAtLogicalDomainMap logical_map;
  logical_map.build();
  auto all_tvs = FusionGuard::getCurFusion()->allTvs();
  for (auto tv : all_tvs) {
    auto consumers = ir_utils::consumerTvsOf(tv);
    for (auto consumer : consumers) {
      // Grab dimensions in producer and consumer that are mappable to eachother
      // based on the computeAtLogicalDomainMap. This will tell us which
      // dimensions can be inlined based on avoiding trying to inline reduction
      // structures.
      auto mappable_roots =
          logical_map.getMappableDims(tv->domain(), consumer->domain());
      for (auto tv_logical_id : tv->getLogicalDomain()) {
        if (mappable_roots.find(tv_logical_id) == mappable_roots.end() &&
            !ir_utils::isSqueezedID(tv, tv_logical_id)) {
          unmappable_dims_.emplace(tv_logical_id);
        }
      }
    }
  }
}

const ValGraph& MaxPosCalculator::inliningGraph() {
  if (id_model_.get() == nullptr) {
    id_model_ = std::make_unique<IdModel>(
        FusionGuard::getCurFusion(), /*build_graphs=*/false);
    id_model_->buildBroadcastGraph();
  }

  return id_model_->idGraph(IdMappingMode::BROADCAST);
}

bool MaxPosCalculator::isAllowedID(
    IterDomain* id,
    TensorView* tv,
    bool best_effort,
    bool allow_reduction,
    bool allow_vectorize,
    bool allow_unmappable) const {
  bool allowed = true;

  if (!allow_reduction) {
    allowed = allowed && !id->isReduction();
  }

  if (uninlinable_ids_.count(id)) {
    return false;
  }

  if (!allow_vectorize) {
    // Avoid inlining if marked as Vectorize or Group. In the case of
    // BestEffort and MostInlined modes, avoid Unroll as well.
    bool is_vectorize = isParallelTypeVectorize(id->getParallelType()) ||
        id->getParallelType() == ParallelType::Group ||
        (best_effort && id->getParallelType() == ParallelType::Unroll) ||
        id->getParallelType() == ParallelType::Bulk;
    allowed = allowed && !is_vectorize;
  }

  if (!allow_unmappable) {
    const auto& logical_dom = tv->getLogicalDomain();
    std::unordered_set<Val*> logical_dom_set(
        logical_dom.begin(), logical_dom.end());
    auto all_vals =
        getValsBetween<IRBFS>({logical_dom.begin(), logical_dom.end()}, {id});
    bool is_unmappable = false;
    for (auto val : all_vals) {
      auto id = val->as<IterDomain>();
      if (logical_dom_set.count(val) > 0 && unmappable_dims_.count(id) > 0) {
        is_unmappable = true;
        break;
      }
    }
    allowed = allowed && !is_unmappable;
  }

  return allowed;
}

size_t MaxPosCalculator::getMaxPosSelf(
    TensorView* tv,
    bool best_effort,
    bool allow_reduction,
    bool allow_vectorize,
    bool allow_unmappable) const {
  const auto& dom = tv->getLoopDomain();
  auto iter = std::find_if(
      dom.begin(),
      dom.end(),
      [this,
       tv,
       best_effort,
       allow_reduction,
       allow_vectorize,
       allow_unmappable](IterDomain* id) {
        return !isAllowedID(
            id,
            tv,
            best_effort,
            allow_reduction,
            allow_vectorize,
            allow_unmappable);
      });
  return std::distance(dom.begin(), iter);
}

// Return the max position in producer that can be inlined to consumer
// Cannot inline:
//   Vectorized dimensions in consumer
//   Unrolled dimensions in consumer
size_t MaxPosCalculator::getMaxProducerPosFromConsumer(
    TensorView* producer,
    TensorView* consumer,
    bool best_effort) {
  // Here, we have two methods to analayze inlining positions. One is
  // the legacy BestEffortReplay-based one that allow forwarding of
  // broadcast domains. This is the default method when both the
  // producer and consumer tensors have loop domains that are derived
  // from logical domains.
  //
  // When either of the two tensors has non-conventional loop domains
  // through TensorView::setLoopDomain, IdModel-based analysis is
  // required. At this moment, the IdModel-based analysis does not
  // implement the broadcast forwarding, and therefore inlining of
  // broadcast-merged domains does not work with this approach. In
  // fact, it is likely we don't implement the forwarding with this
  // approach as it may not be necessary.
  //
  // At this moment, in order to keep the existing behavior as is and
  // at the same time allow inlinig with explicitly set loop domains,
  // the legacy method is used whenever both the producer and consumer
  // have loop domains that are derived from their logical domains
  // with no redundancy. Otherwise, the IdModel-based method is used.

  // TODO: Consider caching these properties in TensorView as they
  // could only change with setLoopDomain
  const bool may_need_forwarding =
      ir_utils::isLoopDomainFullyDerivedFromLogicalDomain(producer) &&
      ir_utils::isLoopDomainFullyDerivedFromLogicalDomain(consumer);

  if (may_need_forwarding) {
    auto pairwise_logical_map = PairwiseLogicalDomainMap(producer, consumer);
    auto replay_CasP = BestEffortReplay::replayCasP(
        consumer, producer, -1, pairwise_logical_map);
    auto p2c_replay_map = replay_CasP.getReplay();

    for (const auto producer_pos : arange(producer->nDims())) {
      // If the producer position is mismatching with the consumer, then we can
      // not inline into this position, otherwise the max producer position of
      // the consumer will become invalid and expression sort will fail.
      if (TransformReplay::getMatchedLeafPosWithoutReplayCasP(
              consumer, producer, producer_pos + 1) < 0) {
        return producer_pos;
      }
      auto map_it = p2c_replay_map.find(producer->axis(producer_pos));
      if (map_it != p2c_replay_map.end()) {
        auto c_id = map_it->second;
        if (!isAllowedID(c_id, consumer, best_effort, true, false, true)) {
          return producer_pos;
        }
      }
    }
    return producer->nDims();
  } else {
    auto consumer_it = consumer->getLoopDomain().begin();
    for (const auto producer_pos : arange(producer->nDims())) {
      auto p_id = producer->getLoopDomain().at(producer_pos);
      // When p_id is a reduction, skip and continue to the next
      // position. Since a producer reduction domain is never allowed
      // to be inlined, it may make more sense to stop the analysis
      // here. For now, just follow the same logic as
      // getMatchedLeafPosWithoutReplayCasP, which simply skips
      // reduction domains.
      if (p_id->isReduction()) {
        continue;
      }

      if (consumer_it == consumer->getLoopDomain().end()) {
        return producer_pos;
      }

      IterDomain* c_id = *consumer_it;
      if (!inliningGraph().disjointValSets().strictAreMapped(p_id, c_id) ||
          !isAllowedID(c_id, consumer, best_effort, true, false, true)) {
        return producer_pos;
      }

      ++consumer_it;
    }

    return producer->nDims();
  }
}

size_t MaxPosCalculator::getMaxPosAll(
    TensorView* tv,
    bool best_effort,
    bool check_siblings) {
  auto max_pos = getMaxPosSelf(tv, best_effort, false, false, false);
  for (auto consumer_tv : ir_utils::consumerTvsOf(tv)) {
    max_pos = std::min<size_t>(
        max_pos, getMaxProducerPosFromConsumer(tv, consumer_tv, best_effort));
  }
  if (check_siblings) {
    for (auto sibling_tv : ir_utils::siblingTvsOf(tv)) {
      max_pos = std::min<size_t>(
          max_pos, getMaxPosAll(sibling_tv, best_effort, false));
    }
  }
  return max_pos;
}

// Try to find the aligned position on consumer's domain corresponding to a
//  position of producer domain. No checking on actual
//  producer-consumer relationship.
int64_t MaxPosCalculator::getConsumerPosAlignedToProducerCA(
    TensorView* consumer,
    TensorView* producer,
    int64_t producer_pos) {
  // Locate consumer's position that aligns with
  //  the producer's position. We need broadcast axes forwarded so we
  //  need to replay PasC as CasP will not forward braodcast dims. For example
  //  if we have:
  // T2[ iS22{( 3 * 1 )} ] ca_pos( 1 ) = broadcast( T1[ iS1{3} ] ca_pos( 1 )
  // produce_pos( 1) ) CasP will have the mapping iS1{3} -> iS2{3} and PasC will
  // have the mapping iS22{( 3 * 1 )} <- iS1{3} We need the latter. Refer to
  // NVFuserTest.FusionComplexBCast1_CUDA

  int64_t consumer_pos = consumer->nDims();

  const bool may_need_forwarding =
      ir_utils::isLoopDomainFullyDerivedFromLogicalDomain(producer) &&
      ir_utils::isLoopDomainFullyDerivedFromLogicalDomain(consumer);

  if (may_need_forwarding) {
    auto disjoint_sets = BestEffortReplay::replayPasC(
                             producer,
                             consumer,
                             -1,
                             PairwiseLogicalDomainMap(producer, consumer))
                             .getIterDomainEquivalence();

    // Find the innermost position of consumer that has
    //  been mapped within the producer ca axis.

    while (consumer_pos > 0) {
      auto consumer_id = consumer->axis(consumer_pos - 1);
      const auto& p_dom = producer->getLoopDomain();
      if (std::any_of(
              p_dom.begin(),
              p_dom.begin() + producer_pos,
              [&consumer_id, &disjoint_sets](IterDomain* p_id) {
                return disjoint_sets.permissiveAreMapped(consumer_id, p_id);
              })) {
        break;
      }
      consumer_pos--;
    }
  } else {
    while (consumer_pos > 0) {
      auto consumer_id = consumer->axis(consumer_pos - 1);
      const auto& p_dom = producer->getLoopDomain();
      if (std::any_of(
              p_dom.begin(),
              p_dom.begin() + producer_pos,
              [&](IterDomain* p_id) {
                return inliningGraph().disjointValSets().strictAreMapped(
                    consumer_id, p_id);
              })) {
        break;
      }
      consumer_pos--;
    }
  }

  return consumer_pos;
}

void inlineMost(const std::unordered_set<IterDomain*>& uninlinable_ids) {
  inlineMost(FusionGuard::getCurFusion()->allTvs(), uninlinable_ids);
}

void inlineMost(
    const std::vector<TensorView*>& tvs,
    const std::unordered_set<IterDomain*>& uninlinable_ids) {
  if (tvs.empty()) {
    return;
  }
  MaxPosCalculator calc(uninlinable_ids);
  for (auto tv : tvs) {
    tv->inlineAt(-1, true, &calc);
  }
}

void inlineMost(
    const std::unordered_set<TensorView*>& tvs,
    const std::unordered_set<IterDomain*>& uninlinable_ids) {
  if (tvs.empty()) {
    return;
  }
  MaxPosCalculator calc(uninlinable_ids);
  for (auto tv : tvs) {
    tv->inlineAt(-1, true, &calc);
  }
}

namespace {

// Find the positions of `selected` tensors that is mapped to the given position
// in the reference tensor.
class FindMappedPositions : public MaxInfoSpanningTree::Propagator {
  std::unordered_map<TensorView*, int64_t>& output_;

 public:
  FindMappedPositions(
      std::unordered_map<TensorView*, int64_t>& output,
      TensorView* reference,
      int64_t reference_pos);

  ~FindMappedPositions() override = default;

  void propagateC2P(TensorView* from, TensorView* to) override;
  void propagateP2C(TensorView* from, TensorView* to) override;
  void propagateSibling(TensorView* from, TensorView* to) override;
};

FindMappedPositions::FindMappedPositions(
    std::unordered_map<TensorView*, int64_t>& output,
    TensorView* reference,
    int64_t reference_pos)
    : output_(output) {
  if (reference_pos < 0) {
    reference_pos += int64_t(reference->nDims()) + 1;
  }
  NVF_CHECK(
      reference_pos >= 0 && reference_pos <= int64_t(reference->nDims()),
      "Invalid axis received ",
      reference_pos,
      " but should be > -",
      reference->nDims(),
      " and <= ",
      reference->nDims(),
      ".");
  output_[reference] = reference_pos;
}

void FindMappedPositions::propagateC2P(TensorView* from, TensorView* to) {
  int64_t from_pos = output_.at(from);
  auto to_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, from_pos);
  // If there is no matching position found, we compute the highest matched
  // position as the closest approximation
  while (to_pos < 0) {
    from_pos--;
    to_pos =
        TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, from_pos);
  }
  output_[to] = to_pos;
}

void FindMappedPositions::propagateP2C(TensorView* from, TensorView* to) {
  int64_t from_pos = output_.at(from);
  auto to_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, from_pos);
  // If there is no matching position found, we compute the highest matched
  // position as the closest approximation
  while (to_pos < 0) {
    from_pos--;
    to_pos =
        TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, from_pos);
  }
  output_[to] = to_pos;
}

void FindMappedPositions::propagateSibling(TensorView* from, TensorView* to) {
  auto from_pos = output_.at(from);
  NVF_CHECK(
      TransformReplay::fullSelfMatching(to, from),
      "Transformations in siblings ",
      from,
      " and ",
      to,
      " does not match with each other.");
  output_[to] = from_pos;
}

std::unordered_map<TensorView*, int64_t> getPositionsMappedTo(
    TensorView* reference_tv,
    int64_t reference_pos) {
  std::unordered_map<TensorView*, int64_t> mapped_positions;
  // Spanning tree traversal should not propagate across Resize ops
  // since inlining should not be done across resized IDs. See
  // ResizeTest.TraversalForInliningPosition for a concrete example.
  MaxLogicalDomainInfoSpanningTree tree(
      reference_tv,
      reference_pos,
      /*selector=*/nullptr,
      /*propagate_through_resize=*/false);
  FindMappedPositions propagator(mapped_positions, reference_tv, reference_pos);
  tree.traverse(&propagator);
  return mapped_positions;
}

} // namespace

void inlineAllAt(
    TensorView* reference_tv,
    int64_t reference_pos,
    bool best_effort,
    const std::unordered_set<IterDomain*>& uninlinable_ids) {
  auto mapped_positions = getPositionsMappedTo(reference_tv, reference_pos);
  MaxPosCalculator calc(uninlinable_ids);
  for (auto pair : mapped_positions) {
    pair.first->inlineAt((int64_t)pair.second, best_effort, &calc);
  }
}

void inlineSelectedAt(
    const std::unordered_set<TensorView*>& selected,
    TensorView* reference_tv,
    int64_t reference_pos,
    bool best_effort,
    const std::unordered_set<IterDomain*>& uninlinable_ids) {
  auto mapped_positions = getPositionsMappedTo(reference_tv, reference_pos);
  MaxPosCalculator calc(uninlinable_ids);
  for (auto pair : mapped_positions) {
    if (selected.count(pair.first) > 0) {
      pair.first->inlineAt((int64_t)pair.second, best_effort, &calc);
    }
  }
}

} // namespace nvfuser

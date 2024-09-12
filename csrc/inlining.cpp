// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <id_model/utils.h>
#include <inlining.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <transform_iter.h>

#include <utility>

#include <fstream>

namespace nvfuser {

MaxPosCalculator::MaxPosCalculator(
    std::unordered_set<IterDomain*> uninlinable_ids,
    bool compute_at_only)
    : uninlinable_ids_(std::move(uninlinable_ids)) {
  buildUnmappableDims(compute_at_only);
  //for (auto id : unmappable_dims_) {
  //    std::cerr << "Unmappable: " << id->toString() << "\n";
  //}
  if (isIdModelOptionEnabled(IdModelEnableOption::Inlining)) {
    id_model_ = std::make_unique<IdModel>(
        FusionGuard::getCurFusion(), /*build_graphs=*/false);
    id_model_->buildExactGraph();
    if (getenv("INLINING_GRAPH")) {
      std::ofstream ofs("exact_graph.dot", std::ofstream::trunc);
      auto dot_string =
          id_model_->idGraph(IdMappingMode::EXACT).toGraphvizDotGraph();
      ofs << dot_string;
      ofs.close();
    }
  }
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
    bool is_unmappable = false;
    auto logical_dom = tv->getLogicalDomain();
    std::unordered_set<Val*> logical_dom_set(
        logical_dom.begin(), logical_dom.end());
    auto all_vals =
        IRBFS::getValsBetween({logical_dom.begin(), logical_dom.end()}, {id});
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
    bool best_effort) const {
  //std::cerr << "getMaxProducerPosFromConsumer: " << producer->toString() << ", "
  //<< consumer->toString() << "\n";

  if (false && lower_utils::hasRootToLoopLinearTransformations(producer) &&
      lower_utils::hasRootToLoopLinearTransformations(consumer)) {
    auto pairwise_logical_map = PairwiseLogicalDomainMap(producer, consumer);
    auto replay_CasP = BestEffortReplay::replayCasP(
        consumer, producer, -1, pairwise_logical_map);
    auto p2c_replay_map = replay_CasP.getReplay();

    for (const auto producer_pos : c10::irange(producer->nDims())) {
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
    NVF_ERROR(
        id_model_.get() != nullptr,
        "Nonconventional loop domains require IdModle: ", producer->toString(),
        ", ", consumer->toString());
    const auto& exact_graph = id_model_->idGraph(IdMappingMode::EXACT);
    for (const auto producer_pos : c10::irange(producer->nDims())) {
      auto p_id = producer->getLoopDomain().at(producer_pos);
      auto c_id_it = std::find_if(
          consumer->getLoopDomain().begin(),
          consumer->getLoopDomain().end(),
          [&](IterDomain* c_id) -> bool {
            return exact_graph.disjointValSets().strictAreMapped(p_id, c_id);
          });
      if (c_id_it == consumer->getLoopDomain().end()) {
        //std::cerr << "No matching consumer id found: " << p_id->toString()
        //<< "\n";
        return producer_pos;
      }

      IterDomain* c_id = *c_id_it;
      if (!isAllowedID(c_id, consumer, best_effort, true, false, true)) {
        return producer_pos;
      }
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
  MaxLogicalDomainInfoSpanningTree tree(reference_tv, reference_pos);
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

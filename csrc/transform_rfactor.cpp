// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <transform_rfactor.h>

#include <fusion.h>
#include <instrumentation.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <ops/arith.h>

namespace nvfuser {

namespace {

// Empty set for replayDomain calls that don't need to ignore any IDs
static const std::unordered_set<IterDomain*> kEmptyIgnoreIds{};

// This class replays the root domains of the producer of an logical domain.
// Axes must be replayed to mark rfactor iter domains as being reductions in the
// producer, but converting the other reductions in the producer as iter
// domains. Those (previously reductions in the producer) iter domains are then
// converted to reduction domains in the consumer. This breaks up the reduction
// into two stages, but maintains the correct values are reduced across those
// stages.
//
// The logical domain of the producer must match the consumers root domain to
// maintain producer-consumer mappings. The following uses the original domain
// being rfactored and marked iter domains as "static_logical_ids". These static
// IDs cannot be changed in the producer as it would invalidate the rfactor, no
// longer matching the consumer.
//
// To find the logical domain in the producer which will be used as the root
// domain in the consumer, we start at the roots of producer, and replay forward
// the root iter domains if that iter domain is marked as a "static_logical_id".
// To do this we maintain the ordering of the iter domains. For example:
//
//       I1
//       /\           //
//     I2  \          //
//     /\  I3
//    / I4  /
//   /    \/
//  I5    I6
//
// If rfactor_axes = {I6}, then "static_logical_id" IDs will be {I6, I4, I3, I2,
// I1}. Then, as we perform the replay the logical domain will be updated as:
// [I1] -> [I2, I3] -> [I5, I4, I3] -> [I5, I6]
//
// ReplayTransformations typically updates the loop ids, but we'll simply use
// the mapping from the original tensor domain so we won't bother updating them
// in this replay.
class ReplayRFactor : public ReplayTransformations {
 private:
  std::vector<IterDomain*>::iterator getPosInDomain(
      std::vector<IterDomain*>& domain,
      IterDomain* id) {
    auto pos = std::find(domain.begin(), domain.end(), id);
    NVF_ERROR(
        pos != domain.end(),
        "Could not find iter domain: ",
        id->toString(),
        " in the domain, domain=",
        toDelimitedString(domain));
    return pos;
  }

  // Perform the update of the given domain by replacing "replace0" with
  // "with0" and if not nullptr "with1", also removes "replace1" if not nullptr.
  void updateDomain(
      std::vector<IterDomain*>& domain,
      IterDomain* replace0,
      IterDomain* replace1,
      IterDomain* with0,
      IterDomain* with1) {
    NVF_ERROR(
        with0 != nullptr,
        "The first provided IterDomain should be a real pointer,",
        " the second iter domain provided can be a nullptr.");
    auto pos = getPosInDomain(domain, replace0);
    domain.insert(pos, with0);
    if (with1 != nullptr) {
      pos = getPosInDomain(domain, replace0);
      domain.insert(pos, with1);
    }
    pos = getPosInDomain(domain, replace0);
    domain.erase(pos);
    if (replace1 != nullptr) {
      pos = getPosInDomain(domain, replace1);
      domain.erase(pos);
    }
  }

  void updateRFactorDomain(
      IterDomain* replace0,
      IterDomain* replace1,
      IterDomain* with0,
      IterDomain* with1) {
    updateDomain(logical_domain_, replace0, replace1, with0, with1);
    if (!allocation_domain_.empty()) {
      updateDomain(allocation_domain_, replace0, replace1, with0, with1);
    }
  }

  // Took a good bit of this from ReplayTransformations::handle(Split...)
  void handle(Split* s) override {
    // Grab input to the split operation
    auto id_in = s->in();
    // Grab our mapping of that ID to the one we're replaying
    auto it = id_map_.find(id_in);
    // Make sure it exists in the map
    NVF_ERROR(
        it != id_map_.end(),
        "Transform traversal failed, dependencies not met.");
    // Grab the ID we're going to replay on
    auto mapped = (*it).second;
    // This ID should be a loop ID (meaning it has no uses we generated)
    NVF_ERROR(
        loop_ids_.find(mapped) != loop_ids_.end(),
        "Transform traversal failed, modified a node but it was not a loop "
        "node.");

    // Check if we need to mark the outputs as an logical domain meaning this
    // transformation must be present in replays otherwise it breaks the compute
    // definition of the fusion. Iter domains are actually not static, its the
    // transformation that's static or not, so if one output is marked as a
    // static id, then both must be.
    bool static_logical_outputs = static_logical_ids_.count(s->outer()) ||
        static_logical_ids_.count(s->inner());

    // A split of a reduction ID may output a non-reduction ID when the split
    // is involved in a prior rfactor transformation. In that case, we need to
    // preserve the non-reduction iteration type, which is not automatically
    // done by IterDomain::split.  This happens when a TV is rfactored multiple
    // times, e.g., test_communication.py::test_allreduce and
    // test_schedule_ops.py::test_rfactor_twice
    std::optional<IterType> outer_iter_type;
    std::optional<IterType> inner_iter_type;
    if (s->in()->isReduction()) {
      if (!rfactor_dep_ids_.count(s->outer())) {
        outer_iter_type = IterType::Iteration;
      }
      if (!rfactor_dep_ids_.count(s->inner())) {
        inner_iter_type = IterType::Iteration;
      }
    }

    auto [ido, idi] = IterDomain::split(
        mapped,
        s->factor(),
        s->innerSplit(),
        static_logical_outputs,
        outer_iter_type,
        inner_iter_type);

    // Remove mapped id from loop IDs
    loop_ids_.erase(mapped);
    // Add outputs to loop IDs
    loop_ids_[ido] = newCounter();
    loop_ids_[idi] = newCounter();

    // Update our ID map to include these outputs
    id_map_[s->outer()] = ido;
    id_map_[s->inner()] = idi;

    if (static_logical_ids_.count(s->in())) {
      updateRFactorDomain(s->in(), nullptr, s->outer(), s->inner());
    }
  }

  void handle(Merge* m) override {
    auto id_outer = m->outer();
    auto id_inner = m->inner();
    auto it_outer = id_map_.find(id_outer);
    auto it_inner = id_map_.find(id_inner);
    NVF_ERROR(
        it_outer != id_map_.end() && it_inner != id_map_.end(),
        "Transform traversal failed, dependencies not met.");

    auto id_outer_mapped = (*it_outer).second;
    auto id_inner_mapped = (*it_inner).second;

    NVF_ERROR(
        loop_ids_.find(id_outer_mapped) != loop_ids_.end() &&
            loop_ids_.find(id_inner_mapped) != loop_ids_.end(),
        "Transform traversal failed, modified ",
        id_outer_mapped,
        " and ",
        id_inner_mapped,
        " however one or both are not loop nodes.");

    // Let IterDomain::merge determine the correct IterType, except
    // when the output is a reduction domain but not part of the
    // rfactored domains. If it isn't involved in the rfactor, it's no
    // longer a redunction domain
    std::optional<IterType> iter_type;
    if (m->out()->isReduction() && !rfactor_dep_ids_.count(m->out())) {
      iter_type = IterType::Iteration;
    }

    IterDomain* merged_id = IterDomain::merge(
        id_outer_mapped,
        id_inner_mapped,
        static_logical_ids_.count(m->out()),
        iter_type);

    // Remove inputs from the loop IDs
    loop_ids_.erase(id_outer_mapped);
    loop_ids_.erase(id_inner_mapped);

    // Add the output to the loop IDs
    loop_ids_[merged_id] = newCounter();

    id_map_[m->out()] = merged_id;

    // Similar to split replay above, check if output needs to be marked as
    // rfactor indicating this transofrmation is static.
    if (static_logical_ids_.count(m->inner()) ||
        static_logical_ids_.count(m->outer())) {
      NVF_ERROR(
          static_logical_ids_.count(m->inner()) ==
              static_logical_ids_.count(m->outer()),
          "If one input to a merge is a static logical id, the other must be "
          "as well.");
      updateRFactorDomain(m->outer(), m->inner(), m->out(), nullptr);
    }
  }

  void handle(Resize* resize) override {
    NVF_THROW("Unexpected expression: ", resize->toString());
  }

  void handle(Swizzle* swizzle) override {
    NVF_THROW("Unexpected expression: ", swizzle->toString());
  }

  void handle(Swizzle2D* swizzle) override {
    NVF_THROW("Unexpected expression: ", swizzle->toString());
  }

  // The IterDomains in the original_domain that are being factored into the
  // first stage of the two stage reduction (the producer).
  std::unordered_set<IterDomain*> rfactor_axes_;
  // All iter domains between the logical and the loop that the
  // rfactor_axes_ depend on
  std::unordered_set<IterDomain*> rfactor_dep_ids_;
  // Iter domains whose history cannot be changed as it would break rfactor
  // dependencies.
  std::unordered_set<IterDomain*> static_logical_ids_;

 public:
  // The updated domain matching the producer's logical domain. This rfactor
  // domain is relative to the iter domains in the origianl_domain and must be
  // updated to grab the mapped id's later. Similarly, the allocation domain is
  // the allocation domain of the original domain and updated similar to logical
  // domain. Empty if no allocation domain is present.
  std::vector<IterDomain*> logical_domain_;
  std::vector<IterDomain*> allocation_domain_;

  ReplayRFactor(
      // Original domain the rfactor is in reference to.
      TensorDomain* original_domain,
      // The root mapping from the original root domain, to the roots of the
      // domain to be replayed.
      std::unordered_map<IterDomain*, IterDomain*> id_map,
      // The rfactor axes in original_domain->loop() to be factored into the
      // two stage reduction.
      std::unordered_set<IterDomain*> rfactor_axes,
      // All the iter domains in original_domain that the rfactor axes are
      // dependant on.
      std::unordered_set<IterDomain*> static_logical_ids)
      : ReplayTransformations(original_domain->loop(), std::move(id_map)),
        rfactor_axes_(std::move(rfactor_axes)),
        static_logical_ids_(std::move(static_logical_ids)),
        logical_domain_(original_domain->logical()) {
    const auto all_dep_vals = DependencyCheck::getAllValsBetween(
        {original_domain->maybeRoot().begin(),
         original_domain->maybeRoot().end()},
        {rfactor_axes_.begin(), rfactor_axes_.end()});

    auto all_dep_ids = ir_utils::filterByType<IterDomain>(all_dep_vals);
    rfactor_dep_ids_.insert(all_dep_ids.begin(), all_dep_ids.end());

    if (original_domain->hasAllocation()) {
      allocation_domain_ = original_domain->allocation();
    }

    setErrorOnFailure(false);
  }
};

// Use the replay_to_target_map to replay the replay_domain to the
// target_domain. ignore_rfactor_ids is true for consumers where the replay will
// not have these ids since they are already reduced. propagate_padding = true
// for loop domain. propagate_parallelization = true for consumers. Device and
// stream parallel types should always be preserved in replay.
void replayDomain(
    const std::vector<IterDomain*>& replay_domain,
    std::vector<IterDomain*>& target_domain,
    std::unordered_map<IterDomain*, IterDomain*>& replay_to_target_map,
    const std::unordered_set<IterDomain*>& ignore_ids = kEmptyIgnoreIds,
    bool propagate_padding = false,
    bool propagate_parallelization = false) {
  for (const auto& replay_id : replay_domain) {
    auto target_id_it = replay_to_target_map.find(replay_id);

    if (ignore_ids.count(replay_id)) {
      continue;
    }

    NVF_ERROR(
        target_id_it != replay_to_target_map.end(),
        "Error during rfactor replay, missing an axis.",
        replay_id->toString());
    IterDomain* target_id = target_id_it->second;

    if (propagate_padding) {
      if (replay_id->hasPaddingToMultipleOfWarp()) {
        target_id->padToMultipleOfWarp(replay_id->getMaybeSizeAfterPadding());
      }
    }

    // Device and stream parallel types should always be preserved in replay.
    // Other parallel types are only relevant to replay of the loop domain.
    if (propagate_parallelization || replay_id->isDeviceDim() ||
        replay_id->isStream()) {
      target_id->parallelize(replay_id->getParallelType());
    }
    target_domain.push_back(target_id);
  }
}

} // namespace

std::pair<TensorDomain*, TensorDomain*> TransformRFactor::runReplay(
    TensorDomain* original_td,
    std::vector<int64_t> axes) {
  FUSER_PERF_SCOPE("TransformRFactor::runReplay");

  NVF_CHECK(!axes.empty(), "No axes provided to rfactor replay.");

  int64_t ndims = original_td->nDims();

  // Adjust and check provided axes
  std::transform(axes.begin(), axes.end(), axes.begin(), [ndims](int64_t i) {
    NVF_CHECK(
        i >= -ndims && i < ndims,
        "Rfactor replay received an axis outside the number of dims in the "
        "tensor, acceptable inclusive range is ",
        -ndims,
        " to ",
        ndims - 1);
    return i < 0 ? i + ndims : i;
  });

  // remove duplicates, and put into a set for searching
  std::unordered_set<int64_t> axes_set(axes.begin(), axes.end());

  NVF_ERROR(
      std::all_of(
          axes_set.begin(),
          axes_set.end(),
          [original_td](int64_t i) {
            return original_td->axis(i)->isReduction();
          }),
      "Cannot rfactor axes that are not reduction axes.");

  // RFactor requires at least one reduction axis to be marked as factored out,
  // and at least one reduction axis that won't. Otherwise it's just a pointwise
  // cacheing operation.
  bool found_non_rfactor_reduction = false;

  // Make a set of final axes that are marked to be rfactored
  std::unordered_set<IterDomain*> rfactor_axes(axes_set.size());
  {
    int i = 0;
    for (auto id : original_td->loop()) {
      if (axes_set.find(i++) != axes_set.end()) {
        rfactor_axes.emplace(id);
      } else if (id->isReduction()) {
        found_non_rfactor_reduction = true;
      }
    }
  }

  NVF_CHECK(
      found_non_rfactor_reduction,
      "Must have at least one reduction axis not marked as rfactor.");

  // Get root IterDomains of the logical domains, these will be the ones we will
  // replay marked as rfactor axes, those marked in the axes set will be
  // reduction=false
  auto rfactor_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(rfactor_axes.begin(), rfactor_axes.end()));
  auto rfactor_root_ids = ir_utils::filterByType<IterDomain>(rfactor_root_vals);

  // Put in a set to make searching easy
  std::unordered_set<IterDomain*> rfactor_root_axes(
      rfactor_root_ids.begin(), rfactor_root_ids.end());

  NVF_ERROR(
      std::none_of(
          rfactor_root_ids.begin(),
          rfactor_root_ids.end(),
          [](IterDomain* id) { return id->maybePartial(); }),
      "rFactor of partial domains not allowed, but at least one found.");

  // For hopper matmuls, the mma_result logical domain is reordered as [M, N, K]
  // using commitLeafToLogical. Thus, the original logical domain is moved to
  // the root domain. In this case, map from producer to consumer's root domain.
  auto original_td_root = original_td->maybeRoot();

  // Generate a new TensorDomain and set up map from one root to this one.
  std::vector<IterDomain*> new_producer_root(original_td_root.size(), nullptr);
  std::unordered_map<IterDomain*, IterDomain*> original_to_producer_root_map;

  {
    for (auto i : arange(original_td_root.size())) {
      auto id = original_td_root[i];
      // If this is an rfactor root, it will be a reduction in this stage
      if (rfactor_root_axes.find(id) != rfactor_root_axes.end()) {
        new_producer_root[i] = IterDomainBuilder(id->start(), id->extent())
                                   .stop_offset(id->stopOffset())
                                   .iter_type(IterType::Reduction)
                                   .is_rfactor_domain(true)
                                   .build();
        // If this is not an rfactor root, but a reduction root, it should be
        // turned into an iteration domain
      } else if (id->isReduction()) {
        new_producer_root[i] = IterDomainBuilder(id->start(), id->extent())
                                   .stop_offset(id->stopOffset())
                                   .build();
      } else {
        new_producer_root[i] = id->cloneWithoutRFactor();
      }
      original_to_producer_root_map[id] = new_producer_root[i];
    }
  }

  // Axes in the original_td that are in the history of the rfactored domains.
  // These will mark which iter domains must be preserved as static
  // transformations to preserve compute semantics.
  auto all_deps_of_logical = DependencyCheck::getAllValsBetween(
      {original_td->logical().begin(), original_td->logical().end()},
      {rfactor_axes.begin(), rfactor_axes.end()});

  auto all_id_deps_of_logical =
      ir_utils::filterByType<IterDomain>(all_deps_of_logical);

  std::unordered_set<IterDomain*> static_logical_ids(
      {all_id_deps_of_logical.begin(), all_id_deps_of_logical.end()});

  // Replay producer dimensions.
  ReplayRFactor replay_rfactor(
      original_td,
      original_to_producer_root_map,
      rfactor_axes,
      static_logical_ids);

  std::unordered_map<IterDomain*, IterDomain*> original_to_producer_id_map =
      replay_rfactor.getReplay();

  std::vector<IterDomain*> new_producer_domain;
  new_producer_domain.reserve(original_td->nDims());
  replayDomain(
      original_td->loop(),
      new_producer_domain,
      original_to_producer_id_map,
      /*ignore_ids=*/kEmptyIgnoreIds,
      /*propagate_padding=*/true,
      /*propagate_parallelization=*/true);

  // Specify the logical domain of the producer which will match the consumer
  // root domain.
  std::vector<IterDomain*> new_producer_logical_domain;
  new_producer_logical_domain.reserve(replay_rfactor.logical_domain_.size());
  replayDomain(
      replay_rfactor.logical_domain_,
      new_producer_logical_domain,
      original_to_producer_id_map,
      /*ignore_ids=*/kEmptyIgnoreIds,
      /*propagate_padding=*/false,
      /*propagate_parallelization=*/false);

  auto* producer_domain = IrBuilder::createInContainer<TensorDomain>(
      original_td->container(),
      new_producer_root,
      new_producer_logical_domain,
      new_producer_domain,
      TensorDomain::getContiguityFilledWith(new_producer_logical_domain, true));

  if (original_td->hasAllocation()) {
    std::vector<IterDomain*> new_producer_allocation_domain;
    new_producer_allocation_domain.reserve(
        replay_rfactor.allocation_domain_.size());
    replayDomain(
        replay_rfactor.allocation_domain_,
        new_producer_allocation_domain,
        original_to_producer_id_map,
        /*ignore_ids=*/kEmptyIgnoreIds,
        /*propagate_padding=*/false,
        /*propagate_parallelization=*/false);
    producer_domain->setAllocationDomain(
        new_producer_allocation_domain,
        TensorDomain::getContiguityFilledWith(
            new_producer_allocation_domain, true));
  }

  // Producer has been finished, now work on consumer.

  // For convenience flip the original to producer map
  std::unordered_map<IterDomain*, IterDomain*> producer_to_original_map;
  for (auto entry : original_to_producer_id_map) {
    producer_to_original_map[entry.second] = entry.first;
  }

  std::vector<IterDomain*> new_consumer_root_domain;
  new_consumer_root_domain.reserve(new_producer_logical_domain.size());
  std::unordered_map<IterDomain*, IterDomain*> original_to_consumer_root_map;
  for (auto p_root_id : new_producer_logical_domain) {
    if (p_root_id->isReduction()) {
      continue;
    }
    auto p2o_it = producer_to_original_map.find(p_root_id);
    NVF_ERROR(
        p2o_it != producer_to_original_map.end(),
        "Missing mapping from original tensor domain to producer tensor "
        "domain.");
    auto original_id = p2o_it->second;
    auto new_consumer_root =
        IterDomainBuilder(original_id->start(), original_id->extent())
            .stop_offset(original_id->stopOffset())
            .iter_type(original_id->getIterType())
            .build();
    new_consumer_root_domain.push_back(new_consumer_root);
    original_to_consumer_root_map[original_id] = new_consumer_root;
  }

  ReplayTransformations consumer_replay(
      original_td->loop(), original_to_consumer_root_map);
  consumer_replay.setErrorOnFailure(false).setReplayResize(true);

  auto original_to_consumer_map = consumer_replay.getReplay();

  std::vector<IterDomain*> new_consumer_domain;
  new_consumer_domain.reserve(original_td->nDims());
  replayDomain(
      original_td->loop(),
      new_consumer_domain,
      original_to_consumer_map,
      /*ignore_ids=*/rfactor_axes,
      /*propagate_padding=*/true,
      /*propagate_parallelization=*/true);

  auto consumer_domain = IrBuilder::createInContainer<TensorDomain>(
      original_td->container(),
      new_consumer_root_domain,
      new_consumer_domain,
      TensorDomain::getContiguityFilledWith(new_consumer_root_domain, true));

  if (original_td->hasAllocation()) {
    std::vector<IterDomain*> new_consumer_allocation_domain;
    new_consumer_allocation_domain.reserve(
        replay_rfactor.allocation_domain_.size());
    replayDomain(
        replay_rfactor.allocation_domain_,
        new_consumer_allocation_domain,
        original_to_consumer_map,
        /*ignore_ids=*/rfactor_axes,
        /*propagate_padding=*/false,
        /*propagate_parallelization=*/false);
    consumer_domain->setAllocationDomain(
        new_consumer_allocation_domain,
        TensorDomain::getContiguityFilledWith(
            new_consumer_allocation_domain, true));
  }

  return std::make_pair(producer_domain, consumer_domain);
}

} // namespace nvfuser

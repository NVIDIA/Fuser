// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

#include <disjoint_set.h>
#include <ir/all_nodes.h>
#include <iter_visitor.h>
#include <unordered_map>
#include <vector>

namespace nvfuser {

class LogicalDomainMap;

namespace {

// Enable pair<IterDomain*, size_t> in a set, size_t must be unique in set
struct id_int_lt {
  bool operator()(
      const std::pair<IterDomain*, size_t>& first,
      const std::pair<IterDomain*, size_t>& second) const {
    return first.second < second.second;
  }
};

} // namespace

// Uses the history of _target_domain, and replays that history using the
// provided map.
//
// target_domain contains the history we want replayed.
//
// id_map maps IterDomains in that history to the IterDomains we want it
// replayed on.
//
// error_on_failure = true will cause the replay to error if we can't replay any
// operation in target_domain's history due to missing IDs in the id_map.
//
// If error_on_failure = false, replay will replay everything it can, and ignore
// operations it can't.
class ReplayTransformations : public IterVisitor {
 public:
  ReplayTransformations(
      const std::vector<IterDomain*>& target_domain,
      std::unordered_map<IterDomain*, IterDomain*> id_map);

  ReplayTransformations& setErrorOnFailure(bool error_on_failure) {
    error_on_failure_ = error_on_failure;
    return *this;
  }

  ReplayTransformations& setReplaySwizzle(bool replay_swizzle) {
    replay_swizzle_ = replay_swizzle;
    return *this;
  }

  ReplayTransformations& setReplayResize(bool replay_resize) {
    replay_resize_ = replay_resize;
    return *this;
  }

  ReplayTransformations& setReplayRFactor(bool replay_rfactor) {
    replay_rfactor_ = replay_rfactor;
    return *this;
  }

  // Replays outputs that were generated from ids.first on ids.second
  void runReplay();

  // Returns map from provided target domain to their corresponding IDs
  const std::unordered_map<IterDomain*, IterDomain*>& getReplay() {
    if (!ran_replay_) {
      runReplay();
    }
    return id_map_;
  }

  // Returns loop_ids_ the size_t marks the order in which they were put into
  // the map, this is part of the structure because it's used to generate the
  // order from 'getLeafIDs'
  const std::unordered_map<IterDomain*, size_t>& getUnorderedLeafIDs() {
    if (!ran_replay_) {
      runReplay();
    }
    return loop_ids_;
  }

  // Returns all terminating IDs that resulted from the replay. Leaf IDs are run
  // to run deterministic, but otherwise in no specific order.
  const std::vector<IterDomain*>& getLeafIDs() {
    if (!ran_replay_) {
      runReplay();
    }
    return loop_vec_;
  }

 protected:
  using IterVisitor::handle;

  // Transform dispatch
  void dispatch(Expr* e) override;

  // We're going to replay this split operation on the corresponding ID
  void handle(Split* s) override;

  // We're going to replay this merge operation on the corresponding IDs
  void handle(Merge* m) override;

  // We're going to replay this swizzle operation on the corresponding IDs
  //  if replaying swizzle is enabled.
  void handle(Swizzle* m) override;
  void handle(Swizzle2D* m) override;

  void handle(Resize* resize) override;

  size_t newCounter() {
    return counter_++;
  }

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const std::vector<IterDomain*>& target_domain_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<IterDomain*, IterDomain*> id_map_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<IterDomain*, size_t> loop_ids_;

  bool error_on_failure_ = true;

 private:
  // Indicates if we want to replay swizzle ops on the replayed
  //  tensor.
  // The swizzle op will be replayed if true,
  // The swizzle inputs will be directly forwarded, and therefore skipping
  //  the swizzle op if false.
  // Currently this options should always be off but
  //  later we may have cases in scheduling large fusions where
  //  this functionality could be useful.
  bool replay_swizzle_ = false;

  // Indicates if we want to replay resize ops on the replayed
  // tensor.
  bool replay_resize_ = false;

  // Whether to copy the `rf` flag from ops producing `target_domain`.
  bool replay_rfactor_ = false;

  size_t counter_ = 0;

  std::vector<IterDomain*> loop_vec_;

  bool ran_replay_ = false; // Mark if replay has been run
};

// Maps that track information relevant to best effort replay about newly added
// or squeezed broadcast axes
//
// For example if we have consumer: T0[i0, b1, b2, i3] and producer:
// T1[i0, i3]
//
// If consumer transformations are:
// -> T[i0, b1o, b1i, b2o, b2i, i3]
// -> T[i0*b1i, b1o, b2o, b2i, i3]
// -> T[i0*b1i*b2o, b1o, b2i, i3]
// -> T[i0*b1i*b2o*i3, b1o, b2i]
//
// forwarding_map would forward i0->i0*b1i and i0*b1i->i0*b1i*b2o
// compliment_map would have the entry i0->b1i and i0*b1i->b2o
//
// The first is to fast forward transformations in consumer involving broadcast
// axes not in producer. The compliment map is to use later to compute what loop
// nodes we may have after the forwarding process is finished. Leaf nodes are
// only important for replayCasP, so look there to see how this is done. Forward
// map is used for replayCasP and replayPasC.
//
// The producer forwarding map is filled when producer broadcast
// domains are squeezed.
class ForwardingInfo {
 public:
  // Map IterDomain* axes that can safely be forwarded to their output.
  std::unordered_map<IterDomain*, IterDomain*> producer_forwarding_map;
  std::unordered_map<IterDomain*, IterDomain*> consumer_forwarding_map;

  // Given a forward id map id_input -> id_forwarded
  // Track the other inputs in the expr that id_input is an input to. These will
  // be used to adjust the replay's loop tracking. Don't need to track one to
  // many as currently transformations on IterDomains can only have maximum 2
  // inputs, but maybe in the future we'll have more.
  std::unordered_map<IterDomain*, std::vector<IterDomain*>>
      producer_compliment_map;
  std::unordered_map<IterDomain*, std::vector<IterDomain*>>
      consumer_compliment_map;

  ForwardingInfo(const TensorView* producer, const TensorView* consumer);

  ForwardingInfo() = delete;
};

/*
 * Short Description:
 *
 * Given an Expr in target_domain, check if its inputs are in replay_map. If so,
 * check if the mapped domain in replay_map are recorded to be transformed by an
 * "equivelent" operation in replay_domain's history. If so, forward the
 * operation and update replay_map to map the outputs of the expressions across
 * target_domain and reference_domain.
 *
 * Long Description:
 *
 * replay_map maps root IDs in the history of target_domain to root IDs in the
 * history replay_domain. PasC and CasP is just a convenient mechanism to have
 * BestEffortReplay make this base root mapping.
 *
 * Note: See ForwardingInfo in transform_iter.cpp for more information on
 * forwarding.
 *
 * Motivation:
 *
 * Consider the following program:
 *
 * T1[I0, R1] = T0[I0, I1]
 * T2[I0] = T1[I0, R1i]
 *
 * T1->split(1, factor)
 * T1->rFactor(2)
 *
 * T4[I0, R1orf, I1irf] = T0[I0, I1]
 * T1[I0, R1i] = T4[I0, R1orf, I1irf]
 * T2[I0] = T1[I0, R1i]
 *
 * There's an issue when we want to replay T4 to have transformations similar to
 * those on T0. Primarily T0's "rfactor" domain has a strict match requirement
 * on T4's root domain. If transformations on top of T0 don't match T4's
 * transformations (from T4's root domain to T4's logical domain), T4 cannot be
 * replayed like T0 on those domains as they would generate incorrect code in
 * the system today.
 *
 * Side note potentially for the future: In theory we could actually disconnect
 * T4's view from it's logical domain. This would allow logical domains to be
 * "reversible". The way this would have to be implemented is that there just
 * needs to be a path of transformations from a tensors loop domains, to its
 * root domains, and its logical domain. It shouldn't really matter if those
 * connections are forward or backward through transformations. The only thing
 * that really matters is they're connected. This is left for future work as it
 * could have significant impact on other parts of the system like how loops are
 * generated and expressions are sorted.
 *
 * T0 doesn't have this constraint if we want to replay T0 as T4, so this is
 * directional based on rfactor. Therefore to replay T0 transformations onto T4
 * we want to make sure those transformations are consistent with T4 (between
 * T4's root and logical domain). Best Effort Replay does not actually add any
 * transformations to the tensors provided. However, it will provide information
 * to determine producers's transformations are consistent with consumers
 * transformations (or the other way around). Best Effort Replay will return
 * discovered mappings between tensors that it detects to be matching based on
 * provided initial information (or just through p2c/c2p root domain mappings).
 *
 * Transformations have a concept of "permissiveness" used for broadcast and
 * squeeze. For example:
 *
 * T1[I0, B1] = T0[I0]
 * T2[I0, I1] = T1[I0, B1]
 *
 * We may want to replay T1 and T0 based on transformations on T2. These
 * transformations may involve B1. We could even have:
 *
 * T2->merge(0, 1)->split(0, 128)
 *
 * resulting in:
 *
 * T2[(I0*I1)/128, 128]
 *
 * T0 doesn't have I1 so it can't technicaly be transformed in an exactly
 * consistent way. However, it may still be desired to "inline" T0 into T1 and
 * in result T1 into T2. It may further be desired to bind BIDx and TIDx to the
 * two dimensions in the problem. This example doesn't "technically" result in
 * thread to thread communication, but since our scope in mind is a shared
 * global memory it results in duplicate reads. These duplicate reads are
 * automatically cached in our memory hierarchy. So in a way there is implicit
 * communication in that a memory location is read by multiple threads.
 *
 * This is where forwarding and permissiveness come into play. When we transform
 * T1 with the first merge, we will mark the result I0*B1 of T1 to be
 * "permissively" mapped to I0 of T0, so when we perform the split, we split
 * T0's I0 dimension to I0/128 and 128. This is to help us mark inlining and
 * paralellization across these dimensions so we can effectively reason about
 * the "not full" dimension in T0. This is where the concept of forward map in
 * BestEffortReplay comes in.
 *
 * Permissiveness can also be considered "symmetric" across broadcast and
 * squeeze as they are similar operations, however broadcast and squeeze do have
 * different implications since squeeze doesn't result in the implicit
 * communication described in the previous paragraph. However, as far as
 * forwarding is concerned they're symmetric. Indexing/parallelization has
 * significant logic dedicated to broadcast resolutions (unlike squeeze).
 *
 * This class provides a mechanism to annalyze all of the above concepts. It
 * can also run through transformations in target according to a manually
 * specified IterDomain to IterDomain replay_map. If equal transformations
 * already exist in replay_domain history, we will not redo those
 * transformations, but instead update replay_map to reflect forwarding the
 * existing transformations based on a notion of expresions being "equal" (input
 * IterDomains mapped and transformation expression parameters matching, or the
 * iter domain that doesn't match is in a forwarding map). The replay map is the
 * "best effort" part of BestEffortReplay, it doesn't actually perform new
 * transformations to enforce matching, it just detects existing matching
 * transforms. However, we still include rfactor validation within.
 */

class BestEffortReplay {
 private:
  std::unordered_map<IterDomain*, IterDomain*> target2replay_id_map_;
  std::unordered_map<IterDomain*, IterDomain*> replay_forward_id_map_;
  std::unordered_map<IterDomain*, IterDomain*> target_forward_id_map_;
  std::unordered_map<IterDomain*, size_t> loop_ids_;
  std::vector<IterDomain*> forwarded_ids_;
  std::unordered_map<IterDomain*, IterDomain*> skipped_resize_id_map_;

  // Need to track which id's have been forwarded. Later will need to make sure
  // loop nodes to produce "compliment" axes are properly tracked. i.e.
  // T[i0, b1, b2, i3]
  // -> T[i0, b1o, b1i, b2o, b2i, i3]
  // -> T[i0*b1i*b2o, b1o, b2i, i3]
  // -> T[i0*b1i*b2o*i3, b1o, b2i]
  // If we forwarded i0 -> i0*b1i*b2o*i3, we need to know that b1o and b2i
  // are loop nodes even though their split wasn't part of targets replay. These
  // are important IterDomains to track for transformation replays as otherwise
  // we could easily drop axes we need by accident

  // Counter to make sure best effort replay loop_ids can be grabbed
  // deterministicly, important to make sure replays are run to run
  // deterministic.
  size_t counter = 0;

  // Determine if current replay will ignore swizzle ops.
  // When not skipping swizzles, swizzle ops will have to be matched
  //  same way as split and merge to progress forward on the mapping.
  //
  // When skipping swizzles, mismatched swizzle ops will not stop matching
  //  further down the tensor domains but only the swizzle outputs will be on
  //  the target to replay map, since we only generate one-to-one maps in
  //  BestEffortReplay and the swizzle outputs is just picked as a convention
  //  for simpler and uniform mapping behavior. The swizzle op inputs will be
  //  added by the disjoint set passes when building the iterdomain graph.
  //
  // Example:
  //   Target:
  //     I0o, I0i   = split I0
  //     Ix0o, Ix0i = swizzle I0o, I0i
  //     I02        = merge Ix0o, Ix0i
  //   Replay:
  //     I1o, I1i = split I1
  //     I12      = merge I1o, I1i
  //
  //   BestEffortReplay **no** skip swizzle gives:
  //  {
  //   I0->I1,
  //   I0o->I1o,
  //   I0i->I1i,
  //  }
  //
  //   BestEffortReplay skip swizzle gives:
  //  {
  //    I0->I1,
  //    Ix0o->I1o,
  //    Ix0i->I1i,
  //    I02->I12
  //  }
  //
  // TODO: Reevaluate swizzle and transform replays. We have some concepts on
  // iter domain mapping we should formalize. It would be good to have these
  // options accessible while specified in a consistent manner.
  // https://github.com/ftxj/pytorch/pull/1#pullrequestreview-1210168522
  bool skip_replay_swizzle_ = true;
  bool skip_target_swizzle_ = true;

  bool error_on_failure_ = true;

  bool inReplayForwardMap(IterDomain* id) const {
    return replay_forward_id_map_.find(id) != replay_forward_id_map_.end();
  }

  bool inTargetForwardMap(IterDomain* id) const {
    return target_forward_id_map_.find(id) != target_forward_id_map_.end();
  }

  IterDomain* getReplayForwardedId(IterDomain* id) const {
    auto forwarded_id_it = replay_forward_id_map_.find(id);
    if (forwarded_id_it == replay_forward_id_map_.end()) {
      return id;
    } else {
      return getReplayForwardedId(forwarded_id_it->second);
    }
  }

  IterDomain* getTargetForwardedId(IterDomain* id) const {
    auto forwarded_id_it = target_forward_id_map_.find(id);
    if (forwarded_id_it == target_forward_id_map_.end()) {
      return id;
    } else {
      return getTargetForwardedId(forwarded_id_it->second);
    }
  }

  //! Adds complimenting IDs of forwarded IDs to the loop map
  void addComplimentLeafIDs(
      const std::unordered_map<IterDomain*, IterDomain*>& forwarding_map,
      const std::unordered_map<IterDomain*, std::vector<IterDomain*>>&
          compliment_map);

  // Skip swizzle step to make sure both target and
  //  replay swizzles are skipped while the mapping
  //  makes progress. This makes sure that, for example
  //  different tensors can still be inlined despite
  //  different local swizzle patterns.
  void skipSwizzles(
      const std::unordered_map<IterDomain*, Expr*>& target_id2expr,
      const std::unordered_map<IterDomain*, Expr*>& replay_id2expr);

  // Skip resize in both target and replay domains
  void skipResizes(
      const std::vector<Expr*>& target_exprs,
      const std::vector<Expr*>& replay_exprs);

 public:
  // When skip_resize is true, resize is ignored or in other words forwarded
  BestEffortReplay(
      const std::vector<IterDomain*>& replay_domain,
      const std::vector<IterDomain*>& target_domain,
      std::unordered_map<IterDomain*, IterDomain*> target2replay_map,
      std::unordered_map<IterDomain*, IterDomain*> replay_forward_id_map = {},
      std::unordered_map<IterDomain*, IterDomain*> target_forward_id_map = {},
      bool skip_replay_swizzle = true,
      bool skip_target_swizzle = true,
      bool skip_resize = false,
      bool error_on_failure = true);

  // Return iter domain map from target_domain IDs to their "replayed"
  // replay_domain IDs. If not in map, was not replayed.
  const std::unordered_map<IterDomain*, IterDomain*>& getReplay() const {
    return target2replay_id_map_;
  }

  // ids in replay that did not have matching transforms in target_domain
  const std::unordered_map<IterDomain*, size_t>& getUnorderedLeafIDs() {
    return loop_ids_;
  }

  // Returned ordered set of IDs in getUnorderedLeafIDs
  std::vector<IterDomain*> getLeafIDs() {
    std::set<std::pair<IterDomain*, size_t>, id_int_lt> ordered_set;
    for (auto entry : loop_ids_) {
      ordered_set.emplace(entry);
    }

    std::vector<IterDomain*> loop_vec_;
    loop_vec_.resize(ordered_set.size());
    std::transform(
        ordered_set.begin(),
        ordered_set.end(),
        loop_vec_.begin(),
        [](std::pair<IterDomain*, size_t> entry) { return entry.first; });
    return loop_vec_;
  }

  // Get a disjoint sets representing the equivalence of IterDomains. The
  // equivalence is defined by forwarding and replay. Two IterDomains are
  // equivalent if:
  // - They are mapped together through forwarding, or
  // - They are mapped together through replay
  // For example, if I have the following producer-consumer pair:
  //   T0[I0, I1]
  //   T1[(I0'*b1)*b2, I1'] = broadcast(T0)
  // Then there will be two equivalent sets"
  //   - {I1, I1'}
  //   - {I0, I0', I0'*b1, (I0'*b1)*b2}
  NVF_API DisjointSets<IterDomain*> getIterDomainEquivalence();

  // Runs a best effort replay that ignores broadcast axes that appear in
  // consumer that are not mapped to producer in logical_map.
  //
  // When skip_resize is true, resize is ignored or in other words forwarded
  NVF_API static BestEffortReplay replayCasP(
      const TensorView* consumer,
      const TensorView* producer,
      int64_t producer_compute_at_axis,
      const LogicalDomainMap& logical_map,
      bool skip_consumer_swizzle = true,
      bool skip_producer_swizzle = true,
      bool skip_resize = true);

  // Runs a best effort replay that ignores broadcast axes that appear in
  // consumer that are not mapped to producer in logical_map.
  //
  // When skip_resize is true, resize is ignored or in other words forwarded
  NVF_API static BestEffortReplay replayPasC(
      const TensorView* producer,
      const TensorView* consumer,
      int64_t consumer_compute_at_axis,
      const LogicalDomainMap& logical_map,
      bool skip_producer_swizzle = true,
      bool skip_consumer_swizzle = true,
      bool skip_resize = true);

  // Find the first position i where td1[i] is not the same as td2[i]. "Same"
  // means the DAG and input IDs to generate td1[i] and td2[i] are the same.
  // td1 and td2 are assumed to have some matching iter domains, as this is a
  // strict same-ness check.
  static int64_t findFirstMismatchedID(
      const TensorDomain* td1,
      const TensorDomain* td2);
};

} // namespace nvfuser

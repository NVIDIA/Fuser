// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/heuristic.h>
#include <scheduler/registry.h>

namespace nvfuser {

class Fusion;
class SchedulerRuntimeInfo;
class HeuristicDataCache;

// ================
// Greedy Scheduler
// ================
//
// The greedy scheduler aims to maximize fusion while accommodating
// scheduling constraints. Unlike some of the
// existing schedulers such as pointwise, a given fusion is scheduled
// as a collection of disjoint sets of tensors. Each set
// consists of tensors that can be scheduled uniformly.
// When a scheduling conflict occurs between tensor sets, it is
// resolved by staging conflicting tensors to shared memory.
//
// The scheduler currently uses certain operations, such as argsort
// and scan, as a trigger to be enabled. Additionally, only a limited
// set of operations are allowed to exist in the given fusion. See
// canScheduleCompileTime for the current set of required and
// supported operations.
//
//
// Scheduling Strategy
// -------------------
//
// The scheduling strategy is primarily based on two principles:
//
// 1. Avoid grid synchronizations
// 2. Accommodate a wide variety of operations
//
// The first principle means that BID types must be used uniformly
// across a fusion. More specifically, all iter domains parallelized
// with BID types must be mapped in the loop graph.
//
// The scheduler attempts to achieve the second principle through a
// three-step process:
// - First, it schedules all constrained tensors--tensor with specific
//   scheduling requirements. For example,
//   ArgsortOp and ScanOp currently require the sorted or scanned iter
//   domain to be parallelized with TID.
// - Next, the schedule of each constrained tensor is propagated
//   to a subset of tensors determined by a disjoint fusion partitioning
//   (see below).
// - Finally, because constrained tensors may have different
//   scheduling requirements, adjacent disjoint sets
//   might be scheduled differently with respect to TID types. The scheduler
//   uses SyncMap to identify conflicting producer-consumer pairs of tensors and
//   copies the producers to the shared memory to resolve these conflicts. Since
//   the first principle only permits conflicts with the TID types, the shared
//   memory followed by syncthreads is sufficient to resolve them.
//
// To partition a fusion into disjoint sets, each constrained
// tensor is marked as a reference and forms an initial set. The
// remaining unconstrained tensors are then iteratively added to a subset
// whose reference has matching iter domains, which allows the schedule of
// the reference tensor to be propagated uniformly.
//
//
// Scheduling with Reshape
// -----------------------
//
// For scheduling, reshapes are processed by propagating their split
// and merge transformations throughout a fusion before
// scheduling constrained tensors. This is to ensure all reshape
// transformations are preserved. Consequently,
// we cannot allow conflicting reshapes. For example, if an input tensor
// is reshaped in one way and also in another, and the two reshape
// do not have the same set of splits and merges, they are not allowed
// to exist in the same segment for scheduling. While this is a
// conservative constraint since some conflicts could be accommodated,
// it is currently enforced for simplicity.
//
// In addition to the above constraint, no reshape merge is allowed
// between the constrained and unconstrained ID groups since those two
// groups of IDs are scheduled separately.
//
// Note that some of these constraints could be lifted by cancelling
// reshape transformations, i.e., setting the root domain of a reshape
// output tensor as its loop domain. This strategy is partially used
// in the resize scheduler, however, it is not used here yet as it has
// its own problems (#4839).
//
//
// Scheduling with Scatter
// -----------------------
//
// To avoid grid synchronizations, the scattered dimension
// must not be parallelized with BID. The output tensor of a
// scatter operation is therefore designated as a constrained tensor.
//
// Additionally, scatter inputs and following consumer tensors also
// need to be designated as constrained tensors. This is because the
// loop domain of the output tensor is not connected with the logical
// domain and thus cannot be used as a reference tensor for any
// surrounding tensors. For example, consider the following
// operations:
//
// ```
// out = scatter(in, idx, src)
// out_consumer = add(out, 1)
// ```
//
// The `out` tensor is a constrained tensor, but that isn't the only
// constrained tensor. Suppose its loop domain is already scheduled.
// Since the loop domain is derived from the idx and src tensors,
// there's no obvious way to propagate its transformation to the
// `out_consumer` tensor. Therefore, the `out_consumer` tensor itself
// needs to be explicitly scheduled, which is done by designating it
// as a constrained tensor. Similarly, all three input tensors need to
// be explicitly scheduled due to the lack of connection between the
// logical and loop domains of the `out` tensor. Therefore, for each
// scatter operation, we mark four input and output tensors as
// constrained. Any following consumer tensors also need to be
// designated as such. For simplicity, however, a copy of the output
// tensor is inserted with cacheAfter, and only the copy tensor is
// designated as a constrained tensor.
//
// Another type of scheduling specific to scatter is allocation domain
// scheduling. The nvFuser ScatterOp is provided as an out-of-place
// operation, but it is internally implemented as an in-place
// operation. This means that the input and output tensors must share
// the same memory buffer. The compile-time checker ensures that the
// input and output tensors can safely share the same buffer, and
// after scheduling, both of the two tensors are assigned the same
// memory type and the allocation domain.
//
//
// Limitations
// -----------
//
// There are still a number of limitations. For a full list of issues
// we plan to address, see https://github.com/NVIDIA/Fuser/issues/5030.

class GreedyParams : public HeuristicParams {
 public:
  GreedyParams();

  // Heuristics parameters for each tensor. Currently, the only
  // heuristic parameter is batching, i.e., the number of items per
  // thread.
  //
  // Note that a single tensor can have multiple TvParms structs, one
  // as the consumer and one as each producer. For example, in the
  // case of scatter, we use the input tensor to store the scheduling
  // parameters for the input and also for the consumers of the
  // output. This input tensor could be produced by another
  // constrained op, so it could have different heuristic
  // parameters. This is not an issue as long as the difference is
  // resolved by staging data to the shared memory. See
  // HeuristicsBuilder::handle(ScatterOp*) for a concrete examle.
  struct TvParams {
    int64_t batch_size;
    std::string toString() const {
      return "Items per thread: " + std::to_string(batch_size);
    }
    bool operator==(const TvParams& other) const {
      return batch_size == other.batch_size;
    }
    bool operator!=(const TvParams& other) const {
      return !(batch_size == other.batch_size);
    }
  };

  using HeuristicParams::HeuristicParams;

  bool sameAs(const HeuristicParams* other_base) const override;

  std::string toString() const override;

  size_t hash() const override;

  std::unique_ptr<HeuristicParams> clone() const override;

  const TvParams& getConsumerParams(TensorView* tv) const {
    return consumer_to_params_.at(tv->name());
  }

  bool setConsumerParams(TensorView* tv, const TvParams& params) {
    return consumer_to_params_.emplace(tv->name(), params).second;
  }

  const TvParams& getProducerParams(
      TensorView* producer_tv,
      TensorView* consumer_tv) const {
    return producer_to_params_.at(
        std::make_pair(producer_tv->name(), consumer_tv->name()));
  }

  bool setProducerParams(
      TensorView* producer_tv,
      TensorView* consumer_tv,
      const TvParams& params) {
    return producer_to_params_
        .emplace(
            std::make_pair(producer_tv->name(), consumer_tv->name()), params)
        .second;
  }

  // Updates mappings by transferring parameters for old_tv to
  // new_tv. Mappings for old_tv are removed.
  void transferConsumerParams(TensorView* old_tv, TensorView* new_tv);

  bool hasConsumerParams(TensorView* consumer_tv);

  // Updates mappings by transferring producer parameters for
  // a pair of old_producer_tv and old_consumer_tv. The mapping for
  // the old pair is removed.
  void transferProducerParams(
      TensorView* old_producer_tv,
      TensorView* old_consumer_tv,
      TensorView* new_producer_tv,
      TensorView* new_consumer_tv);

  bool hasProducerParams(TensorView* producer_tv, TensorView* consumer_tv)
      const;

 private:
  // Parameters for each consumer tensor. Map from tensor names as
  // pointers may not be kept the same
  std::unordered_map<StmtNameType, TvParams> consumer_to_params_;

  struct NamePairHash {
    std::size_t operator()(
        const std::pair<StmtNameType, StmtNameType>& pair) const {
      return std::hash<StmtNameType>{}(pair.first) ^
          std::hash<StmtNameType>{}(pair.second);
    }
  };

  // Parameters for each producer tensor used for a consumer
  // tensor. Map from a pair of a producer name and its consumer name.
  std::unordered_map<
      std::pair<StmtNameType, StmtNameType>,
      TvParams,
      NamePairHash>
      producer_to_params_;
};

class GreedyScheduler : public SchedulerEntry {
 public:
  bool canScheduleCompileTime(Fusion* fusion) override;

  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache) override;

  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::Greedy;
  }
};

} // namespace nvfuser

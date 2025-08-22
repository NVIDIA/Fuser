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

// The Greedy scheduler aims to maximize fusion while accommodating
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
//   scheduling requirements.For example,
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
// There are still a number of limitations. For a full list of issues
// we plan to address, see https://github.com/NVIDIA/Fuser/issues/5030.
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

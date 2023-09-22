// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <array>
#include <ostream>
#include <string>

namespace nvfuser {

//! Each ScheduleHeuristic maps to a scheduler in distinct CPP files.
//! For instance, ScheduleHeuristic::PointWise maps to PointWiseScheduler in
//! pointwise.cpp.
//!
//!    Each of the scheduler needs to provide 3 interface functions:
//!
//!      1. canScheduleCompileTime(Fusion* fusion) :
//!
//!        This function contains compiled-time checks on the graph itself
//!        without runtime input information. Only `fusion` is given in the
//!        argument to make sure only compile-time available info is needed in
//!        the check.
//!
//!        This function is to be called exactly once on each segmented group
//!        created in a segmented fusion so this part will not contribute to
//!        dynamic shape latency.
//!
//!     2. canScheduleRunTime(
//!            Fusion* fusion,
//!            SchedulerRuntimeInfo& runtime_info,
//!           HeuristicSummary* data_cache = nullptr):
//!        This function contains all canSchedule checks that will have to
//!        involve runtime input information, and will be run both by the
//!        segmenter and the kernel cache. The latency of this function will
//!        contribute to dynamic shape latency so `data_cache` should be used as
//!        much as possible to save re-computation.
//!
//!     3. schedule(fusion):
//!
//!        This function will be called when compiling a kernel. It should apply
//!        scheduling to the given fusion

enum class ScheduleHeuristic {
  None,
  NoOp,
  PointWise,
  Reduction,
  InnerPersistent,
  InnerOuterPersistent,
  Persistent,
  Transpose,
  Matmul
};

//! Define a schedule table to loop over all the heuristics in priority order.
constexpr std::array<ScheduleHeuristic, 8> all_heuristics_in_priority_order = {
    ScheduleHeuristic::NoOp,
    ScheduleHeuristic::Reduction,
    ScheduleHeuristic::Transpose,
    ScheduleHeuristic::PointWise,
    ScheduleHeuristic::InnerPersistent,
    ScheduleHeuristic::InnerOuterPersistent,
    ScheduleHeuristic::Persistent,
    ScheduleHeuristic::Matmul};

std::string toString(ScheduleHeuristic sh);

std::ostream& operator<<(std::ostream& os, ScheduleHeuristic sh);

} // namespace nvfuser

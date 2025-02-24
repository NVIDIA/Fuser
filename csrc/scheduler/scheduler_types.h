// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <visibility.h>
#include <array>
#include <ostream>
#include <string>

namespace nvfuser {

//! Each SchedulerType maps to a scheduler in distinct CPP files.
//! For instance, SchedulerType::PointWise maps to PointWiseScheduler in
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
//!           HeuristicDataCache* data_cache = nullptr):
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

enum class SchedulerType {
  None,
  NoOp,
  PointWise,
  Matmul,
  Reduction,
  InnerPersistent,
  InnerOuterPersistent,
  OuterPersistent,
  Transpose,
  ExprEval,
  Resize,
  Communication
};

//! Define a schedule table to loop over all the heuristics in priority order.
constexpr std::array<SchedulerType, 11> all_heuristics_in_priority_order = {
    SchedulerType::ExprEval,
    SchedulerType::Communication,
    SchedulerType::NoOp,
    SchedulerType::Matmul,
    SchedulerType::Reduction,
    SchedulerType::Resize,
    SchedulerType::Transpose,
    SchedulerType::PointWise,
    SchedulerType::InnerPersistent,
    SchedulerType::OuterPersistent,
    SchedulerType::InnerOuterPersistent};

std::string toString(SchedulerType sh);

NVF_API std::ostream& operator<<(std::ostream& os, SchedulerType sh);

} // namespace nvfuser

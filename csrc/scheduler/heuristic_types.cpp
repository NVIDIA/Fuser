// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <exceptions.h>
#include <scheduler/heuristic_types.h>

namespace nvfuser {

std::string toString(ScheduleHeuristic sh) {
  switch (sh) {
    case ScheduleHeuristic::NoOp:
      return "no_op";
    case ScheduleHeuristic::PointWise:
      return "pointwise";
    case ScheduleHeuristic::Reduction:
      return "reduction";
    case ScheduleHeuristic::InnerPersistent:
      return "inner_persistent";
    case ScheduleHeuristic::OuterPersistent:
      return "outer_persistent";
    case ScheduleHeuristic::InnerOuterPersistent:
      return "inner_outer_persistent";
    case ScheduleHeuristic::Transpose:
      return "transpose";
    case ScheduleHeuristic::Matmul:
      return "matmul";
    case ScheduleHeuristic::None:
      return "none";
    default:
      NVF_ERROR(false, "undefined schedule");
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, ScheduleHeuristic sh) {
  os << toString(sh);
  return os;
}

} // namespace nvfuser

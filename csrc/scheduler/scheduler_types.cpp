// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <exceptions.h>
#include <scheduler/scheduler_types.h>

namespace nvfuser {

std::string toString(SchedulerType scheduler_type) {
  switch (scheduler_type) {
    case SchedulerType::NoOp:
      return "no_op";
    case SchedulerType::PointWise:
      return "pointwise";
    case SchedulerType::Reduction:
      return "reduction";
    case SchedulerType::InnerPersistent:
      return "inner_persistent";
    case SchedulerType::OuterPersistent:
      return "outer_persistent";
    case SchedulerType::InnerOuterPersistent:
      return "inner_outer_persistent";
    case SchedulerType::Transpose:
      return "transpose";
    case SchedulerType::Matmul:
      return "matmul";
    case SchedulerType::ExprEval:
      return "expr_eval";
    case SchedulerType::Resize:
      return "resize";
    case SchedulerType::Greedy:
      return "greedy";
    case SchedulerType::Communication:
      return "communication";
    case SchedulerType::None:
      return "none";
    default:
      NVF_THROW("undefined schedule");
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, SchedulerType scheduler_type) {
  os << toString(scheduler_type);
  return os;
}

} // namespace nvfuser

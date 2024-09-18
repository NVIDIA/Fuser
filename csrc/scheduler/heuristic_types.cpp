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

std::string toString(HeuristicType sh) {
  switch (sh) {
    case HeuristicType::NoOp:
      return "no_op";
    case HeuristicType::PointWise:
      return "pointwise";
    case HeuristicType::Reduction:
      return "reduction";
    case HeuristicType::InnerPersistent:
      return "inner_persistent";
    case HeuristicType::OuterPersistent:
      return "outer_persistent";
    case HeuristicType::InnerOuterPersistent:
      return "inner_outer_persistent";
    case HeuristicType::Transpose:
      return "transpose";
    case HeuristicType::Matmul:
      return "matmul";
    case HeuristicType::ExprEval:
      return "expr_eval";
    case HeuristicType::None:
      return "none";
    default:
      NVF_THROW("undefined schedule");
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, HeuristicType sh) {
  os << toString(sh);
  return os;
}

} // namespace nvfuser

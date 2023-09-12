// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ostream>
#include <string>

namespace nvfuser {

enum class ScheduleHeuristic {
  None,
  NoOp,
  PointWise,
  Reduction,
  Persistent,
  Transpose,
  Matmul
};

std::string toString(ScheduleHeuristic sh);

std::ostream& operator<<(std::ostream& os, ScheduleHeuristic sh);

} // namespace nvfuser

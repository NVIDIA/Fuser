/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <scheduler/normalization.h>
#include <scheduler/pointwise.h>
#include <scheduler/reduction.h>
#include <scheduler/transpose.h>

namespace nvfuser {

enum class TORCH_CUDA_CU_API ScheduleHeuristic {
  None,
  NoOp,
  PointWise,
  Reduction,
  Persistent,
  Transpose
};

} // namespace nvfuser

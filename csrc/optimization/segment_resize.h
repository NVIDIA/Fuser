// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <exceptions.h>
#include <optimization/optimization_pass.h>

namespace nvfuser::optimization {

//! ResizeSegmentPass inserts SegmenterSet nodes in order to guarantee that
//! inputs to PadOp and SliceOp are fusion segment inputs.
class ResizeSegmentPass : public OptimizationPass<ResizeSegmentPass> {
  friend class OptimizationPass<ResizeSegmentPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization

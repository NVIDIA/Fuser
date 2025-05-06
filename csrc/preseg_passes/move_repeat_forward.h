// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// When a repeat pattern is detected, move the repeating reshape
// forward. This can be helpful when a repeat is moved to the end of a
// segment as the actual computations of the segment can be done just
// for the pre-repeat shape.
//
//
// For example, when a pattern like below is detected:
//
// t0: [i0]
// t1 = broadcast(t0); // [i0, b(1)]
// t2 = expand(t1); // [i0, b(2)]
// t3 = reshape(t2); // [i0*2]
// t4 = op1(t3); // [i0*2]
// t5 = op2(t4); // [i0*2]
//
// This preseg pass will transform the fusion as shown below:
//
// t1 = broadcast(t0); // [i0, b(1)]
// t2 = expand(t1); // [i0, b(2)]
// t6 = squeeze(t2); // [i0]
// t4 = op1(t6); // [i0]
// t5 = op2(t4); // [i0]
// t7 = broadcast(t5); // [i0, b(1)]
// t8 = expand(t7); // [i0, b(2)]
// t9 = reshape(t8); // [i0*2]
//
// Here, it is assumed that both op1 and op2 do not have any
// dependency with the extent of i0. Normal arithmetic ops should be
// the case, however, for example, if the original op is slicing of
// i0*2, it can't be replaced with slicing of i0, so this
// transformation will not be applied.
class MoveRepeatForwardPass : public OptimizationPass<MoveRepeatForwardPass> {
  friend class OptimizationPass<MoveRepeatForwardPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "MoveRepeatForwardPass";
  }
};

} // namespace nvfuser::preseg_passes

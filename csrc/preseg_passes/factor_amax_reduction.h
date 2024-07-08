// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// FP8 support requires us to track a scaling factor for each tensor. This
// scaling factor requires us to compute absolute, maximum value of a tensor.
//
// Reference:
// https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
//
// Issue:
// Reduction and persistent schedulers expect all reduction axes to be
// consistent. Since the amax scaling factor is a reduction across all
// dimensions, it is not compatible with any partial reductions and is
// separated into a second kernel.
//
// Solution:
// An optimization is to split the amax reduction into two partial reductions.
// The first partial reduction will have the same axes as the reductions in the
// first kernel. The second parital reduction contains the remaining axes. This
// approach is faster because the reduction output is a smaller tensor, saving
// global memory bandwidth.
//
// Limitations:
// This optimization is restricted to the absolute, maximum reduction operation.
//
// Approach:
// 1. Get all reduction operations.
// 2. Find amax pattern.
//  * The reduction is an output tensor.
//  * It has a maximum reduction definition.
//  * Its producer is absolute unary operation.
// 3. Select upstream, partial reduction.
//  * Ideally, we would pick the partial reduction that lets us save the
//    smallest partial amax reduction. However, in this presegmentation pass,
//    we do not know how the fusion could be segmented nor the size of the
//    tensors.
//  * Given this lack of information, we select the partial reduction with
//    shortest dependency chain with amax reduction.
// 4. Given the reduction axes of the selected reference, factor amax into two
// partial reductions.
//
// Error Testing:
//  * A tensor with multiple reduction operations with incompatible axes
//  * Absolute maximum but is not a fusion output
//  * Absolute reduction output but not maximum definition
//  * A maximum output but not an absolute value

class FactorAmaxReductionPass
    : public OptimizationPass<FactorAmaxReductionPass> {
  friend class OptimizationPass<FactorAmaxReductionPass>;

 protected:
  NVF_API static void runPass(Fusion* fusion);
  static std::string name() {
    return "FactorAmaxReductionPass";
  }
};

} // namespace nvfuser::preseg_passes

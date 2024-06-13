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
// 1. Get all reduction operations.
// 2. Find amax pattern.
//  * The reduction is an output tensor.
//  * It has a maximum reduction definition.
//  * Its producer is absolute unary operation.
// 3. Gather all reductions along its dependency chains.
// 4. Given the reduction axes of those operations, factor amax into two partial
// reductions.
//
// The goal of pattern matching is find chain of operations. The general
// solution is to apply BFS to nodes of a tree.
//
// Identify amax scaling factor:
//  * We can use detailed information to avoid BFS. Specifically, the pattern
//    contains an output, so we search the fusion outputs for a maximum
//    reduction. Then, check if the producer of input TensorView is an
//    absolute unary operation.
//
// Find axes for partial reduction:
//  * Get input tensor associated with amax scaling factor.
//  * Get all dependency chains between input and scaling factor.
//  * Filter all reduction operations.
//
// Apply modification:
//  * Factor amax into partial reductions.
//
// Error Testing:
//  * A tensor with multiple reduction operations with incompatible axes
//  * Absolute maximum but is not a fusion output
//  * Absolute reduction output but not maximum definition
//  * A maximum output but not an absolute value

class FactorReductionPass : public OptimizationPass<FactorReductionPass> {
  friend class OptimizationPass<FactorReductionPass>;

 protected:
  NVF_API static void runPass(Fusion* fusion);
  static std::string name() {
    return "AmaxFactorReductionPass";
  }
};

} // namespace nvfuser::preseg_passes

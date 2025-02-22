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

// Translate concat-based repetitions to expand and reshape ops.
//
// For example, given the following fusion:
//
// t0 = [i0];
// t1 = cat({t0, t0}, -1);
//
// It will be translated to:
//
// t0 = [i0]
// t2 = broadcast(t0, {true, false});
// t3 = expand(t2, {2, i0});
// t4 = reshape(t3, {2 * i0});
//
// And all uses of t1 will be replaced by t4. This pattern commonly
// appears in RoPE, e.g.,
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L136.
// While the resize scheduler should be able to handle these patterns for
// pointwise-only segments, it is currently limited to only pointwise
// fusions only. This translation should promote larger fusions
// as it is not specific to any surrounding ops.
//
// Note that there's a potential downside compared to handling cat ops
// directly. Since insertion of broadcast IDs is not represented as
// Fusion IR expressions, a fusion may have more disconnected ID
// graphs after the translation, which may cause a segmentation that
// could be avoided with the original fusion. See
// PresegTest.TranslateRepeatToExpand4 for a concrete example.
class TranslateRepeatToExpand
    : public OptimizationPass<TranslateRepeatToExpand> {
  friend class OptimizationPass<TranslateRepeatToExpand>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "TranslateRepeatToExpand";
  }
};

} // namespace nvfuser::preseg_passes

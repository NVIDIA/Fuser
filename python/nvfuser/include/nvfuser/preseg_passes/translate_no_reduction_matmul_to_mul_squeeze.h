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

// Translate MatmulOp to multiply and squeeze when K=1.
//
// For example, given the following fusion:
//
// t0 = [i0, b1];
// t1 = [b2, i3];
// t2 = matmul(t0, t1);
//
// It will be translated to:
//
// t3 = broadcast(t0) // [i0, b4, b1]
// t4 = transpose(t1) // [i3, b2]
// t5 = broadcast(t4) // [b5, i3, b2]
// t6 = mul(t3, t5) // [i0, i3, b1]
// t2 = squeeze(t6) // [i0, i3]
class TranslateNoReductionMatmulToMulSqueeze
    : public OptimizationPass<TranslateNoReductionMatmulToMulSqueeze> {
  friend class OptimizationPass<TranslateNoReductionMatmulToMulSqueeze>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "TranslateNoReductionMatmulToMulSqueeze";
  }
};

} // namespace nvfuser::preseg_passes

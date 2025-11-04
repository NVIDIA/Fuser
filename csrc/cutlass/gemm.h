// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <string>

namespace nvfuser {

class Expr;
class Fusion;
class TensorView;
class Val;

class CutlassParams;

namespace cutlass_codegen {

// Whether we are doing grouped GEMM or regular scaled GEMM, the default
// epilogue is alpha*acc + beta*bias. Each of alpha, beta, and bias are
// optional. For grouped GEMM, problem_sizes, expert_offsets, and
// scale_factor_offsets must all be non-null, whereas they must be null for
// non-grouped GEMM.
struct MatmulPattern {
  Expr* mma;
  TensorView* a;
  TensorView* b;
  TensorView* a_scale;
  TensorView* b_scale;
  TensorView* alpha = nullptr;
  TensorView* beta = nullptr;
  TensorView* bias = nullptr;
  TensorView* problem_sizes = nullptr;
  TensorView* expert_offsets = nullptr;
  TensorView* scale_factor_offsets = nullptr;
  bool is_grouped = false;
};

//! Detects supported matmul patterns and fills out a MatmulPattern struct. Note
//! that the accumulator is pattern.mma->output(0)
MatmulPattern findPattern(Fusion* fusion);

//! Simply finds the position of a Val in fusion->inputs().
int64_t fusionInputPosition(Fusion* fusion, Val* v);

//! Simply finds the position of a Val in fusion->outputs().
int64_t fusionOutputPosition(Fusion* fusion, Val* v);

std::string generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params);

} // namespace cutlass_codegen

} // namespace nvfuser

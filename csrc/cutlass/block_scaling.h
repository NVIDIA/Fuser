// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <visibility.h>

#include <cstdint>
#include <vector>

namespace nvfuser {

class Fusion;
class TensorView;

namespace cutlass_codegen {

//! Pattern for block-scaled quantized outputs
struct NVF_API BlockScaledOutputPattern {
  TensorView* prescaled_output;
  TensorView* output;
  TensorView* block_scale_factors;
  TensorView* global_scale_factor;
  int64_t block_size;
};

//! Find block-scaled output patterns in a fusion
//! Returns a vector of patterns, one for each block-scaled output
NVF_API std::vector<BlockScaledOutputPattern> findBlockScaledOutputs(
    Fusion* fusion);

} // namespace cutlass_codegen

} // namespace nvfuser

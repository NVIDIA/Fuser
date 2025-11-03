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
#include <string>
#include <vector>

namespace nvfuser {

class Fusion;
class TensorView;

namespace cutlass_codegen {

//! Pattern for block-scaled quantized outputs
struct BlockScaledOutputPattern {
  TensorView* unquantized_output = nullptr;
  TensorView* quantized_output = nullptr;
  TensorView* block_scale_factors = nullptr;
  TensorView* global_scale_factor = nullptr;
  int64_t block_size = -1;

  std::string toString() const;
};

//! Find block-scaled output patterns in a fusion
//! Returns a vector of patterns, one for each block-scaled output
std::vector<BlockScaledOutputPattern> findBlockScaledOutputs(Fusion* fusion);

} // namespace cutlass_codegen

} // namespace nvfuser

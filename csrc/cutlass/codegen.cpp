// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/gemm.h>
#include <exceptions.h>

#include <string>

namespace nvfuser {

class Fusion;

namespace cutlass_codegen {

std::string generateCode(Fusion* fusion) {
  // TODO: match patterns and dispatch to different generators here
  if (hasNvfp4ScaledMmPattern(fusion)) {
    return generateNvfp4ScaledMmKernel(fusion);
  } else {
    NVF_THROW("Unsupported Fusion pattern for CUTLASS executor");
  }
}

} // namespace cutlass_codegen

} // namespace nvfuser

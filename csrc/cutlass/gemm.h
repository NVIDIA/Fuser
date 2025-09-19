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

class Fusion;

class CutlassParams;

namespace cutlass_codegen {

bool hasNvfp4ScaledMmPattern(Fusion* fusion);

std::string generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params);

} // namespace cutlass_codegen

} // namespace nvfuser

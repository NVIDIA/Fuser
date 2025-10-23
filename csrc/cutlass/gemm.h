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
class ScaledMmaOp;

namespace cutlass_codegen {

ScaledMmaOp* findScaledMmaOp(Fusion* fusion);

std::string generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params);

} // namespace cutlass_codegen

} // namespace nvfuser

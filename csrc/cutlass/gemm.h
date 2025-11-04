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
class Val;

class CutlassParams;

namespace cutlass_codegen {

//! Simply finds the position of a Val in fusion->inputs().
int64_t fusionInputPosition(Fusion* fusion, Val* v);

//! Simply finds the position of a Val in fusion->outputs().
int64_t fusionOutputPosition(Fusion* fusion, Val* v);

std::string generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params);

} // namespace cutlass_codegen

} // namespace nvfuser

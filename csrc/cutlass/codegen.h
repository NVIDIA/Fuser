// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <visibility.h>

#include <string>

namespace nvfuser {

class Fusion;
class CutlassParams;

namespace cutlass_codegen {

NVF_API std::string generateCode(Fusion* fusion, const CutlassParams& params);

} // namespace cutlass_codegen

} // namespace nvfuser

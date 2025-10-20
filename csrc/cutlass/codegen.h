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

struct DataType;
class Fusion;
class CutlassParams;

namespace cutlass_codegen {

NVF_API std::string generateCode(Fusion* fusion, const CutlassParams& params);

//! Convert a DataType to a cutlass dtype. For example, DataType::BFloat16 maps
//! to "cutlass::bfloat16_t"
//! https://docs.nvidia.com/cutlass/media/docs/cpp/fundamental_types.html#numeric-types
std::string dtypeToCutlass(const DataType& dtype);

//! Gets the numerical position of v in inputs of fusion. Throws an exception if
//! v is not a fusion input
int64_t fusionInputPosition(Fusion* fusion, Val* v);

//! Gets the numerical position of v in outputs of fusion. Throws an exception
//! if v is not a fusion output
int64_t fusionOutputPosition(Fusion* fusion, Val* v);

} // namespace cutlass_codegen

} // namespace nvfuser

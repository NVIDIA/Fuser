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

struct CutlassGeneratedCode {
  std::string code;

  // This is the number of tensors we will need to allocate for each invocation
  // of the kernel. The size of these tensors will depend on the size of the
  // inputs and will be queried using the dynamically loaded temp_tensor_sizes
  // function.
  int64_t num_temp_tensors;
};

NVF_API CutlassGeneratedCode
generateCode(Fusion* fusion, const CutlassParams& params);

//! Convert a DataType to a cutlass dtype. For example, DataType::BFloat16 maps
//! to "cutlass::bfloat16_t"
//! https://docs.nvidia.com/cutlass/media/docs/cpp/fundamental_types.html#numeric-types
std::string dtypeToCutlass(const DataType& dtype, bool force_unsigned = false);

} // namespace cutlass_codegen

} // namespace nvfuser

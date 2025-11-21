// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/codegen.h>
#include <cutlass/gemm.h>
#include <exceptions.h>
#include <type.h>

#include <string>

namespace nvfuser {

class Fusion;

class CutlassParams;

namespace cutlass_codegen {

std::string dtypeToCutlass(const DataType& dtype, bool force_unsigned) {
  NVF_ERROR(std::holds_alternative<PrimDataType>(dtype.type));
  if (force_unsigned) {
    switch (std::get<PrimDataType>(dtype.type)) {
      case (DataType::Float8_e4m3fn):
        return "cutlass::float_ue4m3_t";
      case (DataType::Float8_e8m0fnu):
        return "cutlass::float_ue8m0_t";
      default:
        NVF_THROW(
            "nvFuser DataType ",
            dtype,
            " is not supported with force_unsigned=true in CUTLASS executor");
    }
  } else {
    switch (std::get<PrimDataType>(dtype.type)) {
      case (DataType::Half):
        return "cutlass::half_t";
      case (DataType::BFloat16):
        return "cutlass::bfloat16_t";
      case (DataType::Float):
        return "float";
      case (DataType::Float8_e5m2):
        return "cutlass::float_e5m2_t";
      case (DataType::Float8_e4m3fn):
        return "cutlass::float_e4m3_t";
      case (DataType::Float8_e8m0fnu):
        return "cutlass::float_ue8m0_t";
      // TODO: support int, complex, and fp6 types
      case (DataType::Float4_e2m1fn):
        // Note that cutlass also provides cutlass::mx_float4_t<float_e2m1_t>>.
        // The difference between these is that the mxfp4 version uses a block
        // size of 32 while nvfp4 uses a block size of 16. In nvFuser the block
        // size is represented separately, and here we assume we're using nvfp.
        // TODO: if block scaling is tied to element type in nvFuser in the
        // future we can update this mapping
        return "cutlass::float_e2m1_t";
      default:
        NVF_THROW(
            "nvFuser DataType ",
            dtype,
            " is not supported in our CUTLASS executor");
    }
  }
}

CutlassGeneratedCode generateCode(Fusion* fusion, const CutlassParams& params) {
  return generateNvfp4ScaledMmKernel(fusion, params);
}

} // namespace cutlass_codegen

} // namespace nvfuser

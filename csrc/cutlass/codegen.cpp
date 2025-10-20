// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/gemm.h>
#include <exceptions.h>
#include <type.h>

#include <string>

namespace nvfuser {

class Fusion;

class CutlassParams;

namespace cutlass_codegen {

std::string dtypeToCutlass(const DataType& dtype) {
  NVF_ERROR(std::holds_alternative<PrimDataType>(dtype.type));
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
    // TODO: support int, complex, and fp6 types
    case (DataType::Float4_e2m1fn):
      // Note that cutlass also provides cutlass::mx_float4_t<float_e2m1_t>>.
      // The difference between these is that the mxfp4 version uses a block
      // size of 32 while nvfp4 uses a block size of 16. In nvFuser the block
      // size is represented separately, and here we assume we're using nvfp.
      // TODO: if block scaling is tied to element type in nvFuser in the future
      // we can update this mapping
      return "cutlass::nv_float4_t<float_e2m1_t>";
    default:
      NVF_THROW(
          "nvFuser DataType ",
          dtype,
          " is not supported in our CUTLASS executor");
  }
}

int64_t fusionInputPosition(Fusion* fusion, Val* v) {
  NVF_ERROR(v->isFusionInput());
  return static_cast<int64_t>(
      std::find(fusion->inputs().begin(), fusion->inputs().end(), v) -
      fusion->inputs().begin());
}

int64_t fusionOutputPosition(Fusion* fusion, Val* v) {
  NVF_ERROR(v->isFusionOutput());
  return static_cast<int64_t>(
      std::find(fusion->outputs().begin(), fusion->outputs().end(), v) -
      fusion->outputs().begin());
}

std::string generateCode(Fusion* fusion, const CutlassParams& params) {
  // TODO: match patterns and dispatch to different generators here
  if (findScaledMmaOp(fusion) != nullptr) {
    return generateNvfp4ScaledMmKernel(fusion, params);
  } else {
    NVF_THROW("Unsupported Fusion pattern for CUTLASS executor");
  }
}

} // namespace cutlass_codegen

} // namespace nvfuser

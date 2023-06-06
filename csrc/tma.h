// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <type.h>

// Note: [TMA support in nvFuser]

namespace nvfuser {

class Val;
class ExpressionEvaluator;

namespace tma {

// TensorMap is the descriptor for a tensor for TMA.
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
//
// The following note is copied from the above documentation:
//
// Tensor map objects are only supported on devices of compute capability 9.0 or
// higher. Additionally, a tensor map object is an opaque value, and, as such,
// should only be accessed through CUDA API calls.

#if (CUDA_VERSION >= 12000)
using TensorMap = CUtensorMap;
#else
// TODO: Is the size guaranteed to be 128 bytes? This is copied from CUTLASS:
// https://github.com/NVIDIA/cutlass/blob/87349d349605c1e24366fcbe8f04d0141dcb617b/include/cute/arch/copy_sm90_desc.hpp#L171-L175
struct TensorMap {
  char bytes[128];
};
#endif

enum class TensorMapSwizzleType { NoSwizzle, B32, B64, B128 };

struct TensorMapInfo {
  PrimDataType dtype;
  TensorMapSwizzleType swizzle;
  std::vector<Val*> gmem_shape;
  std::vector<Val*> gmem_strides; // column major
  std::vector<Val*> box_shape;
  std::vector<Val*> box_strides; // column major
  TensorMap operator()(void* gmem_base_ptr, ExpressionEvaluator& ee) const;
};

} // namespace tma
} // namespace nvfuser

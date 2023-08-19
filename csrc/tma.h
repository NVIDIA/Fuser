// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <ostream>

#include <type.h>

// Note: [TMA support in nvFuser]

namespace nvfuser {

namespace tma {

enum class TensorMapInterleave { NoInterleave, B16, B32 };
enum class TensorMapSwizzle { NoSwizzle, B32, B64, B128 };
enum class TensorMapL2Promotion { NoL2Promotion, B64, B128, B256 };
enum class TensorMapFloatOOBFill { NoOOBFill, NaN_Request_Zero_FMA };

std::ostream& operator<<(std::ostream& os, TensorMapInterleave interleave);
std::ostream& operator<<(std::ostream& os, TensorMapSwizzle swizzle);
std::ostream& operator<<(std::ostream& os, TensorMapL2Promotion l2_promotion);
std::ostream& operator<<(std::ostream& os, TensorMapFloatOOBFill oob_fill);

// Wrapper for driver API cuTensorMapEncodeTiled:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
//
// CUresult cuTensorMapEncodeTiled(
//     CUtensorMap* tensorMap,
//     CUtensorMapDataType tensorDataType,
//     cuuint32_t tensorRank,
//     void* globalAddress,
//     const cuuint64_t* globalDim,
//     const cuuint64_t* globalStrides,
//     const cuuint32_t* boxDim,
//     const cuuint32_t* elementStrides,
//     CUtensorMapInterleave interleave,
//     CUtensorMapSwizzle swizzle,
//     CUtensorMapL2promotion l2Promotion,
//     CUtensorMapFloatOOBfill oobFill);

Val* encodeTensorMapTiled(
    DataType data_type,
    Val* global_address,
    Val* global_dim,
    Val* global_strides,
    Val* box_dim,
    Val* element_strides,
    TensorMapInterleave interleave,
    TensorMapSwizzle swizzle,
    TensorMapL2Promotion l2_promotion,
    TensorMapFloatOOBFill oob_fill);

} // namespace tma
} // namespace nvfuser

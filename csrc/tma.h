// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <ostream>

#include <mma_type.h>
#include <type.h>

// Note: [TMA support in nvFuser]
//
// Recommended reading:
// https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
// https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator
//
// TMA (Tensor Memory Accelerator) is a hardware accelerator for transfering
// tensors up to 5D between global memory and shared memory. It supports tiled
// data and im2col data. nvFuser currently only supports tiled data.
//
// The tiled data transfer allows users to transfer a tile of a tensor between
// global memory and shared memory. It is helpful to think of the tile as a
// slice of the tensor. For example, if you have a 2D tensor you can do
// something like:
//   smem_tensor = gmem_tensor[i:i+16:2, j:j+32:1]
// Or in the language of affine transformations, the gmem_tensor must be
// transformed as:
//   root domain: [I1, I2]
//         split: [I1/16, 16, I2]
//         split: [I1/16, 8, 2, I2]
//         split: [I1/16, 8, 2, I2/32, 32]
//   loop domain: [I1/16, 8, 2, I2/32, 32]
//
// Because TMA does bulk transfer, there is a dedicated paralle type `Bulk` for
// it. In the above example, the gmem_tensor must be parallelized as
//   [I1/16, Bulk{8}, 2, I2/32, Bulk{32}]
// `Bulk` is a bit similar to `Vectorize` in some aspect, for example, both says
// that we are copying a batch of data. Indeed, while considering `Bulk` as
// representing a general N-dimensional slice that can have flexible extent and
// step, we can consider `Vectorize` as a limited version of `Bulk` that must
// represent a one-dimensional slice in the innermost dimension and the step of
// the slice must be 1 and the extent of the slice must be a power of 2. Like
// vectorize, a loop parallelized as `Bulk` is a trivial loop. Currently, we
// only support whole tensor copy, so the consumer tensor of TMA store can not
// be transformed, and all its `IterDomain`s must be parallelized as `Bulk`.
//
// To use TMA, we need to encode a tensor map of the global memory tensor we
// want to transfer. The tensor map is a set of parameters that describes the
// address and layout of the tensor in global memory, as well as the extent and
// step of our slice. There are also other features configured in this tensor
// map, such as: Does this tensor has overlap? How do we want to swizzle the
// shared memory to avoid bank conflict? How do we want L2 cache to be used? Do
// we want to automatically fill out-of-bound data? The tensor map must reside
// in constant memory, and nvFuser implements this as a kernel argument declared
// with __grid_constant__ qualifier. The tensor map is an opaque type to the
// user. It must be created by driver API cuTensorMapEncodeTiled, see:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
// In nvFuser, the task of calling `cuTensorMapEncodeTiled` to get the tensor
// map is on the executor, implemented at: `kir::EncodeTensorMapTiled::evaluate`
//
// TMA are supported in nvFuser by inline PTX in memory.cu. Currently, we do a
// sync immediately after the TMA PTX, which makes TMA synchronous. In the
// future, we will need more advanced sync analysis and insert the correct syncs
// at the correct point.
//
// During lowering, the index of the global tensor of the TMA expr must be
// lowered as a kir::TensorIndex whose index has dtype `struct` with name
// `Hopper::CpAsyncBulkTensorTileIndex`. The first field of this struct is the
// pointer to the tensor map in constant memory, and the second field is an
// array for the N-dimensional coordinate. The tensor map will be defined by an
// expression of `kir::EncodeTensorMapTiled`. The evaluation of this expression
// will be hoisted to the host, and will not be generated in the kernel. Because
// we currently only support whole tensor copy, the N-dimensional coordinate is
// always just zeros.
//
// Currently, because we only support very limited schedule, predicates for TMA
// exprs are not needed, therefore not generated.

namespace nvfuser {
namespace tma {

enum class TensorMapInterleave { NoInterleave, B16, B32 };
enum class TensorMapL2Promotion { NoL2Promotion, B64, B128, B256 };
enum class TensorMapFloatOOBFill { NoOOBFill, NaN_Request_Zero_FMA };

std::ostream& operator<<(std::ostream& os, TensorMapInterleave interleave);
std::ostream& operator<<(std::ostream& os, TensorMapL2Promotion l2_promotion);
std::ostream& operator<<(std::ostream& os, TensorMapFloatOOBFill oob_fill);

// Wrapper for:
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
    MmaInputSmemSwizzle swizzle,
    TensorMapL2Promotion l2_promotion,
    TensorMapFloatOOBFill oob_fill);

} // namespace tma
} // namespace nvfuser

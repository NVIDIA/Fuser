// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nvfuser {

// nvFuser index type
using nvfuser_index_t = int64_t;

// Function declarations for launching argsort test kernels

template <typename DataT>
void launchBasicArgsortTestKernel(
    cudaStream_t stream,
    DataT* input,
    int64_t* output_indices,
    int block_size,
    int items_per_thread,
    bool descending);

template <typename DataT>
void launchMultiDim2dArgsortTestKernel(
    cudaStream_t stream,
    DataT* input,
    int64_t* output_indices,
    int items_per_thread,
    bool descending);

template <typename DataT>
void launchMultiDim3dArgsortTestKernel(
    cudaStream_t stream,
    DataT* input,
    int64_t* output_indices,
    int items_per_thread,
    bool descending);

} // namespace nvfuser

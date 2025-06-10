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

// Function declarations for launching topk test kernels

template <typename DataT, int ITEMS_PER_THREAD>
void launch_basic_topk_test_kernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int block_size,
    int k,
    bool largest);

template <typename DataT, int ITEMS_PER_THREAD>
void launch_multi_dim_2d_topk_test_kernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest);

template <typename DataT, int ITEMS_PER_THREAD>
void launch_multi_dim_3d_topk_test_kernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest);

template <int ITEMS_PER_THREAD>
void launch_bfloat16_topk_test_kernel(
    cudaStream_t stream,
    __nv_bfloat16* input,
    __nv_bfloat16* output_values,
    int64_t* output_indices,
    int k,
    bool largest);

} // namespace nvfuser

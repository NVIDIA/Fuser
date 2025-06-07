// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace nvfuser {

// nvFuser index type
using nvfuser_index_t = int64_t;

// Function declarations for launching argsort test kernels

template<typename DataT>
void launch_basic_argsort_test_kernel(
    cudaStream_t stream,
    DataT* input,
    nvfuser_index_t* output_indices,
    int block_size,
    int items_per_thread,
    bool descending);

template<typename DataT>
void launch_multi_dim_2d_argsort_test_kernel(
    cudaStream_t stream,
    DataT* input,
    nvfuser_index_t* output_indices,
    int items_per_thread,
    bool descending);

template<typename DataT>
void launch_multi_dim_3d_argsort_test_kernel(
    cudaStream_t stream,
    DataT* input,
    nvfuser_index_t* output_indices,
    int items_per_thread,
    bool descending);

void launch_bfloat16_argsort_test_kernel(
    cudaStream_t stream,
    __nv_bfloat16* input,
    nvfuser_index_t* output_indices,
    int items_per_thread,
    bool descending);

void launch_convert_float_to_bfloat16(
    cudaStream_t stream,
    float* input_float,
    __nv_bfloat16* output_bfloat,
    int n);

} // namespace nvfuser
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
#include <torch/torch.h>
#include <cstdint>

namespace nvfuser {

// nvFuser index type
using nvfuser_index_t = int64_t;

// Function declarations for launching topk test kernels

template <typename DataT, int ITEMS_PER_THREAD>
void launchBasicTopkTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int block_size,
    int k,
    bool largest);

template <typename DataT, int ITEMS_PER_THREAD>
void launchMultiDim2dTopkTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest);

template <typename DataT, int ITEMS_PER_THREAD>
void launchMultiDim3dTopkTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest);

// Helper function to validate topk correctness
template <typename DataT>
bool validateTopkOrder(
    const at::Tensor& input_tensor,
    const at::Tensor& values_tensor,
    const at::Tensor& indices_tensor,
    int64_t k,
    bool largest = true);

} // namespace nvfuser

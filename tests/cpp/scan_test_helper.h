// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cstdint>

namespace nvfuser {

// nvFuser index type
using nvfuser_index_t = int64_t;

// Definition just used inside the scan tests to avoid including
// <csrc/type.h>, which isn't straightforward
enum class ScanBinaryOpType { Add, Mul, Max, Min, DiscountAdd };

// Function declarations for launching scan test kernels

// Basic scan test kernel with configurable block size
template <typename DataT, int ITEMS_PER_THREAD>
void launchBasicScanTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output,
    DataT init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);

// Discount scan test kernel: output[j] = output[j-1] * discount[j] + input[j]
template <typename DataT, int ITEMS_PER_THREAD>
void launchDiscountScanTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* discount,
    DataT* output,
    DataT init_value,
    int block_size);

// Validate scan operation correctness
// Compares scan output against PyTorch reference implementation
void validateScanResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    ScanBinaryOpType binary_op_type);

} // namespace nvfuser

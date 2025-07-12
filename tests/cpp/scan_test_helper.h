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

// Forward declaration to avoid heavy include dependencies
enum class BinaryOpType;

// nvFuser index type
using nvfuser_index_t = int64_t;

// Function declarations for launching scan test kernels

// Basic scan test kernel with configurable block size
template <typename DataT, int ITEMS_PER_THREAD>
void launchBasicScanTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output,
    DataT init_value,
    int block_size,
    BinaryOpType binary_op_type);

// Validate scan operation correctness
// Compares scan output against PyTorch reference implementation
bool validateScanResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType binary_op_type);

// Helper to get string name for binary operation type
const char* getBinaryOpName(BinaryOpType binary_op_type);

} // namespace nvfuser

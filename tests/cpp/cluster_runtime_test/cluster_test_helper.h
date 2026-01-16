// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include <ATen/Tensor.h>

namespace nvfuser {

template <typename T, int BLOCK_SIZE, int CLUSTER_SIZE>
void launchStoreSharedRemoteTestKernel(T* input, T* output);

void validateClusterStoreResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    int cluster_size);

template <typename T, int BLOCK_SIZE, int CLUSTER_SIZE, bool is_all_reduce>
void launchClusterReduceTestKernel(T* input, T* output);

void validateClusterReduceResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    bool is_all_reduce,
    int threads_per_block = 128);

} // namespace nvfuser

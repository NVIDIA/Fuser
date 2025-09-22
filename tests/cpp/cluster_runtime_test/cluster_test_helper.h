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

template <typename T, int BLOCK_SIZE, int CLUSTER_SIZE>
void launchStoreSharedRemoteTestKernel(
    cudaStream_t stream,
    T* input,
    T* output,
    int cluster_x,
    int cluster_y,
    int cluster_z);

void validateClusterStoreResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    int cluster_size);

} // namespace nvfuser

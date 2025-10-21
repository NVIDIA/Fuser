// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <optional>
#include <string>

#include <driver_api.h>

namespace nvfuser {

/**
 * Symmetric memory helpers for multi-device execution.
 *
 * - empty_strided_cuda_symmetric: allocate a CUDA symmetric memory tensor
 *   using the CUDA driver virtual memory APIs with read/write access on the
 *   current device. Currently supports only contiguous layouts.
 * - is_symmetric_memory_valid: validate that a tensor points to symmetric
 *   memory with the expected properties (access flags, allocation type,
 *   location, and handle type). Returns an empty string if valid; otherwise,
 *   returns a descriptive error message.
 */

// Allocate a symmetric CUDA tensor with the given size, stride, dtype and
// device. Only contiguous strides are currently supported.
at::Tensor empty_strided_cuda_symmetric(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    at::Device device,
    std::optional<uint64_t> alloc_id);

// Validate that the provided tensor is backed by symmetric CUDA memory.
// Returns "" if valid; otherwise an error description.
std::string is_symmetric_memory_valid(at::Tensor tensor);

int64_t get_granularity(
    const CUmemAllocationProp& prop,
    size_t requested_size_bytes);

} // namespace nvfuser

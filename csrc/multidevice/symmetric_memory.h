// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/ATen.h>

#include <driver_api.h>

namespace nvfuser {

/**
 * Symmetric memory helpers for multi-device execution.
 *
 * - allocateSymmetricTensor: allocate a CUDA symmetric memory tensor
 *   using the CUDA driver virtual memory APIs with read/write access on the
 *   current device. Currently supports only contiguous layouts.
 * - isSymmetricAllocationValid: validate that a tensor points to symmetric
 *   memory with the expected properties (access flags, allocation type,
 *   location, and handle type). Returns an empty string if valid; otherwise,
 *   returns a descriptive error message.
 */

// Allocate a symmetric CUDA tensor with the given size, stride, dtype and
// device. Only contiguous strides are currently supported.
at::Tensor allocateSymmetricTensor(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    at::Device device,
    std::optional<uint64_t> alloc_id);

// Validate that the provided tensor is backed by symmetric CUDA memory.
// Returns "" if valid; otherwise an error description.
std::string isSymmetricAllocationValid(at::Tensor tensor);

int64_t getGranularityForSymmetricMemory(
    const CUmemAllocationProp& prop,
    size_t requested_size_bytes);

} // namespace nvfuser

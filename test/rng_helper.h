// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

namespace at {
struct PhiloxCudaState;
}

namespace nvfuser {

enum RNGTest_t {
  Uniform,
  Normal,
};

template <typename T>
void launch_generate_random_numbers_kernel(
    cudaStream_t stream,
    T* output,
    int64_t size,
    at::PhiloxCudaState philox_args,
    RNGTest_t rng_test);

} // namespace nvfuser

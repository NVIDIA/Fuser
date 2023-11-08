// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Warning: this file should not include any header from nvFuser or pytorch
// (except raw headers). Compiling dynamic_type.h with nvcc is not supported.
// Compiling pytorch with nvcc is not supported either.

#include <cassert>
#include <cstdint>
#include <type_traits>

#include <ATen/cuda/detail/PhiloxCudaStateRaw.cuh>
#include <ATen/cuda/detail/UnpackRaw.cuh>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace nvfuser {

enum RNGTest_t {
  Uniform,
  Normal,
};

template <typename T>
__global__ void generate_random_numbers_kernel(
    T* output,
    int64_t size,
    at::PhiloxCudaState philox_args,
    RNGTest_t rng_test) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  auto seeds = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds), tid, std::get<1>(seeds), &state);

  double2 (*ref_rng_double)(curandStatePhilox4_32_10_t*);
  float4 (*ref_rng_float)(curandStatePhilox4_32_10_t*);
  switch (rng_test) {
    case RNGTest_t::Uniform: {
      ref_rng_double = curand_uniform2_double;
      ref_rng_float = curand_uniform4;
      break;
    }
    case RNGTest_t::Normal: {
      ref_rng_double = curand_normal2_double;
      ref_rng_float = curand_normal4;
      break;
    }
  }

  if (std::is_same<T, double>::value) {
    double2 result = ref_rng_double(&state);
    if (tid * 2 < size) {
      output[tid * 2] = result.x;
    }
    if (tid * 2 + 1 < size) {
      output[tid * 2 + 1] = result.y;
    }
  } else {
    auto is_float = std::is_same<T, float>::value;
    assert(is_float);
    float4 result = ref_rng_float(&state);
    if (tid * 4 < size) {
      output[tid * 4] = result.x;
    }
    if (tid * 4 + 1 < size) {
      output[tid * 4 + 1] = result.y;
    }
    if (tid * 4 + 2 < size) {
      output[tid * 4 + 2] = result.z;
    }
    if (tid * 4 + 3 < size) {
      output[tid * 4 + 3] = result.w;
    }
  }
}

template <typename T>
void launch_generate_random_numbers_kernel(
    cudaStream_t stream,
    T* output,
    int64_t size,
    at::PhiloxCudaState philox_args,
    RNGTest_t rng_test) {
  int64_t block = 128;
  int64_t block_elems = block * 16 / sizeof(T);
  int64_t grid = (size + block_elems - 1) / block_elems;
  generate_random_numbers_kernel<<<grid, block, 0, stream>>>(
      output, size, philox_args, rng_test);
  cudaDeviceSynchronize();
}

template void launch_generate_random_numbers_kernel<float>(
    cudaStream_t stream,
    float* output,
    int64_t size,
    at::PhiloxCudaState philox_args,
    RNGTest_t rng_test);

template void launch_generate_random_numbers_kernel<double>(
    cudaStream_t stream,
    double* output,
    int64_t size,
    at::PhiloxCudaState philox_args,
    RNGTest_t rng_test);

} // namespace nvfuser

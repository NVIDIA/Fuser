// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Warning: this file should not include any header from nvFuser. Compiling
// dynamic_type.h with nvcc is not supported.

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <exceptions.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <cassert>
#include <type_traits>

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

at::Tensor generate_random_numbers(
    int64_t size,
    at::ScalarType dtype,
    RNGTest_t rng_test) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  auto result = at::empty({size}, options);

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(4);
  }

  if (dtype == at::kFloat) {
    int64_t block = 128;
    int64_t block_elems = block * 4;
    int64_t grid = (size + block_elems - 1) / block_elems;
    generate_random_numbers_kernel<<<
        grid,
        block,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<float>(), size, rng_engine_inputs, rng_test);
  } else {
    NVF_CHECK(dtype == at::kDouble);
    int64_t block = 128;
    int64_t block_elems = block * 2;
    int64_t grid = (size + block_elems - 1) / block_elems;
    generate_random_numbers_kernel<<<
        grid,
        block,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<double>(), size, rng_engine_inputs, rng_test);
  }
  return result;
}

at::Tensor generate_uniform(int64_t size, at::ScalarType dtype) {
  return generate_random_numbers(size, dtype, RNGTest_t::Uniform);
}

at::Tensor generate_normal(int64_t size, at::ScalarType dtype) {
  return generate_random_numbers(size, dtype, RNGTest_t::Normal);
}

} // namespace nvfuser

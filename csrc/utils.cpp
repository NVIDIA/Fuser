// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/string_view.h>
#include <cuda_occupancy.h>
#include <nvrtc.h>

#include <cuda_utils.h>
#include <debug.h>
#include <options.h>
#include <runtime/executor_kernel_arg.h>
#include <utils.h>

#include <cstdlib>
#include <iostream>
#include <optional>

namespace nvfuser {

int getNumThreads() {
  const char* option_env_name = "NUM_THREADS";
  auto dump_options = getNvFuserEnv(option_env_name);
  if (dump_options == nullptr) {
    constexpr int default_num_threads = 8;
    return default_num_threads;
  }
  auto num_threads_value = std::atoi(dump_options);
  int max_num_threads = (int)std::thread::hardware_concurrency();
  return std::max(std::min(num_threads_value, max_num_threads), 1);
}

// TODO: clean this up with some knobs
c10::ThreadPool* getThreadPool() {
  static auto num_threads = getNumThreads();
  static c10::ThreadPool pool(num_threads);
  return &pool;
}

std::string debug_str(const at::Tensor& tensor) {
  std::stringstream ss;
  ss << "Tensor:";
  ss << " shape: " << tensor.sizes();
  ss << ", dtype: " << tensor.dtype();
  ss << ", device: " << tensor.device();
  ss << ", pointer: " << reinterpret_cast<size_t>(tensor.data_ptr());

  if (!tensor.is_contiguous()) {
    ss << ", strides: " << tensor.strides();
  }
  return ss.str();
}

bool is_cpu_scalar(const at::Tensor& tensor) {
  return tensor.device().is_cpu() && tensor.numel() == 1 && tensor.dim() == 0;
}

bool is_meta_scalar(const at::Tensor& tensor) {
  return tensor.device().is_meta() && tensor.numel() == 1 && tensor.dim() == 0;
}

int8_t getCommonDeviceCUDA(
    const KernelArgumentHolder& inputs,
    std::optional<int8_t> selected_device) {
  int8_t index = 0;
  // have we found or selected at least one device yet?
  bool found_device = false;
  if (selected_device.has_value()) {
    index = selected_device.value();
    found_device = true;
  }
  for (const auto& input : inputs) {
    if (!input.is<at::Tensor>() || !input.as<at::Tensor>().defined()) {
      continue;
    }
    const auto& device = input.as<at::Tensor>().device();
    // skip cpu scalar tensor as they'll be promoted to scalar later
    if (device.is_cpu() && is_cpu_scalar(input.as<at::Tensor>())) {
      continue;
    }
    NVF_CHECK(
        device.is_cuda() || device.is_meta(),
        "nvfuser only supports cuda or meta device, found: ",
        device);
    auto cur_index = device.index();
    if (found_device && index != cur_index) {
      return -1;
    }
    index = cur_index;
    found_device = true;
  }
  // When there are only scalar inputs, use selected_device or fall back to 0
  return found_device ? index : (int8_t)0;
}

// with cuda-12.9 or later, devices 10.0 support 256 bit vectorization
int64_t getMaxVectorizationSizeInBit() {
  // Cache for max vectorization size to avoid repeated system calls
  static std::optional<int64_t> cached_max_vectorization_size_in_bit =
      std::nullopt;

  if (cached_max_vectorization_size_in_bit.has_value()) {
    return cached_max_vectorization_size_in_bit.value();
  }
  int64_t max_vec_bits = 128;
  int sw_major, sw_minor;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcVersion(&sw_major, &sw_minor));
  if ((sw_major >= 12 && sw_minor >= 9) || (sw_major >= 13)) {
    int hw_major = at::cuda::getCurrentDeviceProperties()->major;
    if (hw_major >= 10) {
      max_vec_bits = 256;
    }
  }
  cached_max_vectorization_size_in_bit = max_vec_bits;
  return max_vec_bits;
}

bool useFallback() {
  // Keep this env var for compatibility
  const char* disable_fb_env = getNvFuserEnv("DISABLE_FALLBACK");
  bool fallback_disabled = disable_fb_env ? atoi(disable_fb_env) : false;
  fallback_disabled =
      fallback_disabled || isOptionDisabled(DisableOption::Fallback);

  return !fallback_disabled;
}

std::vector<int64_t> getTensorSizes(at::TensorTypePtr const& tensor_type) {
  NVF_ERROR(tensor_type != nullptr, "Input must be a Tensor.");
  auto optional_sizes = tensor_type->sizes().concrete_sizes();
  NVF_ERROR(
      optional_sizes.has_value(), "Missing size information for the tensor.");
  return optional_sizes.value();
}

int64_t getRegPerThreadGivenThreadsPerSM(int64_t threads_per_sm) {
  int num_partition = 0;
  int reg_allocation_granularity = 0;
  const auto prop = at::cuda::getCurrentDeviceProperties();
  cudaOccDeviceProp occ_prop(*prop);
  cudaOccSubPartitionsPerMultiprocessor(&num_partition, &occ_prop);
  cudaOccRegAllocationGranularity(&reg_allocation_granularity, &occ_prop);
  int64_t warp_size = prop->warpSize;
  int64_t num_warps = ceilDiv(threads_per_sm, warp_size);

  // warps could be distributed unevenly across partition
  int64_t max_warps_per_sm_partition = ceilDiv(num_warps, num_partition);
  // registers are evenly distributed across partitions, partition with most
  // wraps determins the maximum register available per warp
  int64_t max_reg_per_warp =
      prop->regsPerBlock / num_partition / max_warps_per_sm_partition;
  // clamp down to register allocation granularity at warp level
  int64_t effective_max_reg_per_warp = max_reg_per_warp /
      reg_allocation_granularity * reg_allocation_granularity;
  constexpr int64_t max_reg_count = 255;
  return std::min(max_reg_count, effective_max_reg_per_warp / warp_size);
}

int64_t getThreadsPerSMGivenRegPerThread(int64_t reg_per_thread) {
  int num_partition = 0;
  int reg_allocation_granularity = 0;
  const auto prop = at::cuda::getCurrentDeviceProperties();
  cudaOccDeviceProp occ_prop(*prop);
  cudaOccSubPartitionsPerMultiprocessor(&num_partition, &occ_prop);
  cudaOccRegAllocationGranularity(&reg_allocation_granularity, &occ_prop);
  int64_t warp_size = prop->warpSize;

  int64_t reg_per_warp =
      ceilDiv(reg_per_thread * warp_size, reg_allocation_granularity) *
      reg_allocation_granularity;
  int64_t warps_per_sm_partition =
      prop->regsPerBlock / reg_per_warp / num_partition;
  int64_t num_warps = warps_per_sm_partition * num_partition;
  return num_warps * warp_size;
}

const char* getNvFuserEnv(const char* env_name, const char* default_value) {
  // Prepend the default prefix and try if the variable is defined.
  const std::string prefix = "NVFUSER_";
  auto prefixed_name = prefix + env_name;
  auto env = std::getenv(prefixed_name.c_str());
  if (env) {
    return env;
  }

  // Try the PYTROCH_NVFUSER prefix as well, which is considered
  // deprecated.
  const std::string pyt_prefix = "PYTORCH_NVFUSER_";
  auto pyt_prefixed_name = pyt_prefix + env_name;
  auto pyt_env = std::getenv(pyt_prefixed_name.c_str());
  if (pyt_env) {
    TORCH_WARN(
        "Environment variable, ",
        pyt_prefixed_name,
        ", is deprecated. Please use ",
        prefixed_name,
        " instead.");
    return pyt_env;
  }

  return default_value;
}

size_t deviceAvailableSharedMemoryBytes() {
  const auto properties = at::cuda::getCurrentDeviceProperties();
  const size_t device_smem_limit = properties->sharedMemPerBlockOptin;
  return device_smem_limit;
}

} // namespace nvfuser

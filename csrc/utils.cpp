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
#include <debug.h>
#include <options.h>
#include <utils.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <unordered_map>

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

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-function")
void debugPrint(const c10::TensorTypePtr& type) {
  std::stringstream sizes_s;
  if (auto sizes = type->symbolic_sizes().sizes()) {
    for (const auto& shape_symbol : *sizes) {
      if (shape_symbol.is_static()) {
        sizes_s << shape_symbol.static_size() << ", ";
      } else {
        sizes_s << "s(" << *reinterpret_cast<const int64_t*>(&shape_symbol)
                << "), ";
      }
    }
  } else {
    sizes_s << "no size available";
  }
  debug() << "sizes:" << sizes_s.str() << std::endl;
  if (const auto& stride_properties = type->stride_properties().sizes()) {
    std::stringstream stride_s;
    std::stringstream index_s;
    std::stringstream contig_s;

    for (const auto& stride_property : *stride_properties) {
      if (stride_property.has_value() && stride_property->stride_.has_value()) {
        stride_s << *stride_property->stride_ << ", ";
      } else {
        stride_s << "?, ";
      }
      if (stride_property.has_value() &&
          stride_property->stride_index_.has_value()) {
        index_s << *stride_property->stride_index_ << ", ";
      } else {
        index_s << "?, ";
      }
      if (stride_property.has_value() &&
          stride_property->contiguous_.has_value()) {
        contig_s << *stride_property->contiguous_ << ", ";
      } else {
        contig_s << "?, ";
      }
    }
    debug() << "stride: " << stride_s.str() << std::endl;
    debug() << "stride index: " << index_s.str() << std::endl;
    debug() << "contiguous: " << contig_s.str() << std::endl;
  } else {
    debug() << "no stride properties available" << std::endl;
  }
}
C10_DIAGNOSTIC_POP()

bool is_zero_dim_tensor(const std::shared_ptr<c10::TensorType>& tensor_type) {
  return tensor_type && tensor_type->dim().has_value() &&
      tensor_type->dim().value() == 0;
}

bool is_zero_sized_tensor(const std::shared_ptr<c10::TensorType>& tensor_type) {
  auto opt_sizes = tensor_type->sizes().concrete_sizes();
  if (opt_sizes.has_value()) {
    auto sizes = opt_sizes.value();
    for (const auto& size : sizes) {
      if (size == 0) {
        return true;
      }
    }
  }
  return false;
}

bool is_cpu_scalar(const at::Tensor& tensor) {
  return tensor.device().is_cpu() && tensor.numel() == 1 && tensor.dim() == 0;
}

bool is_cpu_scalar(const c10::TensorType& tensor_type) {
  auto opt_device = tensor_type.device();
  auto opt_dim = tensor_type.dim();
  auto opt_numel = tensor_type.numel();
  return opt_device.has_value() && opt_device->is_cpu() &&
      opt_dim.has_value() && opt_numel.has_value() && opt_dim.value() == 0 &&
      opt_numel.value() == 1;
}

int8_t getCommonDeviceCUDA(
    const at::ArrayRef<c10::IValue>& inputs,
    std::optional<int8_t> selected_device) {
  int8_t index = 0;
  // have we found or selected at least one device yet?
  bool found_device = false;
  if (selected_device.has_value()) {
    index = selected_device.value();
    found_device = true;
  }
  for (const auto& input : inputs) {
    if (!input.isTensor()) {
      continue;
    }
    const auto& device = input.toTensor().device();
    // skip cpu scalar tensor as they'll be promoted to scalar later
    if (device.is_cpu() && is_cpu_scalar(input.toTensor())) {
      continue;
    }
    TORCH_CHECK(device.is_cuda(), "nvfuser only supports cuda device");
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

bool useFallback() {
  // Keep this env var for compatibility
  const char* disable_fb_env = getNvFuserEnv("DISABLE_FALLBACK");
  bool fallback_disabled = disable_fb_env ? atoi(disable_fb_env) : false;
  fallback_disabled =
      fallback_disabled || isOptionDisabled(DisableOption::Fallback);

  return !fallback_disabled;
}

std::vector<int64_t> getTensorSizes(at::TensorTypePtr const& tensor_type) {
  TORCH_INTERNAL_ASSERT(tensor_type != nullptr, "Input must be a Tensor.");
  auto optional_sizes = tensor_type->sizes().concrete_sizes();
  TORCH_INTERNAL_ASSERT(
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
  int warp_size = prop->warpSize;
  int num_warps = (int)ceilDiv(threads_per_sm, warp_size);

  // warps could be distributed unevenly across partition
  int max_warps_per_sm_partition = (int)ceilDiv(num_warps, num_partition);
  // registers are evenly distributed across partitions, partition with most
  // wraps determins the maximum register available per warp
  int max_reg_per_warp =
      prop->regsPerBlock / num_partition / max_warps_per_sm_partition;
  // clamp down to register allocation granularity at warp level
  int effective_max_reg_per_warp = max_reg_per_warp /
      reg_allocation_granularity * reg_allocation_granularity;
  return effective_max_reg_per_warp / warp_size;
}

int64_t getThreadsPerSMGivenRegPerThread(int64_t reg_per_thread) {
  int num_partition = 0;
  int reg_allocation_granularity = 0;
  const auto prop = at::cuda::getCurrentDeviceProperties();
  cudaOccDeviceProp occ_prop(*prop);
  cudaOccSubPartitionsPerMultiprocessor(&num_partition, &occ_prop);
  cudaOccRegAllocationGranularity(&reg_allocation_granularity, &occ_prop);
  int warp_size = prop->warpSize;

  int reg_per_warp =
      (int)ceilDiv(reg_per_thread * warp_size, reg_allocation_granularity) *
      reg_allocation_granularity;
  int warps_per_sm_partition =
      prop->regsPerBlock / reg_per_warp / num_partition;
  int num_warps = warps_per_sm_partition * num_partition;
  return num_warps * static_cast<int64_t>(warp_size);
}

char* getNvFuserEnv(const char* env_name) {
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

  return nullptr;
}

} // namespace nvfuser

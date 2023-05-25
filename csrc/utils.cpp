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
#include <utils.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <unordered_map>

namespace nvfuser {

namespace {

// thread_local variable used only for debugging/testing
thread_local bool overwrite_disable_fma = false;

// OptionEnum must be an enum like DebugDumpOption
template <typename OptionEnum>
auto parseEnvOptions(
    const char* option_env_name,
    const std::unordered_map<std::string, OptionEnum>& available_options) {
  // Make sure available_options includes all of the enum values
  TORCH_INTERNAL_ASSERT(
      available_options.size() == static_cast<int>(OptionEnum::EndOfOption),
      "Invalid available option map");

  std::unordered_map<OptionEnum, std::vector<std::string>> options;

  if (const char* dump_options = std::getenv(option_env_name)) {
    std::string_view options_view(dump_options);
    while (!options_view.empty()) {
      const auto comma_pos = options_view.find_first_of(',');
      const auto lparentheses_pos = options_view.find_first_of('(');
      auto end_pos = std::min(comma_pos, lparentheses_pos);
      const auto token = options_view.substr(0, end_pos);

      auto option_it = available_options.find(std::string(token));

      if (option_it == available_options.end()) {
        std::vector<std::string> option_values;
        std::transform(
            available_options.begin(),
            available_options.end(),
            std::back_inserter(option_values),
            [](const auto& kv) { return kv.first; });
        std::sort(option_values.begin(), option_values.end());
        TORCH_CHECK(
            false,
            "Parsing ",
            option_env_name,
            " failed. Invalid option: '",
            token,
            "'\nAvailable options: ",
            toDelimitedString(option_values));
      }

      options_view = (end_pos != std::string_view::npos)
          ? options_view.substr(end_pos + 1)
          : "";

      std::vector<std::string> arguments;
      if (lparentheses_pos < comma_pos) {
        bool closed = false;
        while (!closed) {
          const auto comma_pos = options_view.find_first_of(',');
          const auto rparentheses_pos = options_view.find_first_of(')');
          TORCH_CHECK(
              rparentheses_pos != std::string_view::npos,
              "Parsing ",
              option_env_name,
              " failed when parsing arguments for ",
              token,
              ". Syntax error: unclosed '('");
          auto end_pos = std::min(comma_pos, rparentheses_pos);
          arguments.emplace_back(options_view.substr(0, end_pos));

          options_view = options_view.substr(end_pos + 1);
          closed = (rparentheses_pos < comma_pos);
        }
        if (!options_view.empty()) {
          TORCH_CHECK(
              options_view[0] == ',',
              "Parsing ",
              option_env_name,
              " failed when parsing arguments for ",
              token,
              ". Syntax error: expect a ',' after ')'");
          options_view = options_view.substr(1);
        }
      }

      options[option_it->second] = std::move(arguments);
    }
  }

  return options;
}

auto parseDebugDumpOptions() {
  const std::unordered_map<std::string, DebugDumpOption> available_options = {
      {"fusion_ir", DebugDumpOption::FusionIr},
      {"fusion_ir_math", DebugDumpOption::FusionIrMath},
      {"fusion_ir_presched", DebugDumpOption::FusionIrPresched},
      {"kernel_ir", DebugDumpOption::KernelIr},
      {"ca_map", DebugDumpOption::ComputeAtMap},
      {"cuda_kernel", DebugDumpOption::CudaKernel},
      {"cuda_full", DebugDumpOption::CudaFull},
      {"cuda_to_file", DebugDumpOption::CudaToFile},
      {"debug_info", DebugDumpOption::DebugInfo},
      {"assert_memory_violation", DebugDumpOption::AssertMemoryViolation},
      {"launch_param", DebugDumpOption::LaunchParam},
      {"segmented_fusion", DebugDumpOption::FusionSegments},
      {"segmenter_logging", DebugDumpOption::FusionSegmenterLog},
      {"fusion_args", DebugDumpOption::FusionArgs},
      {"kernel_args", DebugDumpOption::KernelArgs},
      {"index_type", DebugDumpOption::IndexType},
      {"dump_eff_bandwidth", DebugDumpOption::EffectiveBandwidth},
      {"draw_segmented_fusion", DebugDumpOption::FusionSegmentsDrawing},
      {"ptxas_verbose", DebugDumpOption::PrintPtxasLog},
      {"buffer_reuse_verbose", DebugDumpOption::BufferReuseInfo},
      {"scheduler_params", DebugDumpOption::SchedulerDebug},
      {"scheduler_verbose", DebugDumpOption::SchedulerVerbose},
      {"parallel_dimensions", DebugDumpOption::ParallelDimensions},
      {"halo", DebugDumpOption::Halo},
      {"perf_debug_verbose", DebugDumpOption::PerfDebugVerbose},
      {"python_definition", DebugDumpOption::PythonDefinition},
      {"python_frontend_debug", DebugDumpOption::PythonFrontendDebug},
      {"transform_propagator", DebugDumpOption::TransformPropagator},
      {"cubin", DebugDumpOption::Cubin},
      {"sass", DebugDumpOption::Sass},
      {"ptx", DebugDumpOption::Ptx},
      {"bank_conflict", DebugDumpOption::BankConflictInfo},
      {"sync_map", DebugDumpOption::SyncMap},
      {"lower_verbose", DebugDumpOption::LowerVerbose},
      {"expr_simplify", DebugDumpOption::ExprSimplification},
      {"expr_sort", DebugDumpOption::ExprSort},
      {"loop_rotation", DebugDumpOption::LoopRotation},
      {"matmul_checks", DebugDumpOption::MatmulChecks}};

  return parseEnvOptions("PYTORCH_NVFUSER_DUMP", available_options);
}

const auto& getDebugDumpOptions() {
  static const auto options = parseDebugDumpOptions();
  return options;
}

auto parseDisableOptions() {
  const std::unordered_map<std::string, DisableOption> available_options = {
      {"compile_to_sass", DisableOption::CompileToSass},
      {"fallback", DisableOption::Fallback},
      {"fma", DisableOption::Fma},
      {"grouped_grid_welford_outer_opt",
       DisableOption::GroupedGridWelfordOuterOpt},
      {"index_hoist", DisableOption::IndexHoist},
      {"expr_simplify", DisableOption::ExprSimplify},
      {"nvtx", DisableOption::Nvtx},
      {"predicate_elimination", DisableOption::PredicateElimination},
      {"welford_vectorization", DisableOption::WelfordVectorization},
      {"magic_zero", DisableOption::MagicZero}};

  auto options = parseEnvOptions("PYTORCH_NVFUSER_DISABLE", available_options);

  if (options.count(DisableOption::Fma)) {
    TORCH_WARN(
        "fmad is disabled for nvrtc, which could negatively affect performance. Try removing `fma` from env variable PYTORCH_NVFUSER_DISABLE for optimal performance.");
  }

  return options;
}

const auto& getDisableOptions() {
  static const auto options = parseDisableOptions();
  return options;
}

auto parseEnableOptions() {
  const std::unordered_map<std::string, EnableOption> available_options = {
      {"complex", EnableOption::Complex},
      {"kernel_profile", EnableOption::KernelProfile},
      {"linear_decomposition", EnableOption::LinearDecomposition},
      {"conv_decomposition", EnableOption::ConvDecomposition},
      {"graph_op_fusion", EnableOption::GraphOp},
      {"kernel_db", EnableOption::KernelDb},
      {"warn_register_spill", EnableOption::WarnRegisterSpill}};

  return parseEnvOptions("PYTORCH_NVFUSER_ENABLE", available_options);
}

const auto& getEnableOptions() {
  static const auto options = parseEnableOptions();
  return options;
}

} // namespace

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
  std::cout << "sizes:" << sizes_s.str() << std::endl;
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
    std::cout << "stride: " << stride_s.str() << std::endl;
    std::cout << "stride index: " << index_s.str() << std::endl;
    std::cout << "contiguous: " << contig_s.str() << std::endl;
  } else {
    std::cout << "no stride properties available" << std::endl;
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

bool isDebugDumpEnabled(DebugDumpOption option) {
  return getDebugDumpOptions().count(option);
}

ThreadLocalFmaDisableOverwrite::ThreadLocalFmaDisableOverwrite(bool flag)
    : old_flag_{overwrite_disable_fma} {
  overwrite_disable_fma = flag;
}

ThreadLocalFmaDisableOverwrite::~ThreadLocalFmaDisableOverwrite() {
  overwrite_disable_fma = old_flag_;
}

bool isOptionDisabled(DisableOption option) {
  if (option == DisableOption::Fma && overwrite_disable_fma) {
    return true;
  }
  return getDisableOptions().count(option);
}

bool isOptionEnabled(EnableOption option) {
  return getEnableOptions().count(option);
}

const std::vector<std::string>& getDebugDumpArguments(DebugDumpOption option) {
  return getDebugDumpOptions().at(option);
}

const std::vector<std::string>& getDisableOptionArguments(
    DisableOption option) {
  return getDisableOptions().at(option);
}

const std::vector<std::string>& getEnableOptionArguments(EnableOption option) {
  return getEnableOptions().at(option);
}

bool useFallback() {
  // Keep this env var for compatibility
  const char* disable_fb_env = getenv("PYTORCH_NVFUSER_DISABLE_FALLBACK");
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
} // namespace nvfuser

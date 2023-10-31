// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <options.h>
#include <utils.h>

#include <algorithm>

namespace nvfuser {

namespace {
// OptionEnum must be an enum like DebugDumpOption
template <typename OptionEnum>
auto parseEnvOptions(
    const char* option_env_name,
    const std::unordered_map<std::string, OptionEnum>& available_options) {
  // Make sure available_options includes all of the enum values
  NVF_ERROR(
      available_options.size() == static_cast<int>(OptionEnum::EndOfOption),
      "Invalid available option map");

  std::unordered_map<OptionEnum, std::vector<std::string>> options;

  if (const char* dump_options = getNvFuserEnv(option_env_name)) {
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
        NVF_CHECK(
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
          NVF_CHECK(
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
          NVF_CHECK(
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

} // namespace

template <>
std::unordered_map<DebugDumpOption, std::vector<std::string>> Options<
    DebugDumpOption>::getOptionsFromEnv() {
  const std::unordered_map<std::string, DebugDumpOption> available_options = {
      {"assert_memory_violation", DebugDumpOption::AssertMemoryViolation},
      {"bank_conflict", DebugDumpOption::BankConflictInfo},
      {"buffer_reuse_verbose", DebugDumpOption::BufferReuseInfo},
      {"ca_map", DebugDumpOption::ComputeAtMap},
      {"cubin", DebugDumpOption::Cubin},
      {"cuda_full", DebugDumpOption::CudaFull},
      {"cuda_kernel", DebugDumpOption::CudaKernel},
      {"cuda_to_file", DebugDumpOption::CudaToFile},
      {"debug_info", DebugDumpOption::DebugInfo},
      {"draw_segmented_fusion", DebugDumpOption::FusionSegmentsDrawing},
      {"dump_eff_bandwidth", DebugDumpOption::EffectiveBandwidth},
      {"expr_simplify", DebugDumpOption::ExprSimplification},
      {"expr_sort", DebugDumpOption::ExprSort},
      {"expr_sort_verbose", DebugDumpOption::ExprSortVerbose},
      {"fusion_args", DebugDumpOption::FusionArgs},
      {"fusion_ir", DebugDumpOption::FusionIr},
      {"fusion_ir_concretized", DebugDumpOption::FusionIrConcretized},
      {"fusion_ir_preseg", DebugDumpOption::FusionIrPreseg},
      {"fusion_ir_math", DebugDumpOption::FusionIrMath},
      {"fusion_ir_presched", DebugDumpOption::FusionIrPresched},
      {"halo", DebugDumpOption::Halo},
      {"index_type", DebugDumpOption::IndexType},
      {"kernel_args", DebugDumpOption::KernelArgs},
      {"kernel_ir", DebugDumpOption::KernelIr},
      {"launch_param", DebugDumpOption::LaunchParam},
      {"loop_rotation", DebugDumpOption::LoopRotation},
      {"lower_verbose", DebugDumpOption::LowerVerbose},
      {"occupancy", DebugDumpOption::Occupancy},
      {"parallel_dimensions", DebugDumpOption::ParallelDimensions},
      {"perf_debug_verbose", DebugDumpOption::PerfDebugVerbose},
      {"ptx", DebugDumpOption::Ptx},
      {"ptxas_verbose", DebugDumpOption::PrintPtxasLog},
      {"python_definition", DebugDumpOption::PythonDefinition},
      {"python_frontend_debug", DebugDumpOption::PythonFrontendDebug},
      {"sass", DebugDumpOption::Sass},
      {"segmented_fusion", DebugDumpOption::FusionSegments},
      {"segmenter_logging", DebugDumpOption::FusionSegmenterLog},
      {"scheduler_params", DebugDumpOption::SchedulerDebug},
      {"scheduler_verbose", DebugDumpOption::SchedulerVerbose},
      {"sync_map", DebugDumpOption::SyncMap},
      {"transform_propagator", DebugDumpOption::TransformPropagator}};

  return parseEnvOptions("DUMP", available_options);
}

template <>
std::unordered_map<EnableOption, std::vector<std::string>> Options<
    EnableOption>::getOptionsFromEnv() {
  const std::unordered_map<std::string, EnableOption> available_options = {
      {"complex", EnableOption::Complex},
      {"conv_decomposition", EnableOption::ConvDecomposition},
      {"graph_op_fusion", EnableOption::GraphOp},
      {"kernel_db", EnableOption::KernelDb},
      {"kernel_profile", EnableOption::KernelProfile},
      {"linear_decomposition", EnableOption::LinearDecomposition},
      {"memory_promotion", EnableOption::MemoryPromotion},
      {"warn_register_spill", EnableOption::WarnRegisterSpill}};

  return parseEnvOptions("ENABLE", available_options);
}

template <>
std::unordered_map<DisableOption, std::vector<std::string>> Options<
    DisableOption>::getOptionsFromEnv() {
  const std::unordered_map<std::string, DisableOption> available_options = {
      {"compile_to_sass", DisableOption::CompileToSass},
      {"expr_simplify", DisableOption::ExprSimplify},
      {"fallback", DisableOption::Fallback},
      {"fma", DisableOption::Fma},
      {"grouped_grid_welford_outer_opt",
       DisableOption::GroupedGridWelfordOuterOpt},
      {"index_hoist", DisableOption::IndexHoist},
      {"magic_zero", DisableOption::MagicZero},
      {"nvtx", DisableOption::Nvtx},
      {"parallel_compile", DisableOption::ParallelCompile},
      {"parallel_serde", DisableOption::ParallelSerde},
      {"predicate_elimination", DisableOption::PredicateElimination},
      {"kernel_reuse", DisableOption::KernelReuse},
      {"var_name_remapping", DisableOption::VarNameRemapping},
      {"welford_vectorization", DisableOption::WelfordVectorization},
      {"reuse_mismatched_type_registers",
       DisableOption::ReuseMismatchedTypeRegisters}};

  auto options = parseEnvOptions("DISABLE", available_options);

  if (options.count(DisableOption::Fma)) {
    TORCH_WARN(
        "fmad is disabled for nvrtc, which could negatively affect performance. Try removing `fma` from env variable NVFUSER_DISABLE for optimal performance.");
  }

  return options;
}

template <>
std::unordered_map<ProfilerOption, std::vector<std::string>> Options<
    ProfilerOption>::getOptionsFromEnv() {
  const std::unordered_map<std::string, ProfilerOption> available_options = {
      {"enabled", ProfilerOption::Enabled},
      {"enabled.nocupti", ProfilerOption::EnabledNocupti},
      {"print", ProfilerOption::Print},
      {"print.nocupti", ProfilerOption::PrintNocupti},
      {"print.verbose", ProfilerOption::PrintVerbose},
  };

  auto options = parseEnvOptions("PROF", available_options);

  return options;
}

namespace {

// These may need to be thread local, or their modifications may need to
// be protected by mutual exclusion for thread safety. At this
// moment, the correctness of modifying option values has to be
// guaranteed by the modifying code.

DebugDumpOptions active_dump_options;

EnableOptions active_enable_options;

DisableOptions active_disable_options;

ProfilerOptions active_profiler_options;

} // namespace

template <>
Options<DebugDumpOption>& OptionsGuard<DebugDumpOption>::getCurOptions() {
  return active_dump_options;
}

template <>
Options<EnableOption>& OptionsGuard<EnableOption>::getCurOptions() {
  return active_enable_options;
}

template <>
Options<DisableOption>& OptionsGuard<DisableOption>::getCurOptions() {
  return active_disable_options;
}

template <>
Options<ProfilerOption>& OptionsGuard<ProfilerOption>::getCurOptions() {
  return active_profiler_options;
}

bool isDebugDumpEnabled(DebugDumpOption option) {
  return DebugDumpOptionsGuard::getCurOptions().has(option);
}

const std::vector<std::string>& getDebugDumpArguments(DebugDumpOption option) {
  return DebugDumpOptionsGuard::getCurOptions().getArgs(option);
}

bool isOptionEnabled(EnableOption option) {
  return EnableOptionsGuard::getCurOptions().has(option);
}

const std::vector<std::string>& getEnableOptionArguments(EnableOption option) {
  return EnableOptionsGuard::getCurOptions().getArgs(option);
}

bool isOptionDisabled(DisableOption option) {
  return DisableOptionsGuard::getCurOptions().has(option);
}

const std::vector<std::string>& getDisableOptionArguments(
    DisableOption option) {
  return DisableOptionsGuard::getCurOptions().getArgs(option);
}

bool isProfilerEnabled() {
  return ProfilerOptionsGuard::getCurOptions().hasAny();

}
bool isProfilerEnabledWithoutCupti() {
  return ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::EnabledNocupti) ||
         ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::PrintNocupti);
}
bool isProfilerPrintingEnabled() {
  return ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::Print) ||
         ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::PrintNocupti) ||
         ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::PrintVerbose);
}
bool isProfilerPrintingVerbose() {
  return ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::PrintVerbose);
}

const std::vector<std::string>& getDisableOptionArguments(
    ProfilerOption option) {
  return ProfilerOptionsGuard::getCurOptions().getArgs(option);
}

} // namespace nvfuser

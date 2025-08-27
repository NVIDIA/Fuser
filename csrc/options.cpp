// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <options.h>
#include <utils.h>

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
      {"bank_conflict", DebugDumpOption::BankConflictInfo},
      {"buffer_reuse_verbose", DebugDumpOption::BufferReuseInfo},
      {"ca_map", DebugDumpOption::ComputeAtMap},
      {"cubin", DebugDumpOption::Cubin},
      {"cuda_full", DebugDumpOption::CudaFull},
      {"cuda_kernel", DebugDumpOption::CudaKernel},
      {"cuda_to_file", DebugDumpOption::CudaToFile},
      {"draw_segmented_fusion", DebugDumpOption::FusionSegmentsDrawing},
      {"expr_simplify", DebugDumpOption::ExprSimplification},
      {"expr_sort", DebugDumpOption::ExprSort},
      {"expr_sort_verbose", DebugDumpOption::ExprSortVerbose},
      {"ftrace", DebugDumpOption::FunctionTrace},
      {"fusion_args", DebugDumpOption::FusionArgs},
      {"fusion_ir", DebugDumpOption::FusionIr},
      {"fusion_ir_concretized", DebugDumpOption::FusionIrConcretized},
      {"fusion_ir_graph", DebugDumpOption::FusionIrGraph},
      {"fusion_ir_math", DebugDumpOption::FusionIrMath},
      {"fusion_ir_original", DebugDumpOption::FusionIrOriginal},
      {"fusion_ir_presched", DebugDumpOption::FusionIrPresched},
      {"fusion_ir_preseg", DebugDumpOption::FusionIrPreseg},
      {"global_zeroed_memory", DebugDumpOption::GlobalZeroedMemory},
      {"host_ir_lowering_logging", DebugDumpOption::HostIrLoweringLogging},
      {"host_ir", DebugDumpOption::HostIr},
      {"host_ir_jit", DebugDumpOption::HostIrJit},
      {"index_type", DebugDumpOption::IndexType},
      {"indexing_verbose", DebugDumpOption::IndexingVerbose},
      {"kernel_args", DebugDumpOption::KernelArgs},
      {"kernel_ir", DebugDumpOption::KernelIr},
      {"launch_param", DebugDumpOption::LaunchParam},
      {"loop_rotation", DebugDumpOption::LoopRotation},
      {"lower_verbose", DebugDumpOption::LowerVerbose},
      {"occupancy", DebugDumpOption::Occupancy},
      {"parallel_dimensions", DebugDumpOption::ParallelDimensions},
      {"perf_debug_verbose", DebugDumpOption::PerfDebugVerbose},
      {"pre_segmenter_logging", DebugDumpOption::PreSegmenterLogging},
      {"predicate_elimination", DebugDumpOption::PredicateElimination},
      {"ptx", DebugDumpOption::Ptx},
      {"ptxas_verbose", DebugDumpOption::PrintPtxasLog},
      {"python_definition", DebugDumpOption::PythonDefinition},
      {"python_definition_segments", DebugDumpOption::PythonDefinitionSegments},
      {"python_frontend_debug", DebugDumpOption::PythonFrontendDebug},
      {"sass", DebugDumpOption::Sass},
      {"sass_to_file", DebugDumpOption::SassToFile},
      {"segmented_fusion", DebugDumpOption::FusionSegments},
      {"segmenter_logging", DebugDumpOption::FusionSegmenterLog},
      {"scheduler_params", DebugDumpOption::SchedulerDebug},
      {"dynamic_shared_memory", DebugDumpOption::DynamicSharedMemory},
      {"scheduler_verbose", DebugDumpOption::SchedulerVerbose},
      {"sync_map", DebugDumpOption::SyncMap},
      {"transform_propagator", DebugDumpOption::TransformPropagator},
      {"communication", DebugDumpOption::Communication}};

  return parseEnvOptions("DUMP", available_options);
}

const std::unordered_map<std::string, EnableOption>& getEnableOptions() {
  static const std::unordered_map<std::string, EnableOption> available_options =
      {
          {"fuse_matmul", EnableOption::FuseMatmul},
          {"fuse_multiple_matmuls", EnableOption::FuseMultipleMatmuls},
          {"greedy_scheduler", EnableOption::GreedyScheduler},
          {"id_model", EnableOption::IdModel},
          {"id_model_extra_validation", EnableOption::IdModelExtraValidation},
          {"io_to_lower_precision", EnableOption::IoToLowerPrecision},
          {"kernel_db", EnableOption::KernelDb},
          {"kernel_debug", EnableOption::KernelDebug},
          {"kernel_lineinfo", EnableOption::KernelLineInfo},
          {"kernel_profile", EnableOption::KernelProfile},
          {"memory_promotion", EnableOption::MemoryPromotion},
          {"reuse_zeroed_memory", EnableOption::ReuseZeroedMemory},
          {"static_fusion_count", EnableOption::StaticFusionCount},
          {"wait_debugger", EnableOption::WaitDebugger},
          {"warn_register_spill", EnableOption::WarnRegisterSpill},
          {"ws_normalization", EnableOption::WarpSpecializedNormalization},
          {"host_ir_lowering", EnableOption::HostIrLowering},
          {"fast_math", EnableOption::FastMath},
      };
  return available_options;
}

template <>
std::unordered_map<EnableOption, std::vector<std::string>> Options<
    EnableOption>::getOptionsFromEnv() {
  const auto& available_options = getEnableOptions();
  return parseEnvOptions("ENABLE", available_options);
}

std::optional<EnableOption> stringToEnableOption(
    const std::string& enable_option) {
  const auto& opts = getEnableOptions();
  auto it = opts.find(enable_option);
  if (it != opts.end()) {
    return it->second;
  }
  return std::nullopt;
}

const std::unordered_map<std::string, DisableOption>& getDisableOptions() {
  static const std::unordered_map<std::string, DisableOption>
      available_options = {
          {"compile_to_sass", DisableOption::CompileToSass},
          {"contig_indexing", DisableOption::ContigIndexing},
          {"expr_simplify", DisableOption::ExprSimplify},
          {"fallback", DisableOption::Fallback},
          {"fma", DisableOption::Fma},
          {"grouped_grid_welford_outer_opt",
           DisableOption::GroupedGridWelfordOuterOpt},
          {"id_model", DisableOption::IdModel},
          {"index_hoist", DisableOption::IndexHoist},
          {"magic_zero", DisableOption::MagicZero},
          {"matmul_expr_eval", DisableOption::MatmulExprEval},
          {"nvrtc_caching", DisableOption::NvrtcCaching},
          {"nvtx", DisableOption::Nvtx},
          {"parallel_compile", DisableOption::ParallelCompile},
          {"parallel_serde", DisableOption::ParallelSerde},
          {"predicate_elimination", DisableOption::PredicateElimination},
          {"python_inline_definitions", DisableOption::PythonInlineDefinitions},
          {"kernel_reuse", DisableOption::KernelReuse},
          {"var_name_remapping", DisableOption::VarNameRemapping},
          {"welford_vectorization", DisableOption::WelfordVectorization},
          {"resize_scheduler", DisableOption::ResizeScheduler},
          {"reuse_mismatched_type_registers",
           DisableOption::ReuseMismatchedTypeRegisters},
          {"multidevice", DisableOption::Multidevice}};
  return available_options;
}

template <>
std::unordered_map<DisableOption, std::vector<std::string>> Options<
    DisableOption>::getOptionsFromEnv() {
  const auto& available_options = getDisableOptions();
  auto options = parseEnvOptions("DISABLE", available_options);

  if (options.count(DisableOption::Fma)) {
    TORCH_WARN(
        "fmad is disabled for nvrtc, which could negatively affect "
        "performance. Try removing `fma` from env variable NVFUSER_DISABLE for "
        "optimal performance.");
  }

  return options;
}

std::optional<DisableOption> stringToDisableOption(
    const std::string& disable_option) {
  const auto& opts = getDisableOptions();
  auto it = opts.find(disable_option);
  if (it != opts.end()) {
    return it->second;
  }
  return std::nullopt;
}

template <>
std::unordered_map<ProfilerOption, std::vector<std::string>> Options<
    ProfilerOption>::getOptionsFromEnv() {
  const std::unordered_map<std::string, ProfilerOption> available_options = {
      {"enable", ProfilerOption::Enable},
      {"enable.nocupti", ProfilerOption::EnableNocupti},
      {"print", ProfilerOption::Print},
      {"print.nocupti", ProfilerOption::PrintNocupti},
      {"print.verbose", ProfilerOption::PrintVerbose},
  };

  auto options = parseEnvOptions("PROF", available_options);

  return options;
}

template <>
Options<DebugDumpOption>& OptionsGuard<DebugDumpOption>::getCurOptions() {
  // Note: Make options thread_local.
  // We want the behavior that new threads would inherit options from the *base*
  // threads. We need to figure out how to automatically do that before
  // switching to thread_local. For now we are using mutex to guard option
  // access, which is necessary to avoid data racing.
  static DebugDumpOptions active_dump_options;
  return active_dump_options;
}

template <>
Options<EnableOption>& OptionsGuard<EnableOption>::getCurOptions() {
  static EnableOptions active_enable_options;
  return active_enable_options;
}

template <>
Options<DisableOption>& OptionsGuard<DisableOption>::getCurOptions() {
  static DisableOptions active_disable_options;
  return active_disable_options;
}

template <>
Options<ProfilerOption>& OptionsGuard<ProfilerOption>::getCurOptions() {
  static ProfilerOptions active_profiler_options;
  return active_profiler_options;
}

bool isDebugDumpEnabled(DebugDumpOption option) {
  return DebugDumpOptionsGuard::getCurOptions().has(option);
}

const std::vector<std::string>& getDebugDumpArguments(DebugDumpOption option) {
  return DebugDumpOptionsGuard::getCurOptions().getArgs(option);
}

bool hasDebugDumpArgument(DebugDumpOption option, const std::string& arg) {
  return DebugDumpOptionsGuard::getCurOptions().hasArg(option, arg);
}

bool isOptionEnabled(EnableOption option) {
  return EnableOptionsGuard::getCurOptions().has(option);
}

const std::vector<std::string>& getEnableOptionArguments(EnableOption option) {
  return EnableOptionsGuard::getCurOptions().getArgs(option);
}

bool hasEnableOptionArgument(EnableOption option, const std::string& arg) {
  return EnableOptionsGuard::getCurOptions().hasArg(option, arg);
}

bool isOptionDisabled(DisableOption option) {
  return DisableOptionsGuard::getCurOptions().has(option);
}

const std::vector<std::string>& getDisableOptionArguments(
    DisableOption option) {
  return DisableOptionsGuard::getCurOptions().getArgs(option);
}

bool hasDisableOptionArguments(DisableOption option, const std::string& arg) {
  return DisableOptionsGuard::getCurOptions().hasArg(option, arg);
}

bool isProfilerEnabled() {
  return ProfilerOptionsGuard::getCurOptions().hasAny();
}
bool isProfilerEnabledWithCupti() {
  return ProfilerOptionsGuard::getCurOptions().hasAny() &&
      !(ProfilerOptionsGuard::getCurOptions().has(
            ProfilerOption::EnableNocupti) ||
        ProfilerOptionsGuard::getCurOptions().has(
            ProfilerOption::PrintNocupti));
}
bool isProfilerPrintingEnabled() {
  return ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::Print) ||
      ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::PrintNocupti) ||
      ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::PrintVerbose);
}
bool isProfilerPrintingVerbose() {
  return ProfilerOptionsGuard::getCurOptions().has(
      ProfilerOption::PrintVerbose);
}

} // namespace nvfuser

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
      available_options.size() == static_cast<int>(OptionEnum::EndOfEnum),
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
      {"debug_info", DebugDumpOption::DebugInfo},
      {"draw_segmented_fusion", DebugDumpOption::FusionSegmentsDrawing},
      {"dump_eff_bandwidth", DebugDumpOption::EffectiveBandwidth},
      {"expr_simplify", DebugDumpOption::ExprSimplification},
      {"expr_sort", DebugDumpOption::ExprSort},
      {"expr_sort_verbose", DebugDumpOption::ExprSortVerbose},
      {"ftrace", DebugDumpOption::FunctionTrace},
      {"fusion_args", DebugDumpOption::FusionArgs},
      {"fusion_ir_original", DebugDumpOption::FusionIrOriginal},
      {"fusion_ir_concretized", DebugDumpOption::FusionIrConcretized},
      {"fusion_ir_preseg", DebugDumpOption::FusionIrPreseg},
      {"fusion_ir_presched", DebugDumpOption::FusionIrPresched},
      {"fusion_ir", DebugDumpOption::FusionIr},
      {"fusion_ir_math", DebugDumpOption::FusionIrMath},
      {"global_zeroed_memory", DebugDumpOption::GlobalZeroedMemory},
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
      {"pre_segmenter_logging", DebugDumpOption::PreSegmenterLogging},
      {"predicate_elimination", DebugDumpOption::PredicateElimination},
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
      {"id_model", EnableOption::IdModel},
      {"kernel_db", EnableOption::KernelDb},
      {"kernel_profile", EnableOption::KernelProfile},
      {"memory_promotion", EnableOption::MemoryPromotion},
      {"reuse_zeroed_memory", EnableOption::ReuseZeroedMemory},
      {"static_fusion_count", EnableOption::StaticFusionCount},
      {"warn_register_spill", EnableOption::WarnRegisterSpill},
      {"io_to_lower_precision", EnableOption::IoToLowerPrecision},
  };

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
      {"matmul_expr_eval", DisableOption::MatmulExprEval},
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
      {"enable", ProfilerOption::Enable},
      {"enable.nocupti", ProfilerOption::EnableNocupti},
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
  return ProfilerOptionsGuard::getCurOptions().has(
             ProfilerOption::EnableNocupti) ||
      ProfilerOptionsGuard::getCurOptions().has(ProfilerOption::PrintNocupti);
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

const std::vector<std::string>& getDisableOptionArguments(
    ProfilerOption option) {
  return ProfilerOptionsGuard::getCurOptions().getArgs(option);
}

namespace {

// To define a new feature, create a new enum label in Features in options.h,
// then create a corresponding row in this macro. This macro runs fn for each
// feature; it should be a macro taking the following arguments:
//   short name, enum label, default enabled?, cuda cache key?, description
#define FOR_EACH_FEATURE(fn)                                                     \
  fn("compile_to_sass",                                                          \
     CompileToSass,                                                              \
     true,                                                                       \
     false,                                                                      \
     "Compile directly to SASS, bypassing PTX");                                 \
  fn("expr_simplify", ExprSimplify, true, true, "Simplify index expressions");   \
  /*TODO: What is fallback ?? */                                                 \
  fn("fallback", Fallback, true, true, "fallback???");                           \
  fn("fma", Fma, true, true, "Enable fused-multiply-add");                       \
  fn("grouped_grid_welford_outer_opt",                                           \
     GroupedGridWelfordOuterOpt,                                                 \
     true,                                                                       \
     true,                                                                       \
     "Use outer-optimized grouped grid Welford kernel");                         \
  fn("id_model", IdModel, false, true, "Use IterDomain graphs");                 \
  fn("index_hoist",                                                              \
     IndexHoist,                                                                 \
     true,                                                                       \
     true,                                                                       \
     "Hoist common subexpressions in loop nests");                               \
  fn("io_to_lower_precision",                                                    \
     IoToLowerPrecision,                                                         \
     false,                                                                      \
     true,                                                                       \
     "Enable castInputOutputToLowerPrecision");                                  \
  fn("kernel_db", KernelDb, false, false, "Use kernel database");                \
  fn("kernel_profile",                                                           \
     KernelProfile,                                                              \
     false,                                                                      \
     true,                                                                       \
     "Use intra-kernel performance profiling");                                  \
  fn("kernel_reuse",                                                             \
     KernelReuse,                                                                \
     true,                                                                       \
     false,                                                                      \
     "Re-use possibly suboptimal kernels when possible to avoid recompilation"); \
  fn("magic_zero",                                                               \
     MagicZero,                                                                  \
     true,                                                                       \
     true,                                                                       \
     "Use magic zero to prevent predicate elision for some unrolled loops");     \
  fn("matmul_expr_eval",                                                         \
     MatmulExprEval,                                                             \
     true,                                                                       \
     true,                                                                       \
     "Evaluate all matrix multiplications using cuBLAS");                        \
  fn("memory_promotion",                                                         \
     MemoryPromotion,                                                            \
     false,                                                                      \
     true,                                                                       \
     "Enable promotion of memory types for non-pointwise ops");                  \
  fn("nvtx", Nvtx, true, false, "Place NVTX ranges in compilation stages");      \
  fn("parallel_compile",                                                         \
     ParallelCompile,                                                            \
     true,                                                                       \
     false,                                                                      \
     "Use threading to compile fusion segments in parallel");                    \
  fn("parallel_serde",                                                           \
     ParallelSerde,                                                              \
     true,                                                                       \
     false,                                                                      \
     "Deserialize FusionExecutorCache in parallel");                             \
  fn("predicate_elimination",                                                    \
     PredicateElimination,                                                       \
     true,                                                                       \
     true,                                                                       \
     "Use predicate elimination");                                               \
  fn("reuse_mismatched_type_registers",                                          \
     ReuseMismatchedTypeRegisters,                                               \
     true,                                                                       \
     true,                                                                       \
     "Explicitly re-using registers in some cases when types don't match");      \
  fn("reuse_zeroed_memory",                                                      \
     ReuseZeroedMemory,                                                          \
     false,                                                                      \
     false,                                                                      \
     "[UNSAFE] Re-use zeroed memory for all grid synchronization");              \
  fn("static_fusion_count",                                                      \
     StaticFusionCount,                                                          \
     false,                                                                      \
     true,                                                                       \
     "Use single static count in kernel name");                                  \
  fn("var_name_remapping",                                                       \
     VarNameRemapping,                                                           \
     true,                                                                       \
     true,                                                                       \
     "Rename variables in cuda kernel to smaller numeric IDs");                  \
  fn("warn_register_spill",                                                      \
     WarnRegisterSpill,                                                          \
     true,                                                                       \
     false,                                                                      \
     "Warn at compilation if kernel spills registers to local memory");          \
  fn("welford_vectorization",                                                    \
     WelfordVectorization,                                                       \
     true,                                                                       \
     true,                                                                       \
     "Vectorize Welford ops");

const std::vector<std::string>& featureNames() {
  static std::vector<std::string> feature_names;
  static bool initialized = false;
  if (!initialized) {
    feature_names.resize(enumSize<Feature>(), "UNDEFINED_FEATURE_NAME");

#define SET_FEATURE_NAME(name, label, enabled, cache_key, desc) \
  feature_names.at(toUnderlying(Feature::label)) = name;
    FOR_EACH_FEATURE(SET_FEATURE_NAME);
#undef SET_FEATURE_NAME
    initialized = true;
  }
  return feature_names;
}

std::string featureName(const Feature& feat) {
  return featureNames().at(toUnderlying(feat));
}

void fillDefaultFeatures(FeatureSet* feats) {
  static std::bitset<enumSize<Feature>()> bitset;
  static std::unordered_map<Feature, std::vector<std::string>> all_args;
  static bool initialized = false;
  if (!initialized) {
#define ENABLE_DEFAULT_FEATURE(name, label, enabled, cache_key, desc) \
  if (enabled) {                                                      \
    bitset[toUnderlying(Feature::label)] = true;                      \
  }
    FOR_EACH_FEATURE(ENABLE_DEFAULT_FEATURE);
#undef ENABLE_DEFAULT_FEATURE
    const auto& named_features_map = nameToFeatureMap();
    const auto enabled = parseEnvOptions("ENABLE", named_features_map);
    const auto disabled = parseEnvOptions("DISABLE", named_features_map);
    for (const auto& [feature, feature_args] : enabled) {
      size_t idx = (size_t)toUnderlying(feature);
      NVF_CHECK(
          disabled.find(feature) == disabled.end(),
          "Contradiction in environment variables. Found ",
          featureNames()[idx],
          " in both $NVFUSER_ENABLED and $NVFUSER_DISABLED.");
      std::vector<std::string> args;
      args.reserve(feature_args.size());
      for (auto& arg : feature_args) {
        args.push_back(arg);
      }
      all_args.emplace(feature, args);
      bitset[idx] = true;
    }
    for (const auto& [feature, feature_args] : disabled) {
      size_t idx = (size_t)toUnderlying(feature);
      NVF_CHECK(
          enabled.find(feature) == enabled.end(),
          "Contradiction in environment variables. Found ",
          featureNames()[idx],
          " in both $NVFUSER_ENABLED and $NVFUSER_DISABLED.");
      bitset[idx] = false;
    }
    initialized = true;
  }
  feats->bitset() = bitset;
  feats->setArgs(all_args);
}

} // namespace

const std::unordered_map<std::string, Feature>& nameToFeatureMap() {
  static std::unordered_map<std::string, Feature> named_features_map;
  static bool initialized = false;
  if (!initialized) {
#define INSERT_NAMED_FEATURE(name, label, enabled, cache_key, desc) \
  named_features_map.emplace(name, Feature::label);
    FOR_EACH_FEATURE(INSERT_NAMED_FEATURE);
#undef INSERT_NAMED_FEATURE
    initialized = true;
  }
  return named_features_map;
}

std::string FeatureSet::toString() const {
  std::stringstream ss;
  ss << "FeatureSet[";
  bool first = true;
#define MAYBE_PRINT_FEATURE(name, label, enabled, cache_key, desc) \
  if (has(Feature::label) != enabled) {                            \
    if (!first) {                                                  \
      ss << ", ";                                                  \
    }                                                              \
    first = false;                                                 \
    ss << (enabled ? '-' : '+');                                   \
    ss << name;                                                    \
  }
  FOR_EACH_FEATURE(MAYBE_PRINT_FEATURE);
#undef MAYBE_PRINT_FEATURE
  ss << "]";
  return ss.str();
}

FeatureSet resetNonExecutionFeatures(const FeatureSet& features) {
  FeatureSet output = features;
#define RESET_FEATURE(name, label, enabled, cache_key, desc) \
  if (!cache_key) {                                          \
    output.set(Feature::label, enabled);                     \
  }
  FOR_EACH_FEATURE(RESET_FEATURE);
#undef RESET_FEATURE
  return output;
}

#undef FOR_EACH_FEATURE

std::optional<Feature> nameToFeature(std::string name) {
  const auto& named_features_map = nameToFeatureMap();
  auto it = named_features_map.find(name);
  if (it != named_features_map.end()) {
    return it->second;
  }
  return std::nullopt;
}

FeatureSet::FeatureSet() {
  fillDefaultFeatures(this);
}

const std::vector<std::string>& FeatureSet::getArgs(Feature feat) const {
  auto it = args_.find(feat);
  NVF_ERROR(
      "Arguments requested for feature ",
      featureNames()[(size_t)toUnderlying(feat)],
      " but none exist. FeatureSet::hasArgs() should be used to guard this call");
  return it->second;
}

FeatureSet parseFeatures(
    const std::vector<std::string>& enable_features,
    const std::vector<std::string>& disable_features) {
  FeatureSet features;
  // Track already manually enabled or disabled features
  auto enabled_bitset = features.bitset();
  auto disabled_bitset = features.bitset();
  enabled_bitset.reset();
  disabled_bitset.reset();
  const auto& map = nameToFeatureMap();
  const auto& processFeatures =
      [&map](
          const std::vector<std::string>& feature_names,
          std::bitset<enumSize<Feature>()>& bitset,
          auto enable_or_disable_fn) {
        for (const std::string& name : feature_names) {
          auto it = map.find(name);
          if (it == map.end()) {
            std::vector<std::string> names;
            names.reserve(map.size());
            for (const auto& [k, v] : map) {
              names.push_back(k);
            }
            std::sort(names.begin(), names.end());
            std::stringstream ss;
            bool first = true;
            for (const auto& n : names) {
              if (!first) {
                ss << ", ";
              }
              first = false;
              ss << n;
            }
            NVF_CHECK(
                false,
                "Unknown feature '",
                name,
                "'. Available features: ",
                ss.str());
          }
          Feature f = it->second;
          enable_or_disable_fn(f);
          bitset.set(toUnderlying(f), true);
        }
      };
  processFeatures(enable_features, enabled_bitset, [&features](Feature f) {
    features.insert(f);
  });
  processFeatures(disable_features, disabled_bitset, [&features](Feature f) {
    features.erase(f);
  });
  // Check that we didn't manually both insert and erase a feature
  for (const size_t i : c10::irange(enumSize<Feature>())) {
    NVF_CHECK(
        !enabled_bitset.test(i) || !disabled_bitset.test(i),
        "Feature ambiguously both enabled and disabled");
  }

  return features;
}

std::ostream& operator<<(std::ostream& os, Feature f) {
  os << featureNames().at(toUnderlying(f));
  return os;
}

std::ostream& operator<<(std::ostream& os, FeatureSet feats) {
  os << feats.toString();
  return os;
}

} // namespace nvfuser

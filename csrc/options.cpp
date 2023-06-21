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

} // namespace

EnableOptions::EnableOptions() {
  const std::unordered_map<std::string, EnableOption> available_options = {
      {"complex", EnableOption::Complex},
      {"conv_decomposition", EnableOption::ConvDecomposition},
      {"graph_op_fusion", EnableOption::GraphOp},
      {"kernel_db", EnableOption::KernelDb},
      {"kernel_profile", EnableOption::KernelProfile},
      {"linear_decomposition", EnableOption::LinearDecomposition},
      {"warn_register_spill", EnableOption::WarnRegisterSpill}};

  options_ = parseEnvOptions("PYTORCH_NVFUSER_ENABLE", available_options);
}

DisableOptions::DisableOptions() {
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
      {"predicate_elimination", DisableOption::PredicateElimination},
      {"var_name_remapping", DisableOption::VarNameRemapping},
      {"welford_vectorization", DisableOption::WelfordVectorization}};

  auto options = parseEnvOptions("PYTORCH_NVFUSER_DISABLE", available_options);

  if (options.count(DisableOption::Fma)) {
    TORCH_WARN(
        "fmad is disabled for nvrtc, which could negatively affect performance. Try removing `fma` from env variable PYTORCH_NVFUSER_DISABLE for optimal performance.");
  }
}

namespace {

// This may need to be thread local, or its modifications may need to
// be protected by mutual exclusion for thread safety. At this
// moment, the correctness of modifying option values has to be
// guaranteed by the modifying code.
EnableOptions active_enable_options;

DisableOptions active_disable_options;

// thread_local variable used only for debugging/testing
thread_local bool overwrite_disable_fma = false;

} // namespace

EnableOptionsGuard::EnableOptionsGuard()
    : prev_options_(active_enable_options) {}

EnableOptionsGuard::~EnableOptionsGuard() {
  active_enable_options = prev_options_;
}

EnableOptions& EnableOptionsGuard::getCurOptions() {
  return active_enable_options;
}

bool isOptionEnabled(EnableOption option) {
  return EnableOptionsGuard::getCurOptions().has(option);
}

const std::vector<std::string>& getEnableOptionArguments(
    EnableOption option) {
  return EnableOptionsGuard::getCurOptions().getArgs(option);
}

DisableOptionsGuard::DisableOptionsGuard()
    : prev_options_(active_disable_options) {}

DisableOptionsGuard::~DisableOptionsGuard() {
  active_disable_options = prev_options_;
}

DisableOptions& DisableOptionsGuard::getCurOptions() {
  return active_disable_options;
}

bool isOptionDisabled(DisableOption option) {
  if (option == DisableOption::Fma && overwrite_disable_fma) {
    return true;
  }
  return DisableOptionsGuard::getCurOptions().has(option);
}

const std::vector<std::string>& getDisableOptionArguments(
    DisableOption option) {
  return DisableOptionsGuard::getCurOptions().getArgs(option);
}

ThreadLocalFmaDisableOverwrite::ThreadLocalFmaDisableOverwrite(bool flag)
    : old_flag_{overwrite_disable_fma} {
  overwrite_disable_fma = flag;
}

ThreadLocalFmaDisableOverwrite::~ThreadLocalFmaDisableOverwrite() {
  overwrite_disable_fma = old_flag_;
}



} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace nvfuser {

//! Types of features to enable
//!
//! These can be set through the `PYTORCH_NVFUSER_ENABLE` environment variable
//!
enum class EnableOption {
  Complex, //! Enable complex support on python
  ConvDecomposition, //! Enable conv-bias decomposition
  GraphOp, //! Enable graphOps(index_select/gather/scatter)
  KernelDb, //! Enable Kernel Database
  KernelProfile, //! Enable intra-kernel performance profiling
  LinearDecomposition, //! Enable linear-bias decomposition
  WarnRegisterSpill, //! Enable warnings of register spill
  EndOfOption //! Placeholder for counting the number of elements
};

//! Types of features to disable
//!
//! These can be set through the `PYTORCH_NVFUSER_DISABLE` environment variable
//!
enum class DisableOption {
  CompileToSass, //! Disable direct compilation to sass so the ptx can be
                 //! examined
  ExprSimplify, //! Disable expression simplifier
  Fallback, //! Disable fallback
  Fma, //! Disable FMA instructions
  GroupedGridWelfordOuterOpt, //! Disable use of outer-optimized
                              //! grouped grid welford kernel
  IndexHoist, //! Disable index hoisting
  MagicZero, //! Disable nvfuser_zero
  Nvtx, //! Disable NVTX instrumentation
  PredicateElimination, //! Disable predicate elimination
  VarNameRemapping, //! Disable variable name remapping
  WelfordVectorization, //! Disable vectorizaton of Welford ops
  EndOfOption //! Placeholder for counting the number of elements
};

//! The base template class for the options such as EnableOption
template <typename OptionEnum>
class Options {
 public:
  Options() = default;

  bool has(OptionEnum option) const {
    return options_.count(option);
  }

  const std::vector<std::string>& getArgs(OptionEnum option) const {
    TORCH_INTERNAL_ASSERT(has(option), "Option not set");
    return options_.at(option);
  }

  void set(OptionEnum option_type, std::vector<std::string> option = {}) {
    options_[option_type] = option;
  }

  void unset(OptionEnum option_type) {
    options_.erase(option_type);
  }

  static std::unordered_map<OptionEnum, std::vector<std::string>>
  getOptionsFromEnv();

 protected:
  std::unordered_map<OptionEnum, std::vector<std::string>> options_;
};

template <>
std::unordered_map<EnableOption, std::vector<std::string>> Options<
    EnableOption>::getOptionsFromEnv();

using EnableOptions = Options<EnableOption>;

TORCH_CUDA_CU_API bool isOptionEnabled(EnableOption option);

TORCH_CUDA_CU_API const std::vector<std::string>& getEnableOptionArguments(
    EnableOption option);

template <>
std::unordered_map<DisableOption, std::vector<std::string>> Options<
    DisableOption>::getOptionsFromEnv();

using DisableOptions = Options<DisableOption>;

TORCH_CUDA_CU_API bool isOptionDisabled(DisableOption option);

TORCH_CUDA_CU_API const std::vector<std::string>& getDisableOptionArguments(
    DisableOption option);

//! Utility class to temporarily overrride the Enable options,
//! including those provided by the environment variable
template <typename OptionEnum>
class TORCH_CUDA_CU_API OptionsGuard {
 public:
  OptionsGuard() : prev_options_(getCurOptions()) {}

  ~OptionsGuard() {
    getCurOptions() = prev_options_;
  }

  static Options<OptionEnum>& getCurOptions();

 private:
  Options<OptionEnum> prev_options_;
};

template <>
Options<EnableOption>& OptionsGuard<EnableOption>::getCurOptions();

template <>
Options<DisableOption>& OptionsGuard<DisableOption>::getCurOptions();

using EnableOptionsGuard = OptionsGuard<EnableOption>;

using DisableOptionsGuard = OptionsGuard<DisableOption>;

} // namespace nvfuser

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
  Fallback, //! Disable fallback
  Fma, //! Disable FMA instructions
  GroupedGridWelfordOuterOpt, //! Disable use of outer-optimized
                              //! grouped grid welford kernel
  IndexHoist, //! Disable index hoisting
  ExprSimplify, //! Disable expression simplifier
  Nvtx, //! Disable NVTX instrumentation
  PredicateElimination, //! Disable predicate elimination
  WelfordVectorization, //! Disable vectorizaton of Welford ops
  MagicZero, //! Disable nvfuser_zero
  VarNameRemapping, //! Disable variable name remapping
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

 protected:
  std::unordered_map<OptionEnum, std::vector<std::string>> options_;
};

class TORCH_CUDA_CU_API EnableOptions : public Options<EnableOption> {
 public:
  EnableOptions();
};

class TORCH_CUDA_CU_API DisableOptions : public Options<DisableOption> {
 public:
  DisableOptions();
};

// used only for testing/debugging
class TORCH_CUDA_CU_API ThreadLocalFmaDisableOverwrite {
 public:
  ThreadLocalFmaDisableOverwrite(bool flag = true);
  ~ThreadLocalFmaDisableOverwrite();

 private:
  bool old_flag_;
};

//! Utility class to temporarily overrride the Enable options,
//! including those provided by the environment variable
class TORCH_CUDA_CU_API EnableOptionsGuard {
 public:
  EnableOptionsGuard();

  ~EnableOptionsGuard();

  static EnableOptions& getCurOptions();

 private:
  EnableOptions prev_options_;
};

TORCH_CUDA_CU_API bool isOptionEnabled(EnableOption option);

TORCH_CUDA_CU_API const std::vector<std::string>& getEnableOptionArguments(
    EnableOption option);

//! Utility class to temporarily overrride the Disable options,
//! including those provided by the environment variable
class TORCH_CUDA_CU_API DisableOptionsGuard {
 public:
  DisableOptionsGuard();

  ~DisableOptionsGuard();

  static DisableOptions& getCurOptions();

 private:
  DisableOptions prev_options_;
};

TORCH_CUDA_CU_API bool isOptionDisabled(DisableOption option);

TORCH_CUDA_CU_API const std::vector<std::string>& getDisableOptionArguments(
    DisableOption option);

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

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

template <typename OptionEnum>
class Options {
 public:
  Options() = default;

  // Options(const Options& other_opts): options_(other_opts.options_) {}

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

} // namespace nvfuser

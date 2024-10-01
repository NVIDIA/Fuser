// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <mma_type.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/matmul_heuristic_plugin_api.h>
#include <scheduler/mma_utils.h>

#include <memory>
#include <optional>

namespace nvfuser {

namespace matmul_heuristic_plugin {

//! Returns true if KernelConfigFactoryGuard is active indicating an imitated
//! plugin, or if a shared library plugin has been provided using the
//! environment variable NVFUSER_MATMUL_HEURISTIC_PLUGIN.
bool hasPlugin();

//! If there is no user-defined plugin (see hasPlugin()) we return false.
//! Otherwise, we use the plugin to modify the heuristic parameters in place. M,
//! N, K, layout (inner allocated dimension roles of each operand), and
//! precision must be provided. For convenience, we use `roles_map` to build the
//! precision string.
bool updateMatmulParams(
    MatmulParams* params,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t batch_size,
    const mma_utils::MatmulOperandInnerDims& inner_dims,
    const mma_utils::TensorRolesMap& tensor_roles);

//! Defines the type of the "makeConfig" symbol
using KernelConfigFactory = std::function<std::unique_ptr<KernelConfig>()>;

//! This function can be used to imitate a plugin. To do so, subclass
//! KernelConfig, implementing a custom `configure` method, then create a guard
//! object like this:
//!
//!   KernelConfigFactoryGuard kfg([]() { return (KernelConfig*)(new
//!   MyKernelConfig);});
//!
//! When kfg passes out of scope, the config factory will be reset to its prior
//! value.
class KernelConfigFactoryGuard {
 public:
  explicit KernelConfigFactoryGuard(KernelConfigFactory func);
  ~KernelConfigFactoryGuard();

 private:
  KernelConfigFactory prev_factory_;
  bool prev_factory_modified_;
};

} // namespace matmul_heuristic_plugin

} // namespace nvfuser

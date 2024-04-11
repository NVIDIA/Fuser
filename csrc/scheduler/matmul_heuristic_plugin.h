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

#include <optional>

namespace nvfuser {

namespace matmul_heuristic_plugin {

//! Try to load plugin whose location is provided by the environment variable
//! NVFUSER_MATMUL_HEURISTIC_PLUGIN and return whether or not we succeed.
bool hasPlugin();

//! If loading the plugin fails (see hasPlugin()) we return false. Otherwise, we
//! use the plugin to modify the heuristic parameters in place. M, N, K, layout,
//! and precision must also provided.
bool updateMatmulParams(
    MatmulParams& params,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t batch_size,
    MmaLayout layout,
    const mma_utils::RolesMap& roles_map);

//! Defines ConfigFactoryFuncPtr as type of the "makeConfig" symbol
typedef KernelConfig* (*KernelConfigFactoryPointer)();

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
  explicit KernelConfigFactoryGuard(KernelConfigFactoryPointer func);
  ~KernelConfigFactoryGuard();

 private:
  KernelConfigFactoryPointer prev_factory_;

  static thread_local KernelConfigFactoryPointer active_factory_;
};

} // namespace matmul_heuristic_plugin

} // namespace nvfuser

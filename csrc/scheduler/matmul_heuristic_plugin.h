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
#include <scheduler/mma_utils.h>

#include <optional>

namespace nvfuser {

namespace matmul_heuristic_plugin {

//! This is intended as a minimal interface for enabling matmul heuristics.
//! In order to plug in your own custom heuristic, create a dynamic library
//! that exports a function whose signature matches HeuristicFunc (see below).
//! If that library is located at /path/to/libfoo.so you can set
//! NVFUSER_MATMUL_HEURISTIC_PLUGIN=/path/to/libfoo.so to use the plugin to
//! determine matmul parameters automatically.

//! This is the information available to the plugin to determine the kernel configuration
struct ProblemDescription {
  struct Shapes {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batch_size;
    // layout is in row-major and takes values 0 thru 3 in order NN NT TN TT
    uint8_t layout;
  } shapes;
  const char* precision; // e.g. HSH, TST, HSS, etc.
};

//! This is the return type of a HeuristicFunc (defined below) implemented in a
//! plugin. This is used to set values in MatmulParams
struct KernelConfig {
  uint16_t cta_tile[3];
  uint16_t warp_tile[3];
  uint16_t instr_tile[3];
  uint16_t splitk_factor;
  uint8_t load_stages;
  uint8_t grid_swizzle;
  uint8_t cta_order;
  // CGA configuration describing cluster X and Y dimensions. This is currently unused.
  uint8_t cga_config[2];
};

//! Utility to standardize conversion of MmaLayout to uint8_t
uint8_t layoutToByte(MmaLayout layout);

//! Defines HeuristicFun as type of the "getConfig" symbol
typedef KernelConfig (*HeuristicFunc)(const ProblemDescription&);

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
    MmaLayout layout,
    const mma_utils::RolesMap& roles_map);

} // namespace matmul_heuristic_plugin

} // namespace nvfuser


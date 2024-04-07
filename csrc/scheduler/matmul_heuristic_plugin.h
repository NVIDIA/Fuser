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

//! This is the information available to the plugin to determine the kernel
//! configuration. For API stability, these should not be accessed directly when
//! implementing plugins, but rather through these accessors.
//! matmul_heuristic_plugin_api.h
struct NVF_API ProblemDescription;
NVF_API uint32_t getProblemM(const ProblemDescription* problem);
NVF_API uint32_t getProblemN(const ProblemDescription* problem);
NVF_API uint32_t getProblemK(const ProblemDescription* problem);
NVF_API uint32_t getProblemBatchSize(const ProblemDescription* problem);
NVF_API uint8_t getProblemLayout(const ProblemDescription* problem);
NVF_API const char* getProblemPrecision(const ProblemDescription* problem);

//! This is the return type of a HeuristicFunc (defined below) implemented in a
//! plugin. This is used to set values in MatmulParams. For API stability, these
//! should not be accessed directly when implementing plugins, but rather
//! through the accessors defined in matmul_heuristic_plugin_api.h
struct NVF_API KernelConfig;
NVF_API void setCtaTile(
    KernelConfig* config,
    uint16_t m,
    uint16_t n,
    uint16_t k);
NVF_API void setWarpTile(
    KernelConfig* config,
    uint16_t m,
    uint16_t n,
    uint16_t k);
NVF_API void setInstructionTile(
    KernelConfig* config,
    uint16_t m,
    uint16_t n,
    uint16_t k);
NVF_API void setSplitKFactor(KernelConfig* config, uint16_t f);
NVF_API void setLoadStages(KernelConfig* config, uint8_t s);
NVF_API void setGridSwizzleFactor(KernelConfig* config, uint8_t g);
NVF_API void setCtaOrder(KernelConfig* config, uint8_t o);

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

} // namespace matmul_heuristic_plugin

} // namespace nvfuser

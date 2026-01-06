// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace nvfuser::python {

void initNvFuserPythonBindings(nb::module_& nvfuser);

// Add bindings for Fusion IR
void bindFusionIr(nb::module_& nvfuser);

// Add bindings for Internal Fusion and Kernel IR
void bindInternalIr(nb::module_& nvfuser);

// Add bindings for Enums
void bindEnums(nb::module_& nvfuser);

// Add bindings for Fusion and FusionExecutorCache
void bindRuntime(nb::module_& nvfuser);

// Add bindings for LaunchParams, CompileParams, and HeuristicParams
void bindHeuristicParams(nb::module_& nvfuser);

// Add bindings for CPP Fusion Operations
void bindOperations(nb::module_& nvfuser);

// Add bindings for CPP Schedule Operators
void bindScheduleOperators(nb::module_& nvfuser);

// Add bindings for MultiDevice features
void bindMultiDevice(nb::module_& nvfuser);

// Add bindings for IdModel and ValGraph
void bindIdModel(nb::module_& nvfuser);

// Add bindings for Fusion Profiler
void bindProfile(nb::module_& nvfuser);

// Add bindings for LRU Cache
void bindLRUCache(nb::module_& nvfuser);

// Translate a CPP Fusion to a bindings python function
std::string translateFusion(Fusion* f);

#ifdef NVFUSER_ENABLE_CUTLASS
// Add bindings for Cutlass GEMM Operations
void bindCutlass(nb::module_& nvfuser);
#endif

} // namespace nvfuser::python

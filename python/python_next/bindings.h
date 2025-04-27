// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion_definition.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace nvfuser::python {

void initNvFuserPythonBindings(PyObject* module);

// Add bindings for Fusion IR
void bindFusionIr(py::module& nvfuser);

// Add direct bindings for Fusion and FusionExecutorCache
void bindRuntime(py::module& nvfuser);

// Add bindings for FusionDefinition
void bindFusionDefinition(py::module& nvfuser);

// Add bindings for CPP Fusion Operations
void bindOperations(py::class_<FusionDefinition>& fusion_def);

// Add bindings for CPP Schedule Operators
void bindScheduleOperators(py::class_<FusionDefinition>& fusion_def);

} // namespace nvfuser::python

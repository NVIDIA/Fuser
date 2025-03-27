// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion_definition.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace direct_bindings {

void initNvFuserDirectBindings(PyObject* module);

// Add direct bindings for Fusion IR
void bindFusionIr(py::module& direct_bindings);

// Add direct bindings for Fusion and FusionExecutorCache
void bindRuntime(py::module& direct_bindings);

// Add direct bindings for FusionDefinition
void bindFusionDefinition(py::module& direct_bindings);

// Add direct bindings for CPP Fusion Operations
void bindDirectOperations(py::class_<DirectFusionDefinition>& fusion_def);

// Add direct bindings for CPP Schedule Operators
void bindDirectScheduleOperators(
    py::class_<DirectFusionDefinition>& fusion_def);

} // namespace direct_bindings

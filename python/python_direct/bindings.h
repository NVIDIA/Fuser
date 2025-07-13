// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace nvfuser::python {

void initNvFuserPythonBindings(PyObject* module);

// Add bindings for Fusion IR
void bindFusionIr(py::module& nvfuser);

// Add bindings for Enums
void bindEnums(py::module& nvfuser);

// Add bindings for Fusion and FusionExecutorCache
void bindRuntime(py::module& nvfuser);

// Add bindings for CPP Fusion Operations
void bindOperations(py::module& nvfuser);

// Add bindings for MultiDevice features
void bindMultiDevice(py::module& nvfuser);

// Translate a CPP Fusion to a bindings python function
std::string translateFusion(Fusion* f);

#ifdef NVFUSER_ENABLE_CUTLASS
// Add bindings for Cutlass GEMM Operations
void bindCutlass(py::module& nvfuser);
#endif

} // namespace nvfuser::python

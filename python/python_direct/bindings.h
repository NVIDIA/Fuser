// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace nvfuser::python {

void initNvFuserPythonBindings(PyObject* module);

// Add bindings for Fusion IR
void bindFusionIr(py::module& nvfuser);

// Add bindings for Enums
void bindEnums(py::module& nvfuser);

} // namespace nvfuser::python

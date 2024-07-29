// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

#include <visibility.h>

namespace nvfuser::python_frontend {
NVF_API void initNvFuserPythonBindings(PyObject* module);

NVF_API void cleanup();
} // namespace nvfuser::python_frontend

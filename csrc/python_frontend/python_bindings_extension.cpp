// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <python_frontend/python_bindings.h>
#include <torch/extension.h>

PYBIND11_MODULE(EXTENSION_NAME, m) {
  m.doc() = "nvfuser C API python binding"; // optional module docstring

  nvfuser::python_frontend::initNvFuserPythonBindings(m.ptr());

  auto cleanup = []() -> void { nvfuser::python_frontend::cleanup(); };
  m.add_object("_cleanup", py::capsule(cleanup));
}

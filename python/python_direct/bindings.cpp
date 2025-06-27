// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <bindings.h>

namespace nvfuser::python {

void initNvFuserPythonBindings(PyObject* module) {
  auto nvfuser = py::handle(module).cast<py::module>();
  bindEnums(nvfuser);
  bindFusionIr(nvfuser);
  bindRuntime(nvfuser);
  bindOperations(nvfuser);
  bindMultiDevice(nvfuser);
  nvfuser.def("translate_fusion", &translateFusion);
}

} // namespace nvfuser::python

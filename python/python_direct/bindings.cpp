// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <bindings.h>
#include <multidevice/communicator.h>

namespace nvfuser::python {

void initNvFuserPythonBindings(PyObject* module) {
  auto nvfuser = py::handle(module).cast<py::module>();
  bindEnums(nvfuser);
  bindFusionIr(nvfuser);
  bindRuntime(nvfuser);
  bindOperations(nvfuser);
  bindMultiDevice(nvfuser);
  nvfuser.def("translate_fusion", &translateFusion);
#ifdef NVFUSER_ENABLE_CUTLASS
  bindCutlass(nvfuser);
#endif

  auto cleanup = []() -> void { Communicator::getInstance().cleanup(); };
  nvfuser.add_object("_cleanup", py::capsule(cleanup));
}

} // namespace nvfuser::python

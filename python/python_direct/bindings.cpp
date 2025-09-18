// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <bindings.h>
#include <multidevice/communicator.h>
#include <python_common/python_utils.h>

namespace nvfuser::python {

void initNvFuserPythonBindings(PyObject* module) {
  auto nvfuser = py::handle(module).cast<py::module>();
  bindEnums(nvfuser);
  bindHeuristicParams(nvfuser);
  bindFusionIr(nvfuser);
  bindRuntime(nvfuser);
  bindOperations(nvfuser);
  bindMultiDevice(nvfuser);
  nvfuser.def(
      "translate_fusion",
      &translateFusion,
      py::arg("fusion"),
      R"(Translate a Fusion to a Python string.)");
  nvfuser.def(
      "compute_tensor_descriptor",
      &computeTensorDescriptor,
      py::arg("sizes"),
      py::arg("strides"),
      R"(
    Compute the tensor descriptor for a given shape and stride.
  )");
#ifdef NVFUSER_ENABLE_CUTLASS
  bindCutlass(nvfuser);
#endif

  auto cleanup = []() -> void {
    auto& c = Communicator::getInstance();
    // In the transition period, both nvfuser and nvfuser_direct may be
    // imported and share one Communicator singleton.  Without the is_available
    // check, each tries to call Communicator::cleanup() at process exit.
    if (c.is_available()) {
      c.cleanup();
    }
  };
  nvfuser.add_object("_cleanup", py::capsule(cleanup));
}

} // namespace nvfuser::python

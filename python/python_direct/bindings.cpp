// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <bindings.h>
#include <direct_utils.h>
#include <multidevice/communicator.h>
#include <python_common/python_utils.h>
#include <validator_utils.h>

namespace nvfuser::python {

void initNvFuserPythonBindings(PyObject* module) {
  auto nvfuser = py::handle(module).cast<py::module>();
  bindEnums(nvfuser);
  bindHeuristicParams(nvfuser);
  bindFusionIr(nvfuser);
  bindInternalIr(nvfuser);
  bindRuntime(nvfuser);
  bindOperations(nvfuser);
  bindScheduleOperators(nvfuser);
  bindMultiDevice(nvfuser);
  bindIdModel(nvfuser);
  bindProfile(nvfuser);
  bindLRUCache(nvfuser);
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
  nvfuser.def(
      "validate_with_auto_inferred_outputs",
      [](Fusion& fusion,
         const py::iterable& fusion_outputs,
         const py::iterable& args) {
        return testValidate(
            &fusion, from_pyiterable(fusion_outputs), from_pyiterable(args));
      },
      py::arg("fusion"),
      py::arg("fusion_outputs"),
      py::arg("args"),
      R"(
Validate the fusion outputs with auto inferred outputs.

Parameters
----------
fusion : Fusion
    The fusion to validate.
fusion_outputs : iterable
    The fusion outputs to validate.
args : iterable
    The arguments to validate the fusion outputs with.

Returns
-------
None
)");
  nvfuser.def(
      "get_val_tolerances",
      [](Fusion& fusion, const py::iterable& args) {
        return getValTolerances(&fusion, from_pyiterable(args));
      },
      py::arg("fusion"),
      py::arg("args"),
      R"(
Get the validation tolerances for the fusion.

Parameters
----------
fusion : Fusion
    The fusion to get the validation tolerances for.
args : iterable
    The arguments to get the validation tolerances for.

Returns
-------
list of tuple of float
    The validation tolerances for the fusion.
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

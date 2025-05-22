// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <python_utils.h>

#include <fusion.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser::python {

// direct_bindings/runtime.cpp contains Fusion and FusionExecutorCache.
// TODO IrContainer and Kernel would be implemented here.

namespace {

void bindFusion(py::module& nvfuser) {
  py::class_<FusionGuard>(nvfuser, "FusionGuard")
      .def(
          py::init<Fusion*>(),
          py::arg("fusion"),
          R"(
Create a new FusionGuard to manage the active fusion context.

A FusionGuard is a RAII-style guard that sets the active fusion context for the current scope.
When the guard is created, it sets the provided fusion as the active fusion.
When the guard is destroyed, it restores the previous fusion context.

Parameters
----------
fusion : Fusion
    The fusion to set as the active fusion context.
)");

  // NOTE: manage, get_managed, get_managed_safe, stop_managing, has_managed are
  // template functions in the Fusion class. Templates do not exist in the
  // python language. To bind these functions, you must instantiate a full
  // (explicit) template specialization.
  py::class_<Fusion, std::unique_ptr<Fusion, py::nodelete>>(nvfuser, "Fusion")
      .def(py::init<>(), R"(
Create a new Fusion.

A Fusion represents a computation graph that can be compiled and executed on CUDA devices.
It manages the IR nodes, inputs/outputs, and transformations needed to generate efficient CUDA kernels.

Examples
--------
>>> fusion = Fusion()
>>> # Add inputs
>>> t0 = fusion.add_input(...)
>>> # Define computations
>>> t1 = ops.add(t0, t0)
>>> # Register outputs
>>> fusion.add_output(t1)
)")
      .def("add_input", &Fusion::addInput, py::arg("input"), R"(
Register a value as an input to the fusion.

Parameters
----------
input : Val
    The value to register as an input.

Returns
-------
Val
    The registered input value.

Notes
-----
- The input must not already be registered as an input.
- The input must not have a definition within the fusion.
)")
      .def("add_output", &Fusion::addOutput, py::arg("output"), R"(
Register a value as an output of the fusion.

Parameters
----------
output : Val
    The value to register as an output.

Returns
-------
Val
    The registered output value.

Notes
-----
- The output must be defined within the fusion or be an input.
- The same value can be registered as an output multiple times.
)")
      .def(
          "print_math",
          [](Fusion& f, bool from_outputs_only) {
            // Send debug messages to stringstream
            std::stringstream ss;
            DebugStreamGuard dsg(ss);

            f.printMath(from_outputs_only);
            return ss.str();
          },
          py::arg("from_outputs_only") = true,
          R"(
Print arithmetic expressions in the fusion.

Parameters
----------
from_outputs_only : bool, optional
    If True, only print expressions reachable from outputs.
    If False, print all expressions.
    Default is True.

Returns
-------
str
    The fusion intermediate representation (IR) as a string.
)");
}

} // namespace

void bindRuntime(py::module& nvfuser) {
  bindFusion(nvfuser);
}

} // namespace nvfuser::python

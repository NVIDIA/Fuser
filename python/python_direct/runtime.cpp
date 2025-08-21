// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <direct_utils.h>
#include <python_common/distributed_tensor.h>
#include <python_utils.h>

#include <fusion.h>
#include <options.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>
#include <validator_utils.h>

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
  py::class_<Fusion>(nvfuser, "Fusion")
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
      .def(
          "inputs",
          &Fusion::inputs,
          py::return_value_policy::reference,
          R"(
Get the inputs of the fusion.

Returns
-------
list of Val
    The inputs of the fusion.
)")
      .def(
          "outputs",
          &Fusion::outputs,
          py::return_value_policy::reference,
          R"(
Get the outputs of the fusion.

Returns
-------
list of Val
    The outputs of the fusion.
)")
      .def(
          "vals",
          [](Fusion& self) { return self.vals(); },
          py::return_value_policy::reference,
          R"(
Return all Vals registered in the fusion.

Returns
-------
list of Val
    The Vals registered in the fusion.
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
None

Notes
-----
- The output must be defined within the fusion or be an input.
- The same value can be registered as an output multiple times.
)")
      .def(
          "add_output",
          [](Fusion& self, Val* output, Val* alias_input) {
            self.aliasOutputToInput(
                output, alias_input, AllocationType::ReuseBuffer);
          },
          py::arg("output"),
          py::arg("alias_input"),
          R"(
Alias an output to an input.

Parameters
----------
output : Val
    The value to alias as an output.
alias_input : Val
    The value to alias the output to.

Returns
-------
None

Notes
-----
- This output is not returned from the fusion.
- The same value can be registered as a regular output, so it is returned from
the fusion.
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

void bindFusionExecutorCache(py::module& nvfuser) {
  py::class_<FusionExecutorCache>(nvfuser, "FusionExecutorCache")
      .def(
          py::init([](const Fusion* fusion,
                      int64_t fusion_id,
                      bool auto_schedule) {
            // Make a copy of the fusion for FusionExecutorCache to own.
            return new FusionExecutorCache(
                std::make_unique<Fusion>(*fusion), fusion_id, auto_schedule);
          }),
          py::arg("fusion"),
          py::arg("fusion_id") = 0,
          py::arg("auto_schedule") = true,
          R"(
Create a new FusionExecutorCache.

The cache automatically handles compilation and execution of the fusion for
different input configurations.

Parameters
----------
fusion : Fusion
    The fusion to be executed.
    The FusionExecutorCache takes ownership of this pointer.
fusion_id : int, optional
    A unique identifier for this fusion. Default is 0.
auto_schedule : bool, optional
    Whether to automatically schedule the fusion.
    If False, the fusion must be manually scheduled.
    Default is True.

Examples
--------
>>> fusion = Fusion()
>>> # ... define fusion operations ...
>>> executor_cache = FusionExecutorCache(fusion)
>>> outputs = executor_cache.execute([input1, input2])
)")
      .def(
          "execute",
          [](FusionExecutorCache& self,
             const py::iterable& iter,
             std::optional<int64_t> device,
             std::vector<std::string> _enable_options,
             std::vector<std::string> _disable_options) {
            KernelArgumentHolder args = from_pyiterable(iter, device);

            EnableOptionsGuard enable_opt_guard;
            for (const auto& _enable_option : _enable_options) {
              std::optional<EnableOption> opt =
                  stringToEnableOption(_enable_option);
              NVF_CHECK(
                  opt.has_value(),
                  "Unrecognized enable_option: ",
                  _enable_option);
              EnableOptionsGuard::getCurOptions().set(opt.value());
            }

            DisableOptionsGuard disable_opt_guard;
            for (const auto& _disable_option : _disable_options) {
              std::optional<DisableOption> opt =
                  stringToDisableOption(_disable_option);
              NVF_CHECK(
                  opt.has_value(),
                  "Unrecognized disable_option: ",
                  _disable_option);
              DisableOptionsGuard::getCurOptions().set(opt.value());
            }

            KernelArgumentHolder outputs = self.runFusionWithInputs(
                args, std::nullopt, args.getDeviceIndex());

            return to_tensor_vector(outputs);
          },
          py::arg("inputs"),
          py::kw_only(),
          py::arg("device") = py::none(),
          py::arg("_enable_options") = py::list(),
          py::arg("_disable_options") = py::list(),
          R"(
Execute the fusion with the given inputs.

Parameters
----------
inputs : iterable
    An iterable of input tensors or values.
    All tensor inputs must be on the same device.
    Cpu scalar tensor can interoperate with gpu tensors.
device : int, optional
    The device index to execute the fusion on.
    It must be a non-negative integer less than 256.
    If None, uses the device of the input tensors.
    Default is None.
_enable_options : list of str, optional
    A list of enable options.
    Default is None.
_disable_options : list of str, optional
    A list of disable options.
    Default is None.

Returns
-------
list of torch.Tensor
    The output tensors produced by the fusion.
)")
      .def(
          "validate_with_auto_inferred_outputs",
          [](FusionExecutorCache& self,
             const py::iterable& fusion_outputs,
             const py::iterable& args) {
            return testValidate(
                self.fusion(),
                from_pyiterable(fusion_outputs),
                from_pyiterable(args));
          },
          py::arg("fusion_outputs"),
          py::arg("args"),
          R"(
Validate the fusion outputs with auto inferred outputs.

Parameters
----------
fusion_outputs : iterable
    The fusion outputs to validate.
args : iterable
    The arguments to validate the fusion outputs with.

Returns
-------
None
)")
      .def(
          "get_val_tolerances",
          [](FusionExecutorCache& self, const py::iterable& args) {
            return getValConstants(self.fusion(), from_pyiterable(args));
          },
          py::arg("args"),
          R"(
Get the validation tolerances for the fusion.

Parameters
----------
args : iterable
    The arguments to get the validation tolerances for.

Returns
-------
list of tuple of float
    The validation tolerances for the fusion.
)")
      .def(
          "is_compiled",
          [](FusionExecutorCache& self,
             const py::iterable& iter,
             std::optional<int64_t> device) {
            return self.isCompiled(from_pyiterable(iter, device));
          },
          py::arg("inputs"),
          py::arg("device") = 0,
          R"(
Check if a compiled kernel exists for the given input configuration.

Parameters
----------
inputs : iterable
    An iterable of input tensors or values to check.
device : int, optional
    The target device index. Default is 0.

Returns
-------
bool
    True if a compiled kernel exists for the input configuration.
)")
      .def(
          "fusion",
          static_cast<Fusion* (FusionExecutorCache::*)()>(
              &FusionExecutorCache::fusion),
          py::return_value_policy::reference,
          R"(
Get the underlying fusion object.

Returns
-------
Fusion
    The fusion object being executed by this cache.
)")
      .def(
          "get_cuda_kernel",
          [](FusionExecutorCache& self,
             const py::iterable& iter,
             bool intrinsic_code,
             std::optional<int64_t> device) {
            return self.getCodeFor(
                from_pyiterable(iter, device), intrinsic_code);
          },
          py::arg("inputs"),
          py::arg("intrinsic_code") = false,
          py::arg("device") = 0,
          R"(
Get the CUDA kernel code for the given input configuration.

Parameters
----------
inputs : iterable
    An iterable of input tensors or values.
device : int, optional
    The target device index. Default is 0.

Returns
-------
str
    The generated CUDA kernel code as a string.
)")
      .def(
          "get_scheduled_ir",
          [](FusionExecutorCache& self,
             const py::iterable& iter,
             bool tensor_transforms,
             std::optional<int64_t> device) {
            return self.getScheduledIrFor(
                from_pyiterable(iter, device), tensor_transforms);
          },
          py::arg("inputs"),
          py::arg("tensor_transforms") = false,
          py::arg("device") = 0,
          R"(
Get the scheduled IR for the given input configuration.

Parameters
----------
inputs : iterable
    An iterable of input tensors or values.
tensor_transforms : bool, optional
    Whether to include tensor transformations in the output. Default is False.
device : int, optional
    The target device index. Default is 0.

Returns
-------
str
    The scheduled intermediate representation (IR) as a string.
)")
      .def(
          "get_most_recent_scheduled_ir",
          &FusionExecutorCache::getMostRecentScheduledIr,
          py::arg("tensor_transforms") = false,
          R"(
Get the scheduled IR from the most recent execution.

Parameters
----------
tensor_transforms : bool, optional
    Whether to include tensor transformations in the output. Default is False.

Returns
-------
str
    The scheduled intermediate representation (IR) as a string.

Notes
-----
- Returns None if execution has occurred yet.
)")
      .def(
          "get_output_shardings",
          [](FusionExecutorCache& self) {
            Fusion* fusion = self.getMostRecentKernelRuntime()
                                 ->fusionSegments()
                                 ->completeFusion();
            std::vector<Sharding> output_shardings = getOutputShardings(fusion);
            NVF_ERROR(
                output_shardings.empty() ||
                    std::ssize(output_shardings) ==
                        (int64_t)fusion->outputs().size(),
                "Found ",
                std::ssize(output_shardings),
                " output shardings but expected ",
                fusion->outputs().size(),
                " or 0.");
            return output_shardings;
          },
          R"(
Get the output shardings of the fusion.

Returns
-------
list of Sharding
    The output shardings of the fusion.
)");
}

} // namespace

void bindRuntime(py::module& nvfuser) {
  bindFusion(nvfuser);
  bindFusionExecutorCache(nvfuser);
}

} // namespace nvfuser::python

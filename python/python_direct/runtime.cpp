// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include <bindings.h>
#include <direct_utils.h>
#include <python_common/distributed_tensor.h>
#include <python_utils.h>
#include <tensor_caster.h>

#include <fusion.h>
#include <options.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser::python {

// direct_bindings/runtime.cpp contains Fusion and FusionExecutorCache.
// TODO IrContainer and Kernel would be implemented here.

namespace {

void bindFusion(nb::module_& nvfuser) {
  nb::class_<FusionGuard>(nvfuser, "FusionGuard")
      .def(
          nb::init<Fusion*>(),
          nb::arg("fusion"),
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
  nb::class_<Fusion>(nvfuser, "Fusion")
      .def(nb::init<>(), R"(
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
          "__eq__",
          &Fusion::sameDefinition,
          R"(

Whether the fusion definitions are the same.

Returns
-------
bool
    The equality of the fusion definitions.
)")
      .def(
          "inputs",
          &Fusion::inputs,
          nb::rv_policy::reference,
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
          nb::rv_policy::reference,
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
          nb::rv_policy::reference_internal,
          R"(
Return all Vals registered in the fusion.

Returns
-------
list of Val
    The Vals registered in the fusion.
)")
      .def("add_input", &Fusion::addInput, nb::arg("input"), R"(
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
      .def("add_output", &Fusion::addOutput, nb::arg("output"), R"(
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
          nb::arg("output"),
          nb::arg("alias_input"),
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
          nb::arg("from_outputs_only") = true,
          R"(
Return a string representing the arithmetic expressions in the fusion.

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
)")
      .def(
          "print_kernel",
          [](Fusion& f, const CompileParams& compile_params) {
            // Send debug messages to stringstream
            std::stringstream ss;
            DebugStreamGuard dsg(ss);

            f.printKernel(compile_params);
            return ss.str();
          },
          nb::arg("compile_params") = CompileParams(),
          R"(
Lower the fusion and return the generated CUDA kernel as a string.

Parameters
----------
compile_params : CompileParams, optional
Parameters to control the compilation process.
Default is default-constructed CompileParams.

Returns
-------
str
    The CUDA kernel as a string.
)");
}

void bindFusionExecutorCache(nb::module_& nvfuser) {
  nb::class_<FusionExecutorCache>(nvfuser, "FusionExecutorCache")
      .def(
          "__init__",
          [](FusionExecutorCache* self,
             const Fusion* fusion,
             int64_t fusion_id,
             bool auto_schedule) {
            // Make a copy of the fusion for FusionExecutorCache to own.
            new (self) FusionExecutorCache(
                std::make_unique<Fusion>(*fusion), fusion_id, auto_schedule);
          },
          nb::arg("fusion"),
          nb::arg("fusion_id") = 0,
          nb::arg("auto_schedule") = true,
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
             const nb::iterable& iter,
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
          nb::arg("inputs"),
          nb::kw_only(),
          nb::arg("device") = nb::none(),
          nb::arg("_enable_options") = nb::list(),
          nb::arg("_disable_options") = nb::list(),
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
          "is_compiled",
          [](FusionExecutorCache& self,
             const nb::iterable& iter,
             std::optional<int64_t> device) {
            return self.isCompiled(from_pyiterable(iter, device));
          },
          nb::arg("inputs"),
          nb::arg("device") = 0,
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
          nb::rv_policy::reference,
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
             const nb::iterable& iter,
             bool intrinsic_code,
             std::optional<int64_t> device) {
            return self.getCodeFor(
                from_pyiterable(iter, device), intrinsic_code);
          },
          nb::arg("inputs"),
          nb::arg("intrinsic_code") = false,
          nb::arg("device") = 0,
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
             const nb::iterable& iter,
             bool tensor_transforms,
             std::optional<int64_t> device) {
            return self.getScheduledIrFor(
                from_pyiterable(iter, device), tensor_transforms);
          },
          nb::arg("inputs"),
          nb::arg("tensor_transforms") = false,
          nb::arg("device") = 0,
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
          nb::arg("tensor_transforms") = false,
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

void bindKernelExecutor(nb::module_& nvfuser) {
  nb::class_<KernelExecutor>(nvfuser, "KernelExecutor")
      .def(
          nb::init<int64_t, int64_t, int64_t, int64_t>(),
          R"(
               Create a new KernelExecutor.

               A KernelExecutor is responsible for compiling and executing CUDA kernels.
               It manages the compilation process, kernel caching, and runtime execution.

               Parameters
               ----------
               fusion_id : int, optional
                   A unique identifier for the fusion. Used for caching compiled kernels.
                   Default is 0.
               concrete_id : int, optional
                   A unique identifier for the concrete implementation of the fusion.
                   Used for caching compiled kernels.
                   Default is 0.
               runtime_id : int, optional
                   A unique identifier for the runtime instance.
                   Used for caching compiled kernels.
                   Default is 0.
               group_id : int, optional
                   A unique identifier for the group of operations.
                   Used for segmented fusions.
                   Default is 0.

               Examples
               --------
               >>> executor = KernelExecutor()
               >>> executor.compile(fusion)
               >>> outputs = executor.run(inputs)
             )",
          nb::arg("fusion_id") = 0,
          nb::arg("concrete_id") = 0,
          nb::arg("runtime_id") = 0,
          nb::arg("group_id") = 0)
      .def(
          "compile",
          [](KernelExecutor& self,
             Fusion* fusion,
             const nb::iterable& args,
             const LaunchParams& launch_constraints,
             const CompileParams& compile_params,
             SchedulerType scheduler_type) {
            self.compile(
                fusion,
                from_pyiterable(args),
                launch_constraints,
                compile_params,
                scheduler_type);
          },
          R"(
              Compile a fusion into a CUDA kernel.

              Parameters
              ----------
              fusion : Fusion
                  The fusion to compile.
              args : KernelArgumentHolder, optional
                  The kernel arguments. If empty, will be populated during run.
              launch_constraints : LaunchParams, optional
                  Constraints for kernel launch parameters.
              compile_params : CompileParams, optional
                  Parameters for kernel compilation.
              scheduler_type : SchedulerType, optional
                  The type of scheduler to use (default: None).

              Returns
              -------
              None
            )",
          nb::arg("fusion"),
          nb::arg("args") = nb::list(),
          nb::arg("launch_constraints") = LaunchParams(),
          nb::arg("compile_params") = CompileParams(),
          nb::arg("scheduler_type") = SchedulerType::None)
      .def(
          "run",
          [](KernelExecutor& self,
             const nb::iterable& args,
             const LaunchParams& launch_constraints,
             const CompileParams& compile_params) {
            KernelArgumentHolder outputs = self.run(
                from_pyiterable(args), {}, launch_constraints, compile_params);
            return to_tensor_vector(outputs);
          },
          R"(
              Run the compiled kernel with the given arguments.

              Parameters
              ----------
              args : KernelArgumentHolder
                  The input arguments for the kernel.
              launch_constraints : LaunchParams, optional
                  Constraints for kernel launch parameters.
              compile_params : CompileParams, optional
                  Parameters for kernel compilation.

              Returns
              -------
              KernelArgumentHolder
                  The output arguments containing the results.
            )",
          nb::arg("args"),
          nb::arg("launch_constraints") = LaunchParams(),
          nb::arg("compile_params") = CompileParams())
      .def(
          "is_compiled",
          &KernelExecutor::isCompiled,
          R"(
               Check if the kernel has been compiled.

               Returns
               -------
               bool
                   True if the kernel has been compiled, False otherwise.
             )");
}

} // namespace

void bindRuntime(nb::module_& nvfuser) {
  bindFusion(nvfuser);
  bindFusionExecutorCache(nvfuser);
  bindKernelExecutor(nvfuser);
}

} // namespace nvfuser::python

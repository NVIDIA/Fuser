// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/container.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ops/all_ops.h>
#include <python_frontend/python_bindings.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>
#include <scheduler/tools/inlining.h>

namespace nvfuser::python_frontend {

// direct_bindings/runtime.cpp contains Fusion and FusionExecutorCache.
// TODO IrContainer and Kernel would be implemented here.

namespace {

void bindIrContainer(py::module& fusion) {
  py::class_<nvfuser::FusionGuard>(fusion, "FusionGuard")
      .def(
          py::init<nvfuser::Fusion*>(),
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

Examples
--------
>>> fusion = Fusion()
>>> with FusionGuard(fusion):
...     # Define fusion operations here
...     t0 = ops.add(x, y)
...     # The fusion context is automatically restored when exiting the with block

Notes
-----
- Only one fusion can be active at a time
- The guard automatically handles saving and restoring the previous fusion context
- It's recommended to use the guard in a with statement for automatic cleanup
)");

  // NOTE: manage, get_managed, get_managed_safe, stop_managing, has_managed are
  // template functions. Pybind requires explicit template specialization.
  py::class_<nvfuser::Fusion, std::unique_ptr<nvfuser::Fusion, py::nodelete>>(
      fusion, "Fusion")
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
      .def("clear", &nvfuser::Fusion::clear, R"(
Clear all nodes and reset the fusion to its initial state.

This removes all expressions, values, inputs, and outputs from the fusion.
)")
      .def("remove_expr", &nvfuser::Fusion::removeExpr, py::arg("expr"), R"(
Remove an expression from the fusion.

Parameters
----------
expr : Expr
    The expression to remove.
)")
      .def("remove_val", &nvfuser::Fusion::removeVal, py::arg("val"), R"(
Remove a value from the fusion.

Parameters
----------
val : Val
    The value to remove.
)")
      .def("add_input", &nvfuser::Fusion::addInput, py::arg("input"), R"(
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
      .def("add_output", &nvfuser::Fusion::addOutput, py::arg("output"), R"(
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
      .def("remove_input", &nvfuser::Fusion::removeInput, py::arg("input"), R"(
Deregister a value as an input to the fusion.

Parameters
----------
input : Val
    The input value to deregister.
)")
      .def(
          "remove_output",
          &nvfuser::Fusion::removeOutput,
          py::arg("output"),
          R"(
Deregister a value as an output of the fusion.

Parameters
----------
output : Val
    The output value to deregister.
)")
      .def(
          "print_math",
          &nvfuser::Fusion::printMath,
          py::arg("from_outputs_only") = true,
          R"(
Print arithmetic expressions in the fusion.

Parameters
----------
from_outputs_only : bool, optional
    If True, only print expressions reachable from outputs.
    If False, print all expressions.
    Default is True.
)")
      .def("print_transforms", &nvfuser::Fusion::printTransforms, R"(
Print all transformations used in the fusion.

This shows how tensor views have been transformed through operations like
split, merge, and reorder.
)")
      .def(
          "print_kernel",
          &nvfuser::Fusion::printKernel,
          py::arg("compile_params") = nvfuser::CompileParams(),
          R"(
Lower the fusion and print the generated CUDA kernel.

Parameters
----------
compile_params : CompileParams, optional
    Parameters to control the compilation process.
    Default is default-constructed CompileParams.
)")
      .def("exprs", &nvfuser::Fusion::exprs, R"(
Get all expressions in the fusion in topological order.

Returns
-------
list of Expr
    The expressions in topological order.
)")
      .def("used_math_vals", &nvfuser::Fusion::usedMathVals, R"(
Get all values in math expressions that cannot be eliminated.

Returns
-------
list of Val
    The values that must be computed.
)")
      .def(
          "terminating_math_vals",
          &nvfuser::Fusion::terminatingMathVals,
          R"(
Get all values that are produced by used math expressions and have no further consumers.

Returns
-------
list of Val
    The terminating values in math expressions.
)")
      .def(
          "inputs",
          &nvfuser::Fusion::inputs,
          py::return_value_policy::reference,
          R"(
Get all inputs to the fusion.

Returns
-------
list of Val
    The fusion inputs in registration order.
)")
      .def("inputs_and_created", &nvfuser::Fusion::inputsAndCreated, R"(
Get all inputs and values created within the fusion.

Returns
-------
list of Val
    All inputs and created values.
)")
      .def(
          "outputs",
          &nvfuser::Fusion::outputs,
          py::return_value_policy::reference,
          R"(
Get all outputs of the fusion.

Returns
-------
list of Val
    The fusion outputs in registration order.
)")
      .def(
          "get_terminating_outputs",
          &nvfuser::Fusion::getTerminatingOutputs,
          R"(
Get outputs that are not used by any other expression.

Returns
-------
list of Val
    The terminating outputs.
)")
      .def(
          "alias_output_to_input",
          &nvfuser::Fusion::aliasOutputToInput,
          py::arg("output"),
          py::arg("input"),
          py::arg("alias_info"),
          R"(
Alias an output to an input value.

Parameters
----------
output : Val
    The output value to alias.
input : Val
    The input value to alias to.
alias_info : AliasInfo
    Information about how the values alias.
)")
      .def(
          "has_dynamic_transform",
          &nvfuser::Fusion::hasDynamicTransform,
          R"(
Check if any tensor has a symbolic axis.

Returns
-------
bool
    True if any tensor has a symbolic axis, False otherwise.
)")
      .def("copy", &nvfuser::Fusion::copy, R"(
Create a deep copy of this fusion.

Returns
-------
Fusion
    A new fusion containing copies of all nodes and relationships.
)")
      .def(
          "all_tvs",
          &nvfuser::Fusion::allTvs,
          R"(
Get all TensorViews in the fusion.

Returns
-------
list of TensorView
    All TensorViews in cached order.

Notes
-----
- This is a cached version that is invalidated when the fusion changes.
)");
}

//! Convert a py::iterable to a KernelArgumentHolder
KernelArgumentHolder from_pyiterable(
    const py::iterable& iter,
    std::optional<int64_t> device = std::nullopt) {
  KernelArgumentHolder args;
  for (py::handle obj : iter) {
    // Allows for a Vector of Sizes to be inputed as a list/tuple
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
      for (py::handle item : obj) {
        args.push(torch::jit::toIValue(item, c10::AnyType::get()));
      }
    } else {
      args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
    }
  }

  // Transform int64_t device to int8_t
  std::optional<int8_t> selected_device = std::nullopt;
  if (device.has_value()) {
    NVF_CHECK(device.value() < 256, "Maximum device index is 255");
    selected_device = (int8_t)device.value();
  }
  args.setDeviceIndex(selected_device);
  return args;
}

//! Convert a KernelArgumentHolder to a std::vector<at::Tensor>
std::vector<at::Tensor> to_tensor_vector(const KernelArgumentHolder& outputs) {
  // Convert outputs KernelArgumentHolder to std::vector<at::Tensor>
  std::vector<at::Tensor> out_tensors;
  out_tensors.reserve(outputs.size());
  std::transform(
      outputs.begin(),
      outputs.end(),
      std::back_inserter(out_tensors),
      [](const PolymorphicValue& out) { return out.as<at::Tensor>(); });
  return out_tensors;
}

void bindFusionExecutorCache(py::module& fusion) {
  py::class_<nvfuser::FusionExecutorCache>(fusion, "FusionExecutorCache")
      .def(
          py::init([](Fusion* fusion, int64_t fusion_id, bool auto_schedule) {
            return new FusionExecutorCache(
                std::unique_ptr<Fusion>(fusion), fusion_id, auto_schedule);
          }),
          py::arg("fusion"),
          py::arg("fusion_id") = 0,
          py::arg("auto_schedule") = true,
          R"(
Create a new FusionExecutorCache.

A cache that manages compiled versions of a fusion for different input sizes/types.
The cache automatically handles compilation and execution of the fusion for different input configurations.

Parameters
----------
fusion : Fusion
    The fusion to be executed. The FusionExecutorCache takes ownership of this pointer.
fusion_id : int, optional
    A unique identifier for this fusion. Default is 0.
auto_schedule : bool, optional
    Whether to automatically schedule the fusion. If False, the fusion must be manually scheduled.
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
             std::optional<int64_t> device) {
            KernelArgumentHolder args = from_pyiterable(iter, device);
            KernelArgumentHolder outputs = self.runFusionWithInputs(
                args, std::nullopt, args.getDeviceIndex());
            return to_tensor_vector(outputs);
          },
          py::arg("inputs"),
          py::kw_only(),
          py::arg("device") = py::none(),
          R"(
Execute the fusion with the given inputs.

Parameters
----------
inputs : iterable
    An iterable of input tensors or values. Can include lists/tuples for size vectors.
    All tensor inputs must be on the same device.
device : int, optional
    The device index to execute the fusion on. Must be < 256.
    If None, uses the device of the input tensors.
    Default is None.

Returns
-------
list of torch.Tensor
    The output tensors produced by the fusion.

Notes
-----
- The function automatically handles compilation for new input configurations.
- For best performance, reuse the same input configuration when possible to avoid recompilation.
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
    True if a compiled kernel exists for the input configuration, False otherwise.
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
          "print_fusion",
          &FusionExecutorCache::printFusion,
          R"(
Print the fusion IR to stdout.

This is useful for debugging and understanding the structure of the fusion.
)")
      .def(
          "get_cuda_kernel",
          [](FusionExecutorCache& self,
             const py::iterable& iter,
             std::optional<int64_t> device) {
            return self.getCodeFor(from_pyiterable(iter, device), false);
          },
          py::arg("inputs"),
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

Notes
-----
- This method compiles the kernel if it hasn't been compiled for this input configuration.
- The returned code is the actual CUDA C++ kernel that would be executed.
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

Notes
-----
- This method shows the fusion after scheduling transformations have been applied.
- Useful for understanding how the fusion is actually being executed.
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
- Returns None if no execution has occurred yet.
- This is a faster alternative to get_scheduled_ir() if you want to inspect
  the IR from the last execution.
)");
}

void bindKernelExecutor(py::module& fusion) {
  py::class_<KernelExecutor>(fusion, "KernelExecutor")
      .def(
          py::init<int64_t, int64_t, int64_t, int64_t>(),
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
          py::arg("fusion_id") = 0,
          py::arg("concrete_id") = 0,
          py::arg("runtime_id") = 0,
          py::arg("group_id") = 0)
      .def(
          "compile",
          [](KernelExecutor& self,
             Fusion* fusion,
             const py::iterable& args,
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
          py::arg("fusion"),
          py::arg("args") = py::list(),
          py::arg("launch_constraints") = LaunchParams(),
          py::arg("compile_params") = CompileParams(),
          py::arg("scheduler_type") = SchedulerType::None)
      .def(
          "run",
          [](KernelExecutor& self,
             const py::iterable& args,
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
            outputs : KernelArgumentHolder, optional
                Pre-allocated output tensors. If empty, will be allocated.
            launch_constraints : LaunchParams, optional
                Constraints for kernel launch parameters.
            compile_params : CompileParams, optional
                Parameters for kernel compilation.

            Returns
            -------
            KernelArgumentHolder
                The output arguments containing the results.
          )",
          py::arg("args"),
          py::arg("launch_constraints") = LaunchParams(),
          py::arg("compile_params") = CompileParams())
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

void bindDirectRuntime(py::module& fusion) {
  bindIrContainer(fusion);
  bindFusionExecutorCache(fusion);
  bindKernelExecutor(fusion);
}

} // namespace nvfuser::python_frontend

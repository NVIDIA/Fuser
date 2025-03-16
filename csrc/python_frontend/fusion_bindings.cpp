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

// For all nodes, use multiple inheritance to disable destructor with
// std::unique_ptr<nvfuser::Statement, py::nodelete>. This class will
// disable memory management because it is handled automatically by IrContainer.

namespace {
void bindBaseNodes(py::module& fusion) {
  // Statement
  py::class_<
      nvfuser::Statement,
      std::unique_ptr<nvfuser::Statement, py::nodelete>>(fusion, "Statement")
      .def(
          "name",
          &nvfuser::Statement::name,
          R"(
Get the unique identifier of this statement.

Returns
-------
int
    The integer that represents this statement's unique identifier.
)")
      .def(
          "is_val",
          &nvfuser::Statement::isVal,
          R"(
Check if this statement is a value.

Returns
-------
bool
    True if this statement is a Val, False otherwise.
)")
      .def(
          "is_expr",
          &nvfuser::Statement::isExpr,
          R"(
Check if this statement is an expression.

Returns
-------
bool
    True if this statement is an Expr, False otherwise.
)")
      .def(
          "fusion",
          &nvfuser::Statement::fusion,
          R"(
Get the fusion this statement belongs to.

Returns
-------
Fusion
    The fusion container that owns this statement.
)")
      .def(
          "same_type",
          &nvfuser::Statement::sameType,
          py::arg("other"),
          R"(
Check if this statement has the same type as another statement.

Parameters
----------
other : Statement
    The statement to compare types with.

Returns
-------
bool
    True if both statements are of the same type, False otherwise.
)")
      .def(
          "__eq__",
          &nvfuser::Statement::sameAs,
          R"(
Check if this statement is equal to another statement.

Parameters
----------
other : Statement
    The statement to compare with.

Returns
-------
bool
    True if the statements are equal, False otherwise.
)")
      .def(
          "__str__",
          [](Statement* self) { return self->toString(); },
          "Convert the IterDomain to a string representation.");

  // Val
  py::class_<
      nvfuser::Val,
      nvfuser::Statement,
      std::unique_ptr<nvfuser::Val, py::nodelete>>(fusion, "Val")
      .def(
          "vtype",
          &nvfuser::Val::vtype,
          R"(
Get the value type of this Val.

Returns
-------
ValType
    The type of value (e.g., Scalar, IterDomain, TensorView, etc.).
)")
      .def(
          "dtype",
          &nvfuser::Val::dtype,
          R"(
Get the data type of this Val.

Returns
-------
DataType
    The data type (e.g., Float, Half, Int, etc.).
)")
      .def(
          "is_symbolic",
          &nvfuser::Val::isSymbolic,
          R"(
Check if this value is symbolic (not a concrete value).

Returns
-------
bool
    True if the value is symbolic, False otherwise.
)")
      .def(
          "is_scalar",
          &nvfuser::Val::isScalar,
          R"(
Check if this value is a scalar.

Returns
-------
bool
    True if the value is a scalar, False otherwise.
)")
      .def(
          "is_const_scalar",
          &nvfuser::Val::isConstScalar,
          R"(
Check if this value is a constant scalar.

Returns
-------
bool
    True if all dependencies are constant scalars, False otherwise.
)")
      .def(
          "is_const_int",
          &nvfuser::Val::isConstInt,
          R"(
Check if this value is a constant integer.

Returns
-------
bool
    True if all dependencies are constant integers, False otherwise.
)")
      .def(
          "is_integral_scalar",
          &nvfuser::Val::isIntegralScalar,
          R"(
Check if this value is an integral scalar.

Returns
-------
bool
    True if the value is an integral scalar, False otherwise.
)")
      .def(
          "is_floating_point_scalar",
          &nvfuser::Val::isFloatingPointScalar,
          R"(
Check if this value is a floating point scalar.

Returns
-------
bool
    True if the value is a floating point scalar, False otherwise.
)")
      .def(
          "is_a_bool",
          &nvfuser::Val::isABool,
          R"(
Check if this value is a boolean.

Returns
-------
bool
    True if the value is a boolean, False otherwise.
)")
      .def(
          "is_const",
          &nvfuser::Val::isConst,
          R"(
Check if this value is a constant with no dependencies.

Returns
-------
bool
    True if the value is a constant scalar with no dependencies, False otherwise.
)")
      .def(
          "is_zero",
          &nvfuser::Val::isZero,
          R"(
Check if this value is zero.

Returns
-------
bool
    True if the value is zero, False otherwise.
)")
      .def(
          "is_zero_int",
          &nvfuser::Val::isZeroInt,
          R"(
Check if this value is the integer zero.

Returns
-------
bool
    True if the value is the integer zero, False otherwise.
)")
      .def(
          "is_one",
          &nvfuser::Val::isOne,
          R"(
Check if this value is one.

Returns
-------
bool
    True if the value is one, False otherwise.
)")
      .def(
          "is_one_int",
          &nvfuser::Val::isOneInt,
          R"(
Check if this value is the integer one.

Returns
-------
bool
    True if the value is the integer one, False otherwise.
)")
      .def(
          "is_true",
          &nvfuser::Val::isTrue,
          R"(
Check if this value is true.

Returns
-------
bool
    True if the value is true, False otherwise.
)")
      .def(
          "is_false",
          &nvfuser::Val::isFalse,
          R"(
Check if this value is false.

Returns
-------
bool
    True if the value is false, False otherwise.
)")
      .def(
          "definition",
          &nvfuser::Val::definition,
          R"(
Get the expression that defines this value.

Returns
-------
Expr
    The expression that produces this value, or None if it's an input.
)")
      .def(
          "uses",
          &nvfuser::Val::uses,
          R"(
Get all expressions that use this value as an input.

Returns
-------
list of Expr
    The expressions that consume this value.
)")
      .def(
          "is_fusion_input",
          &nvfuser::Val::isFusionInput,
          R"(
Check if this value is a fusion input.

Returns
-------
bool
    True if the value is a fusion input, False otherwise.
)")
      .def(
          "is_fusion_output",
          &nvfuser::Val::isFusionOutput,
          R"(
Check if this value is a fusion output.

Returns
-------
bool
    True if the value is a fusion output, False otherwise.
)");

  // Expr
  py::class_<
      nvfuser::Expr,
      nvfuser::Statement,
      std::unique_ptr<nvfuser::Expr, py::nodelete>>(fusion, "Expr")
      .def(
          "input",
          &nvfuser::Expr::input,
          py::arg("index"),
          R"(
Get the input value at the specified index.

Parameters
----------
index : int
    The index of the input to retrieve.

Returns
-------
Val
    The input value at the given index.
)")
      .def(
          "output",
          &nvfuser::Expr::output,
          py::arg("index"),
          R"(
Get the output value at the specified index.

Parameters
----------
index : int
    The index of the output to retrieve.

Returns
-------
Val
    The output value at the given index.
)")
      .def(
          "__eq__",
          &nvfuser::Expr::sameAs,
          py::arg("other"),
          R"(
Check if this expression is equal to another expression.

Parameters
----------
other : Expr
    The expression to compare with.

Returns
-------
bool
    True if the expressions are equal, False otherwise.
)")
      .def(
          "get_op_string",
          &nvfuser::Expr::getOpString,
          R"(
Get the string representation of this expression's operation.

Returns
-------
str
    The name/type of the operation this expression performs.
)");
}

void bindInternalBaseNodes(py::module& fusion) {
  // IterDomain
  py::class_<
      nvfuser::IterDomain,
      nvfuser::Val,
      std::unique_ptr<nvfuser::IterDomain, py::nodelete>>(fusion, "IterDomain")
      .def(
          "__eq__",
          &nvfuser::IterDomain::sameAs,
          py::arg("other"),
          R"(
Check if this IterDomain is equal to another IterDomain.

Parameters
----------
other : IterDomain
    The IterDomain to compare with.

Returns
-------
bool
    True if the domains are equal, False otherwise.
)")
      .def(
          "__str__",
          [](IterDomain* self) { return self->toString(/*indent_size=*/0); },
          "Convert the IterDomain to a string representation.")
      .def(
          "is_reduction",
          &nvfuser::IterDomain::isReduction,
          R"(
Check if this is a reduction domain.

Returns
-------
bool
    True if this is a reduction domain, False otherwise.
)")
      .def(
          "is_iteration",
          &nvfuser::IterDomain::isIteration,
          R"(
Check if this is an iteration domain.

Returns
-------
bool
    True if this is an iteration domain, False otherwise.
)")
      .def(
          "is_broadcast",
          &nvfuser::IterDomain::isBroadcast,
          R"(
Check if this is a broadcast domain.

Returns
-------
bool
    True if this is a broadcast domain, False otherwise.
)")
      .def(
          "is_symbolic",
          &nvfuser::IterDomain::isSymbolic,
          R"(
Check if this is a symbolic domain.

Returns
-------
bool
    True if this is a symbolic domain, False otherwise.
)")
      .def(
          "is_rfactor_product",
          &nvfuser::IterDomain::isRFactorProduct,
          R"(
Check if this domain is an rfactor product.

Returns
-------
bool
    True if this is an rfactor product, False otherwise.
)")
      .def(
          "is_parallelized",
          &nvfuser::IterDomain::isParallelized,
          R"(
Check if this domain is parallelized.

Returns
-------
bool
    True if this domain is parallelized, False otherwise.
)")
      .def(
          "get_parallel_type",
          &nvfuser::IterDomain::getParallelType,
          R"(
Get the parallel type of this domain.

Returns
-------
ParallelType
    The parallel type of this domain.
)")
      .def(
          "get_iter_type",
          &nvfuser::IterDomain::getIterType,
          R"(
Get the iteration type of this domain.

Returns
-------
IterType
    The iteration type of this domain.
)")
      .def(
          "extent",
          &nvfuser::IterDomain::extent,
          R"(
Get the extent of this domain.

Returns
-------
Val
    The extent of this domain.
)")
      .def(
          "has_expanded_extent",
          &nvfuser::IterDomain::hasExpandedExtent,
          R"(
Check if this domain has an expanded extent.

Returns
-------
bool
    True if this domain has an expanded extent, False otherwise.
)")
      .def(
          "expanded_extent",
          &nvfuser::IterDomain::expandedExtent,
          R"(
Get the expanded extent of this domain.

Returns
-------
Val
    The expanded extent of this domain.
)")
      .def(
          "maybe_partial",
          &nvfuser::IterDomain::maybePartial,
          R"(
Check if this domain may be partial.

Returns
-------
bool
    True if this domain may be partial, False otherwise.
)")
      .def(
          "parallelize",
          &nvfuser::IterDomain::parallelize,
          py::arg("parallel_type"),
          R"(
Set the parallel type of this domain.

Parameters
----------
parallel_type : ParallelType
    The type of parallelization to apply (e.g., BIDx, TIDx, etc.).

Notes
-----
This is a key function used in scheduling to specify how the domain should be parallelized
across CUDA threads and blocks.
)");

  py::class_<
      nvfuser::TensorDomain,
      nvfuser::Val,
      std::unique_ptr<nvfuser::TensorDomain, py::nodelete>>(
      fusion, "TensorDomain")
      .def(
          "__str__",
          [](TensorDomain* self) { return self->toString(/*indent_size=*/0); },
          "Convert the TensorDomain to a string representation.")
      .def(
          "get_root_domain",
          &nvfuser::TensorDomain::root,
          R"(
Get the root domain of this tensor.

Returns
-------
list of IterDomain
    The root iteration domains.
)")
      .def(
          "get_allocation_domain",
          &nvfuser::TensorDomain::allocation,
          R"(
Get the allocation domain of this tensor.

Returns
-------
list of IterDomain
    The allocation iteration domains.
)")
      .def(
          "get_loop_domain",
          &nvfuser::TensorDomain::loop,
          R"(
Get the loop domain of this tensor.

Returns
-------
list of IterDomain
    The loop iteration domains.
)")
      .def(
          "get_logical_domain",
          &nvfuser::TensorDomain::logical,
          R"(
Get the logical domain of this tensor.

Returns
-------
list of IterDomain
    The logical iteration domains.
)")
      .def(
          "get_maybe_root_domain",
          &nvfuser::TensorDomain::maybeRoot,
          R"(
Get the root domain if it exists.

Returns
-------
list of IterDomain
    The root iteration domains, or empty list if not available.
)")
      .def(
          "get_maybe_allocation_domain",
          &nvfuser::TensorDomain::maybeAllocation,
          R"(
Get the allocation domain if it exists.

Returns
-------
list of IterDomain
    The allocation iteration domains, or empty list if not available.
)")
      .def(
          "is_maybe_root",
          &nvfuser::TensorDomain::isMaybeRoot,
          py::arg("id"),
          R"(
Check if the given IterDomain is potentially a root domain.

Parameters
----------
id : IterDomain
    The IterDomain to check.

Returns
-------
bool
    True if the domain is potentially a root domain, False otherwise.
)");
}

void bindInterfaceNodes(py::module& fusion) {
  py::class_<
      nvfuser::TensorView,
      nvfuser::Val,
      std::unique_ptr<nvfuser::TensorView, py::nodelete>>(fusion, "TensorView")
      .def(
          "__str__",
          [](TensorView* self) { return self->toString(/*indent_size=*/0); },
          "Convert the TensorView to a string representation.")
      .def(
          "num_dims",
          &nvfuser::TensorView::nDims,
          R"(
Get the number of dimensions in this tensor.

Returns
-------
int
    The number of dimensions.
)")
      .def(
          "domain",
          &nvfuser::TensorView::domain,
          R"(
Get the domain of this tensor.

Returns
-------
TensorDomain
    The tensor domain object that describes the dimensionality and properties
    of this tensor. The tensor domain contains information about:
    - Root domain (original dimensions)
    - Allocation domain (how memory is allocated)
    - Loop domain (how iterations are structured)
    - Logical domain (current transformed state)

Notes
-----
The TensorDomain is a fundamental part of the tensor that manages all aspects
of its dimensional properties and transformations.
)")
      .def(
          "get_logical_domain",
          &nvfuser::TensorView::getLogicalDomain,
          R"(
Get the logical domain of this tensor.

Returns
-------
list of IterDomain
    The logical iteration domains.
)")
      .def(
          "get_maybe_root_domain",
          &nvfuser::TensorView::getMaybeRootDomain,
          R"(
Get the root domain of this tensor if it exists.

Returns
-------
list of IterDomain
    The root iteration domains.
)")
      .def(
          "get_maybe_allocation_domain",
          &nvfuser::TensorView::getMaybeAllocationDomain,
          R"(
Get the allocation domain of this tensor if it exists.

Returns
-------
list of IterDomain
    The allocation iteration domains.
)")
      .def(
          "get_loop_domain",
          &nvfuser::TensorView::getLoopDomain,
          R"(
Get the loop domain of this tensor.

Returns
-------
list of IterDomain
    The loop iteration domains.
)")
      .def(
          "axis",
          &nvfuser::TensorView::axis,
          py::arg("index"),
          py::return_value_policy::reference,
          R"(
Get the iteration domain at the specified axis.

Parameters
----------
index : int
    The axis index.

Returns
-------
IterDomain
    The iteration domain at the specified axis.
)")
      .def(
          "has_reduction",
          &nvfuser::TensorView::hasReduction,
          R"(
Check if this tensor has any reduction axes.

Returns
-------
bool
    True if the tensor has reduction axes, False otherwise.
)")
      .def(
          "has_broadcast",
          &nvfuser::TensorView::hasBroadcast,
          R"(
Check if this tensor has any broadcast axes.

Returns
-------
bool
    True if the tensor has broadcast axes, False otherwise.
)")
      .def(
          "is_fusion_input",
          &nvfuser::TensorView::isFusionInput,
          R"(
Check if this tensor is a fusion input.

Returns
-------
bool
    True if the tensor is a fusion input, False otherwise.
)")
      .def(
          "definition",
          &nvfuser::TensorView::definition,
          py::return_value_policy::reference,
          R"(
Get the expression that defines this tensor.

Returns
-------
Expr
    The defining expression, or None if this is an input.
)")
      .def(
          "cache_before",
          &nvfuser::TensorView::cacheBefore,
          py::arg("op_type") = LoadStoreOpType::Set,
          py::return_value_policy::reference,
          R"(
Create a cache of this tensor before its computation.

Parameters
----------
op_type : LoadStoreOpType, optional
    The type of load/store operation. Default is Set.

Returns
-------
TensorView
    The newly created cache tensor.
)")
      .def(
          "cache_after",
          &nvfuser::TensorView::cacheAfter,
          py::arg("op_type") = LoadStoreOpType::Set,
          py::arg("cache_op") = CacheOp::Unspecified,
          py::arg("propagate_allocation_domain") = true,
          py::arg("cached_uses") = std::vector<Expr*>{},
          py::return_value_policy::reference,
          R"(
Create a cache of this tensor after its computation.

Parameters
----------
op_type : LoadStoreOpType, optional
    The type of load/store operation. Default is Set.
cache_op : CacheOp, optional
    The type of cache operation. Default is Unspecified.
propagate_allocation_domain : bool, optional
    Whether to propagate the allocation domain. Default is True.

Returns
-------
TensorView
    The newly created cache tensor.
)")
      .def(
          "set_memory_type",
          &nvfuser::TensorView::setMemoryType,
          py::arg("memory_type"),
          R"(
Set the memory type of this tensor.

Parameters
----------
memory_type : MemoryType
    The memory type to set (e.g., Global, Shared, Local).
)")
      .def(
          "split",
          static_cast<TensorView* (
              nvfuser::TensorView::*)(int64_t, int64_t, bool)>(
              &nvfuser::TensorView::split),
          py::arg("axis"),
          py::arg("factor"),
          py::arg("inner_split") = true,
          py::return_value_policy::reference,
          R"(
Split an axis into two axes.

Parameters
----------
axis : int
    The axis to split.
factor : int
    The factor to split by.
inner_split : bool, optional
    If True, the factor determines the size of the inner domain.
    If False, the factor determines the size of the outer domain.
    Default is True.

Returns
-------
TensorView
    A TensorView with the split axes in its loop domain.
)")
      .def(
          "merge",
          static_cast<TensorView* (nvfuser::TensorView::*)(int64_t)>(
              &nvfuser::TensorView::merge),
          py::arg("axis"),
          py::return_value_policy::reference,
          R"(
Merge an axis with the following axis into one.

Parameters
----------
axis : int
    The axis to merge.

Returns
-------
TensorView
    A TensorView with the merged axis in its loop domain.
)")
      .def(
          "reorder",
          static_cast<TensorView* (
              nvfuser::
                  TensorView::*)(const std::unordered_map<int64_t, int64_t>&)>(
              &nvfuser::TensorView::reorder),
          py::arg("old2new"),
          R"(
Reorder the axes according to the given mapping.

Parameters
----------
old2new : dict of int to int
    Mapping from old positions to new positions.

Returns
-------
TensorView
    A TensorView with its loop domain reordered.
)")
      .def(
          "rfactor",
          static_cast<TensorView* (
              nvfuser::TensorView::*)(const std::vector<int64_t>&)>(
              &nvfuser::TensorView::rFactor),
          py::arg("axes"),
          py::return_value_policy::reference,
          R"(
Perform an rfactor transformation on the specified axes.

Parameters
----------
axes : list of int
    The axes to apply rfactor to.

Returns
-------
TensorView
    The newly created rfactor tensor.
)");

  py::class_<nvfuser::TensorViewBuilder>(fusion, "TensorViewBuilder")
      .def(py::init<>(), R"(
Create a new TensorViewBuilder.

A builder class for creating TensorViews with specified properties like dimensions,
data type, contiguity, shape, and stride order.

Examples
--------
>>> builder = TensorViewBuilder()
>>> tv = (builder
...       .num_dims(2)
...       .dtype(DataType.Float)
...       .shape([3, 4])
...       .contiguity(True)
...       .build())
)")
      .def(
          "num_dims",
          &nvfuser::TensorViewBuilder::ndims,
          py::arg("num_dimensions"),
          R"(
Set the number of dimensions for the TensorView.

Parameters
----------
num_dimensions : int
    Number of dimensions for the tensor.

Returns
-------
TensorViewBuilder
    The builder instance for method chaining.
)")
      .def(
          "dtype",
          &nvfuser::TensorViewBuilder::dtype,
          py::arg("dtype"),
          R"(
Set the data type for the TensorView.

Parameters
----------
dtype : DataType
    The data type for the tensor (e.g., DataType.Float, DataType.Half).

Returns
-------
TensorViewBuilder
    The builder instance for method chaining.
)")
      .def(
          "contiguity",
          static_cast<nvfuser::TensorViewBuilder& (
              nvfuser::TensorViewBuilder::*)(std::vector<std::optional<bool>>)>(
              &nvfuser::TensorViewBuilder::contiguity),
          py::arg("contiguity"),
          R"(
Set the contiguity for each dimension of the TensorView.

Parameters
----------
contiguity : list of Optional[bool]
    List of contiguity flags for each dimension. Use None for unspecified contiguity.

Returns
-------
TensorViewBuilder
    The builder instance for method chaining.
)")
      .def(
          "contiguity",
          static_cast<nvfuser::TensorViewBuilder& (
              nvfuser::TensorViewBuilder::*)(bool)>(
              &nvfuser::TensorViewBuilder::contiguity),
          py::arg("contiguous"),
          R"(
Set uniform contiguity for all dimensions of the TensorView.

Parameters
----------
contiguous : bool
    If True, make all dimensions contiguous. If False, make all dimensions non-contiguous.

Returns
-------
TensorViewBuilder
    The builder instance for method chaining.
)")
      .def(
          "shape",
          static_cast<nvfuser::TensorViewBuilder& (
              nvfuser::TensorViewBuilder::*)(std::vector<nvfuser::Val*>)>(
              &nvfuser::TensorViewBuilder::shape),
          py::arg("shape"),
          R"(
Set the shape of the TensorView using Val pointers.

Parameters
----------
shape : list of Val
    List of Val pointers defining the size of each dimension.

Returns
-------
TensorViewBuilder
    The builder instance for method chaining.
)")
      .def(
          "shape",
          static_cast<nvfuser::TensorViewBuilder& (
              nvfuser::TensorViewBuilder::*)(const std::vector<int64_t>&)>(
              &nvfuser::TensorViewBuilder::shape),
          py::arg("shape"),
          R"(
Set the shape of the TensorView using integer values.

Parameters
----------
shape : list of int
    List of integers defining the size of each dimension.

Returns
-------
TensorViewBuilder
    The builder instance for method chaining.
)")
      .def(
          "expanded",
          &nvfuser::TensorViewBuilder::expanded,
          py::arg("expanded"),
          R"(
Set whether dimensions are expanded.

Parameters
----------
expanded : list of bool
    List of flags indicating whether each dimension is expanded.

Returns
-------
TensorViewBuilder
    The builder instance for method chaining.
)")
      .def(
          "stride_order",
          &nvfuser::TensorViewBuilder::strideOrder,
          py::arg("stride_order"),
          R"(
Set the stride order of the dimensions.

Parameters
----------
stride_order : list of int
    List of indices defining the stride ordering of dimensions.
    The ordering is from fastest varying (innermost) to slowest varying (outermost).

Returns
-------
TensorViewBuilder
    The builder instance for method chaining.
)")
      .def(
          "build",
          &nvfuser::TensorViewBuilder::build,
          py::return_value_policy::reference,
          R"(
Build and return the configured TensorView.

Returns
-------
TensorView
    A new TensorView instance with the configured properties.

Notes
-----
- All required properties (dimensions, dtype, shape) must be set before building.
- The build method validates the configuNVFUSER::DIMENSION SEPARATOR POSITION.
)");
}

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

namespace {
//! Convert a py::iterable to a KernelArgumentHolder
KernelArgumentHolder from_pyiterable(
    const py::iterable& iter,
    std::optional<int8_t> device) {
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
} // namespace

void bindRuntime(py::module& fusion) {
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
            // Transform py::iterable to KernelArgumentHolder
            KernelArgumentHolder args = from_pyiterable(iter, device);

            // Run fusion with inputs
            KernelArgumentHolder outputs = self.runFusionWithInputs(
                args, std::nullopt, args.getDeviceIndex());

            // Convert outputs KernelArgumentHolder to std::vector<at::Tensor>
            std::vector<at::Tensor> out_tensors;
            out_tensors.reserve(outputs.size());
            std::transform(
                outputs.begin(),
                outputs.end(),
                std::back_inserter(out_tensors),
                [](const PolymorphicValue& out) {
                  return out.as<at::Tensor>();
                });
            return out_tensors;
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

} // namespace

void bindFusion(py::module& nvfuser) {
  py::module fusion = nvfuser.def_submodule("fusion", "CPP Fusion");
  bindIrContainer(fusion);
  bindBaseNodes(fusion);
  bindInternalBaseNodes(fusion);
  bindInterfaceNodes(fusion);
  bindOperations(fusion);
  bindRuntime(fusion);
}

} // namespace nvfuser::python_frontend

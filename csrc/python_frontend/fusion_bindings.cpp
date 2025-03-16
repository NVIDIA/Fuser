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
void bindBaseNodes(py::module& nvfuser) {
  // Statement
  py::class_<
      nvfuser::Statement,
      std::unique_ptr<nvfuser::Statement, py::nodelete>>(nvfuser, "Statement")
      .def(
          "name",
          &nvfuser::Statement::name,
          "Return the int that represents its name")
      .def(
          "is_val",
          &nvfuser::Statement::isVal,
          "Short cut to figure out if it is a value")
      .def(
          "is_expr",
          &nvfuser::Statement::isExpr,
          "Short cut to figure out if it is an expression")
      .def(
          "fusion",
          &nvfuser::Statement::fusion,
          "Return the fusion this statement belongs to")
      .def(
          "container",
          &nvfuser::Statement::container,
          "Return the container this statement belongs to")
      .def(
          "same_type",
          &nvfuser::Statement::sameType,
          "Return if this statement is the same type as another statement")
      .def(
          "same_as",
          &nvfuser::Statement::sameAs,
          "Return if this statement is the same as another statement")
      .def(
          "to_string",
          &nvfuser::Statement::toString,
          "Return the string representation of the statement");

  // Val
  py::class_<
      nvfuser::Val,
      nvfuser::Statement,
      std::unique_ptr<nvfuser::Val, py::nodelete>>(nvfuser, "Val")
      .def("vtype", &nvfuser::Val::vtype, "Return the ValType of the value")
      .def("dtype", &nvfuser::Val::dtype, "Return the DataType of the value")
      .def(
          "is_symbolic",
          &nvfuser::Val::isSymbolic,
          "Returns if the value is symbolic")
      .def(
          "is_scalar",
          &nvfuser::Val::isScalar,
          "Returns if the Val is a scalar")
      .def(
          "is_const_scalar",
          &nvfuser::Val::isConstScalar,
          "Returns if all dependencies are constant scalars")
      .def(
          "is_const_int",
          &nvfuser::Val::isConstInt,
          "Returns if all dependencies are constant integers")
      .def(
          "is_integral_scalar",
          &nvfuser::Val::isIntegralScalar,
          "Returns if it is an integral scalar")
      .def(
          "is_floating_point_scalar",
          &nvfuser::Val::isFloatingPointScalar,
          "Returns if it is a floating point scalar")
      .def("is_a_bool", &nvfuser::Val::isABool, "Returns if it is a boolean")
      .def(
          "evaluate",
          &nvfuser::Val::evaluate,
          "If this Val's history is comprised only of constant values, will return a PolymorphicValue.")
      .def(
          "is_const",
          &nvfuser::Val::isConst,
          "Returns if no dependencies and is a constant scalar.")
      .def("is_zero", &nvfuser::Val::isZero, "Returns if the value is zero")
      .def(
          "is_zero_int",
          &nvfuser::Val::isZeroInt,
          "Returns if the value is zero integer")
      .def("is_one", &nvfuser::Val::isOne, "Returns if the value is one")
      .def(
          "is_one_int",
          &nvfuser::Val::isOneInt,
          "Returns if the value is one integer")
      .def("is_true", &nvfuser::Val::isTrue, "Returns if the value is true")
      .def("is_false", &nvfuser::Val::isFalse, "Returns if the value is false")
      .def(
          "definition",
          &nvfuser::Val::definition,
          "Returns the Expr that this value is an output of, returns nullptr if none was found")
      .def(
          "uses",
          &nvfuser::Val::uses,
          "Returns the Exprs for which this is an input.")
      .def(
          "is_fusion_input",
          &nvfuser::Val::isFusionInput,
          "Returns if the value is a fusion input")
      .def(
          "is_fusion_output",
          &nvfuser::Val::isFusionOutput,
          "Returns if the value is a fusion output");

  // Expr
  py::class_<
      nvfuser::Expr,
      nvfuser::Statement,
      std::unique_ptr<nvfuser::Expr, py::nodelete>>(nvfuser, "Expr")
      .def(
          "input",
          &nvfuser::Expr::input,
          py::arg("index"),
          "Returns the input at the given index.\n"
          "Args:\n"
          "    index (int): The index of the input to retrieve.")
      .def(
          "output",
          &nvfuser::Expr::output,
          py::arg("index"),
          "Returns the output at the given index.\n"
          "Args:\n"
          "    index (int): The index of the output to retrieve.")
      .def(
          "attribute_val",
          &nvfuser::Expr::attributeVal,
          "Returns the attribute value at the given index")
      .def(
          "same_op",
          &nvfuser::Expr::sameOp,
          "Check that if this and other are the same operator.")
      .def(
          "same_as",
          &nvfuser::Expr::sameAs,
          "Return if this and other are the same")
      .def(
          "get_op_string",
          &nvfuser::Expr::getOpString,
          "Get the name of an expression");
}

void bindInternalBaseNodes(py::module& nvfuser) {
  // IterType
  py::enum_<nvfuser::IterType>(nvfuser, "IterType")
      .value("Iteration", nvfuser::IterType::Iteration)
      .value("Reduction", nvfuser::IterType::Reduction)
      .value("Broadcast", nvfuser::IterType::Broadcast)
      .value("Stride", nvfuser::IterType::Stride)
      .value("GatherScatter", nvfuser::IterType::GatherScatter)
      .value("VectorComponent", nvfuser::IterType::VectorComponent)
      .value("Symbolic", nvfuser::IterType::Symbolic);

  // SwizzleType
  py::enum_<nvfuser::SwizzleType>(nvfuser, "SwizzleType")
      .value("NoSwizzle", nvfuser::SwizzleType::NoSwizzle)
      .value("XOR", nvfuser::SwizzleType::XOR)
      .value("CyclicShift", nvfuser::SwizzleType::CyclicShift);

  // IterDomain
  py::class_<
      nvfuser::IterDomain,
      nvfuser::Val,
      std::unique_ptr<nvfuser::IterDomain, py::nodelete>>(nvfuser, "IterDomain")
      .def(
          "same_as",
          &nvfuser::IterDomain::sameAs,
          "Return if this statement is the same as another statement")
      .def(
          "to_string",
          (std::string(nvfuser::IterDomain::*)(
              int))&nvfuser::IterDomain::toString,
          py::arg("indent_size") = 0,
          "Return the string representation of the statement")
      .def(
          "is_reduction",
          &nvfuser::IterDomain::isReduction,
          "Return if this iter domain is reduction")
      .def(
          "is_iteration",
          &nvfuser::IterDomain::isIteration,
          "Return if this iter domain is iteration")
      .def(
          "is_r_factor_product",
          &nvfuser::IterDomain::isRFactorProduct,
          "Return if this iter domain is rfactor product")
      .def(
          "is_broadcast",
          &nvfuser::IterDomain::isBroadcast,
          "Return if this iter domain is broadcast")
      .def(
          "is_symbolic",
          &nvfuser::IterDomain::isSymbolic,
          "Return if this iter domain is symbolic")
      .def(
          "is_gather_scatter",
          &nvfuser::IterDomain::isGatherScatter,
          "Return if this iter domain is gather scatter")
      .def(
          "is_stride",
          &nvfuser::IterDomain::isStride,
          "Return if this iter domain is stride")
      .def(
          "is_vector_component",
          &nvfuser::IterDomain::isVectorComponent,
          "Return if this iter domain is vector component")
      .def(
          "is_parallelized",
          &nvfuser::IterDomain::isParallelized,
          "Return if this iter domain is parallelized")
      .def(
          "is_block_dim",
          &nvfuser::IterDomain::isBlockDim,
          "Return if this iter domain is mapped to a grid dimension")
      .def(
          "is_thread_dim",
          &nvfuser::IterDomain::isThreadDim,
          "Return if this iter domain is mapped to a block dimension")
      .def(
          "is_thread",
          &nvfuser::IterDomain::isThread,
          "Return if this iter domain is either mapped to a block or grid dimension")
      .def(
          "is_device_dim",
          &nvfuser::IterDomain::isDeviceDim,
          "Return if this iter domain is device dimension")
      .def("parallelize", &nvfuser::IterDomain::parallelize, py::arg("t"), "")
      .def("get_parallel_type", &nvfuser::IterDomain::getParallelType, "")
      .def("get_iter_type", &nvfuser::IterDomain::getIterType, "")
      .def("start", &nvfuser::IterDomain::start, "")
      .def("stop", &nvfuser::IterDomain::stop, "")
      .def("stop_offset", &nvfuser::IterDomain::stopOffset, "")
      .def("extent", &nvfuser::IterDomain::extent, "")
      .def("has_expanded_extent", &nvfuser::IterDomain::hasExpandedExtent, "")
      .def(
          "expanded_extent",
          &nvfuser::IterDomain::expandedExtent,
          "Returns the expanded extent of a strided broadcast entry.")
      .def(
          "get_maybe_expanded_extent",
          &nvfuser::IterDomain::getMaybeExpandedExtent,
          "")
      .def(
          "has_padding_to_multiple_of_warp",
          &nvfuser::IterDomain::hasPaddingToMultipleOfWarp,
          "Indicates if this iterdomain had padding dynamical or statical")
      .def(
          "get_maybe_size_after_padding",
          &nvfuser::IterDomain::getMaybeSizeAfterPadding,
          "Returns a concrete value if this iterdomain has been padded to a statical size.")
      .def(
          "maybe_partial",
          &nvfuser::IterDomain::maybePartial,
          "True if range of iteration domain isn't across the full extent")
      .def(
          "is_implicit_broadcast",
          &nvfuser::IterDomain::isImplicitBroadcast,
          "Check if IterDomain is a broadcast axis with compile-time known extent. "
          "This is the case with all size-1 IterDomains on a TensorView's root domain "
          "when the TensorView is created.")
      .def(
          "strided_split",
          &nvfuser::IterDomain::stridedSplit,
          py::arg("factor"),
          "Split for stride by a given factor. "
          "It effectively does an inner split by the factor and sets the inner domain as a Stride domain.")
      .def(
          "is_mma",
          &nvfuser::IterDomain::isMma,
          "Marks that this id represents a instruction loop, mma use only. "
          "An instruction loop can be considered a generalization of vectorization. "
          "It also represents a loop that's implemented by an instruction and "
          "should not be realized by codegen and cannot be inlined with. "
          "As an example, if a mma macro, call it mma_eg implements: "
          "for m in M for n in N for k in K C[m,n] += A[m,k]*B[k,n], "
          "But the generated code should simply be: mma_eg(C,A,B) "
          "without the 3 level loopnest, i.e. they're instruction loops. "
          "In the actual mma macros, the loopnests it implements is a "
          "transformed version of above to match the mma swizzle. "
          "So it's different implicit loopnest for different macros. "
          "MmaSwizzler will label the instruction loops case-by-case.")
      .def(
          "is_bulk",
          &nvfuser::IterDomain::isBulk,
          "Marks that this id represents an instruction loop, cp.async.bulk use only.");

  py::class_<
      nvfuser::TensorDomain,
      nvfuser::Val,
      std::unique_ptr<nvfuser::TensorDomain, py::nodelete>>(
      nvfuser, "TensorDomain")
      .def("__eq__", &TensorDomain::operator==, "Equality operator")
      .def("__ne__", &TensorDomain::operator!=, "Inequality operator")
      .def("num_dims", &TensorDomain::nDims, "Number of dimensions")
      .def(
          "same_as",
          (bool(TensorDomain::*)(const Statement*) const) &
              TensorDomain::sameAs,
          "Check if same as Statement")
      .def(
          "same_as",
          (bool (*)(
              const std::vector<IterDomain*>&,
              const std::vector<IterDomain*>&))&TensorDomain::sameAs,
          "Check if same as IterDomains")
      .def(
          "to_string",
          (std::string(TensorDomain::*)(int, bool)
               const)&TensorDomain::toString,
          py::arg("indent_size") = 0,
          py::arg("loop_only") = false,
          "String representation")
      .def(
          "to_string",
          (std::string(TensorDomain::*)(int) const)&TensorDomain::toString,
          py::arg("indent_size") = 0,
          "String representation")
      .def(
          "to_inline_string",
          &TensorDomain::toInlineString,
          py::arg("indent_size") = 0,
          "Inline string representation")
      .def("contiguity", &TensorDomain::contiguity, "Contiguity vector")
      .def("stride_order", &TensorDomain::strideOrder, "Stride order")
      .def("set_contiguity", &TensorDomain::setContiguity, "Set contiguity")
      .def(
          "get_contiguity_string",
          &TensorDomain::getContiguityString,
          "Contiguity string")
      .def(
          "has_block_reduction",
          &TensorDomain::hasBlockReduction,
          "Has block reduction")
      .def(
          "has_grid_reduction",
          &TensorDomain::hasGridReduction,
          "Has grid reduction")
      .def(
          "has_block_broadcast",
          &TensorDomain::hasBlockBroadcast,
          "Has block broadcast")
      .def(
          "has_grid_broadcast",
          &TensorDomain::hasGridBroadcast,
          "Has grid broadcast")
      .def("has_root", &TensorDomain::hasRoot, "Has root")
      .def("has_allocation", &TensorDomain::hasAllocation, "Has allocation")
      .def(
          "has_view_like_r_factor",
          &TensorDomain::hasViewLikeRFactor,
          "Has view-like rfactor")
      .def("has_vectorize", &TensorDomain::hasVectorize, "Has vectorize")
      .def(
          "has_symbolic_axis",
          &TensorDomain::hasSymbolicAxis,
          "Has symbolic axis")
      .def(
          "get_reduction_axis",
          &TensorDomain::getReductionAxis,
          "Reduction axis")
      .def("root", &TensorDomain::root, "Root domain")
      .def("maybe_root", &TensorDomain::maybeRoot, "Maybe root domain")
      .def("is_root", &TensorDomain::isRoot, "Is root ID")
      .def("is_maybe_root", &TensorDomain::isMaybeRoot, "Is maybe root ID")
      .def("logical", &TensorDomain::logical, "Logical domain")
      .def("is_logical", &TensorDomain::isLogical, "Is logical ID")
      .def("allocation", &TensorDomain::allocation, "Allocation domain")
      .def("is_allocation", &TensorDomain::isAllocation, "Is allocation ID")
      .def("loop", &TensorDomain::loop, "Loop domain")
      .def("initial_loop", &TensorDomain::initialLoop, "Initial loop domain")
      .def("is_loop", &TensorDomain::isLoop, "Is loop ID")
      .def(
          "is_initial_loop", &TensorDomain::isInitialLoop, "Is initial loop ID")
      .def("all_ids", &TensorDomain::allIDs, "All IDs")
      .def("all_exprs", &TensorDomain::allExprs, "All ID expressions")
      .def("all_statements", &TensorDomain::allStatements, "All ID statements")
      .def(
          "maybe_allocation",
          &TensorDomain::maybeAllocation,
          "Maybe allocation domain")
      .def("additional_ids", &TensorDomain::additionalIDs, "Additional IDs")
      .def("set_loop_domain", &TensorDomain::setLoopDomain, "Set loop domain")
      .def(
          "set_allocation_domain",
          (void(TensorDomain::*)(
              std::vector<IterDomain*>, std::vector<std::optional<bool>>)) &
              TensorDomain::setAllocationDomain,
          "Set allocation domain")
      .def(
          "set_allocation_domain",
          (void(TensorDomain::*)(std::vector<IterDomain*>, bool)) &
              TensorDomain::setAllocationDomain,
          "Set allocation domain")
      .def("reset_domains", &TensorDomain::resetDomains, "Reset domains")
      .def("axis", &TensorDomain::axis, "Get axis")
      .def("pos_of", &TensorDomain::posOf, "Position of IterDomain")
      .def(
          "root_pos_of",
          &TensorDomain::rootPosOf,
          "Position of root IterDomain")
      .def("broadcast", &TensorDomain::broadcast, "Broadcast IterDomain")
      .def("split", &TensorDomain::split, "Split axis")
      .def("merge", &TensorDomain::merge, "Merge axes")
      .def("reorder", &TensorDomain::reorder, "Reorder axes")
      .def(
          "swizzle",
          (void(TensorDomain::*)(SwizzleType, int64_t, int64_t)) &
              TensorDomain::swizzle,
          "Apply 2D swizzle")
      .def(
          "swizzle",
          (void(TensorDomain::*)(
              Swizzle2DType, int64_t, int64_t, SwizzleMode)) &
              TensorDomain::swizzle,
          "Apply 2D swizzle")
      .def("resize", &TensorDomain::resize, "Resize axis")
      .def("view", &TensorDomain::view, "Transform TensorView")
      .def("flatten", &TensorDomain::flatten, "Flatten dimensions")
      .def("ordered_as", &TensorDomain::orderedAs, "Reorder IterDomains")
      .def(
          "no_devices", &TensorDomain::noDevices, "IterDomains with no devices")
      .def(
          "get_contiguity_filled_with",
          &TensorDomain::getContiguityFilledWith,
          "Contiguity filled with value")
      .def("r_factor", &TensorDomain::rFactor, "Apply rFactor");
}

void bindInterfaceNodes(py::module& nvfuser) {
  py::class_<
      nvfuser::TensorView,
      nvfuser::Val,
      std::unique_ptr<nvfuser::TensorView, py::nodelete>>(nvfuser, "TensorView")
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

  py::class_<nvfuser::TensorViewBuilder>(nvfuser, "TensorViewBuilder")
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

void bindIrContainer(py::module& nvfuser) {
  py::class_<nvfuser::FusionGuard>(nvfuser, "FusionGuard")
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
      nvfuser, "Fusion")
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

void bindOperations(py::module& nvfuser) {
  py::module ops = nvfuser.def_submodule("ops", "CPP Fusion Operations");
  // Add functions to the submodule
  ops.def(
      "add",
      static_cast<nvfuser::Val* (*)(nvfuser::Val*, nvfuser::Val*)>(
          nvfuser::add),
      "Add two Vals");
  ops.def(
      "add",
      static_cast<nvfuser::TensorView* (*)(nvfuser::TensorView*,
                                           nvfuser::Val*)>(nvfuser::add),
      "Add TensorView and Val");
  ops.def(
      "add",
      static_cast<nvfuser::TensorView* (*)(nvfuser::Val*,
                                           nvfuser::TensorView*)>(nvfuser::add),
      "Add Val and TensorView");
  ops.def(
      "add",
      static_cast<nvfuser::TensorView* (*)(nvfuser::TensorView*,
                                           nvfuser::TensorView*)>(nvfuser::add),
      "Add two TensorViews");
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

void bindRuntime(py::module& nvfuser) {
  py::class_<nvfuser::FusionExecutorCache>(nvfuser, "FusionExecutorCache")
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

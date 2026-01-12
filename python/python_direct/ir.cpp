// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <ranges>

#include <bindings.h>
#include <direct_utils.h>
#include <python_utils.h>

// size and shape operations are a part of TensorView bindings but not a
// part of TensorView IR node.
#include <ops/arith.h>

#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>

namespace nvfuser::python {

// For all nodes, memory management is handled automatically by IrContainer.

namespace {

void bindBaseNodes(nb::module_& nvfuser) {
  // Statement
  nb::class_<Statement>(nvfuser, "Statement")
      .def(
          "__str__",
          [](Statement* self) { return self->toString(); },
          R"(Get string representation of Statement.)");

  // Val
  nb::class_<Val, Statement>(nvfuser, "Val")
      .def(
          "is_symbolic",
          &Val::isSymbolic,
          R"(
Check if this value is symbolic (not a concrete value).

Returns
-------
bool
    True if the value is symbolic, False otherwise.
)")
      .def(
          "is_tensor",
          [](Val* self) { return self->isA<TensorView>(); },
          R"(
Check if this value is a TensorView.

Returns
-------
bool
    True if the value is a TensorView, False otherwise.
)")
      .def(
          "definition",
          &Val::definition,
          nb::rv_policy::reference_internal,
          R"(
Get the definition of this expression.

Returns
-------
Expr
    The definition of this expression.
)")
      .def(
          "uses",
          &Val::uses,
          nb::rv_policy::reference_internal,
          R"(
Get the uses of this expression.

Returns
-------
Expr
    The uses of this expression.
)");

  // Expr
  nb::class_<Expr, Statement>(nvfuser, "Expr")
      .def(
          "input",
          &Expr::input,
          nb::arg("index"),
          nb::rv_policy::reference_internal,
          R"(
Get the input of this expression.

Parameters
----------
index : int
    The index of the input.

Returns
-------
Expr
    The input of this expression.
)")
      .def(
          "output",
          &Expr::output,
          nb::arg("index"),
          nb::rv_policy::reference_internal,
          R"(
Get the output of this expression.

Parameters
----------
index : int
    The index of the output.

Returns
-------
Expr
    The output of this expression.
)");
}

void bindInternalBaseNodes(nb::module_& nvfuser) {
  // IterDomain
  nb::class_<IterDomain, Val>(nvfuser, "IterDomain")
      .def(
          "__str__",
          [](IterDomain* self) { return self->toString(/*indent_size=*/0); },
          "Convert the IterDomain to a string representation.")
      .def(
          "is_reduction",
          &IterDomain::isReduction,
          R"(
Check if this domain is a reduction domain.

Returns
-------
bool
    True if the domain is a reduction domain, False otherwise.
)")
      .def(
          "extent",
          &IterDomain::extent,
          nb::rv_policy::reference_internal,
          R"(
Get the extent of this domain.

Returns
-------
Val
    The extent of this domain.
)")
      .def(
          "parallelize",
          &IterDomain::parallelize,
          nb::arg("parallel_type"),
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

  // TensorDomain
  nb::class_<TensorDomain, Val>(nvfuser, "TensorDomain")
      .def(
          "__str__",
          [](TensorDomain* self) { return self->toString(/*indent_size=*/0); },
          "Convert the TensorDomain to a string representation.");
}

void bindInterfaceNodes(nb::module_& nvfuser) {
  nb::class_<TensorView, Val>(nvfuser, "TensorView")
      .def(
          "__str__",
          [](TensorView* self) { return self->toString(/*indent_size=*/0); },
          "Convert the TensorView to a string representation.")
      .def_prop_ro(
          "ndim",
          [](TensorView* self) {
            return std::ranges::distance(
                self->getLogicalDomain() | TensorDomain::kNoReductions);
          },
          R"(
Get the number of dimensions in this tensor.

Returns
-------
int
    The number of dimensions.
)")
      .def(
          "size",
          [](TensorView* self, int64_t dim) { return size(self, dim); },
          nb::arg("dim"),
          nb::rv_policy::reference_internal,
          R"(
Get the size of this tensor.

Parameters
----------
dim : int
    The dimension in the tensor.

Returns
-------
int
    The size of the dimension.
)")
      .def(
          "shape",
          [](TensorView* self) { return shape(self); },
          nb::rv_policy::reference_internal,
          R"(
Get the shape of this tensor.

Returns
-------
list of Val
    The shape of this tensor.
)")
      .def(
          "dtype",
          [](TensorView* self) -> PrimDataType {
            DataType dt = self->dtype();
            NVF_CHECK(
                std::holds_alternative<PrimDataType>(dt.type),
                "Expected PrimDataType but got type: ",
                dt);
            return std::get<PrimDataType>(dt.type);
          },
          R"(
Get the data type of this tensor.

Returns
-------
DataType
    The data type of this tensor.
)")
      .def("has_root", &TensorView::hasRoot, R"(
Check if this tensor has a root domain.

Returns
-------
bool
    True if the tensor has a root domain, False otherwise.
)")
      .def(
          "domain",
          &TensorView::domain,
          nb::rv_policy::reference_internal,
          R"(
Get the TensorDomain of this tensor.

Returns
-------
TensorDomain
    The tensor domain object that describes the dimensionality and properties
    of this tensor. The tensor domain contains information about:
    - Root domain (The original dimensions if logical domain contains rFactor iterDomains.)
    - Logical domain (The original dimensions. It may contain rFactor iterDomains.)
    - Allocation domain (How the memory is allocated for the tensor?)
    - Loop domain (The for-loop structure for the tensor.)
)")
      .def(
          "get_logical_domain",
          &TensorView::getLogicalDomain,
          nb::rv_policy::reference_internal,
          R"(
Get the logical domain of this tensor.

Returns
-------
list of IterDomain
    The logical iteration domain.
)")
      .def(
          "get_loop_domain",
          &TensorView::getLoopDomain,
          nb::rv_policy::reference_internal,
          R"(
Get the loop domain of this tensor.

Returns
-------
list of IterDomain
    The loop iteration domains.
)")
      .def(
          "get_root_domain",
          &TensorView::getRootDomain,
          nb::rv_policy::reference_internal,
          R"(
Get the root domain of this tensor.

Returns
-------
list of IterDomain
    The root iteration domains.
)")
      .def(
          "cache_after",
          &TensorView::cacheAfter,
          nb::arg("op_type") = LoadStoreOpType::Set,
          nb::arg("cache_op") = CacheOp::Unspecified,
          nb::arg("propagate_allocation_domain") = true,
          nb::arg("cached_uses") = nb::list(),
          nb::rv_policy::reference,
          R"(
      Cache the TensorView after the specified operation.

      Parameters
      ----------
      op_type : LoadStoreOpType, optional
          The type of load/store operation (default: Set).
      cache_op : CacheOp, optional
          The type of cache operation (default: Unspecified).

      Returns
      -------
      TensorView
          The new cached TensorView.
    )")
      .def(
          "cache_before",
          &TensorView::cacheBefore,
          nb::arg("op_type") = LoadStoreOpType::Set,
          nb::rv_policy::reference,
          R"(
      Cache the TensorView before the specified operation.

      Parameters
      ----------
      op_type : LoadStoreOpType, optional
          The type of load/store operation (default: Set).

      Returns
      -------
      TensorView
          The new cached TensorView.
    )")
      .def(
          "set_memory_type",
          &TensorView::setMemoryType,
          nb::arg("memory_type"),
          R"(
      Set the memory type for the TensorView.

      Parameters
      ----------
      memory_type : MemoryType
          The memory type to set.

      Returns
      -------
      None
    )")
      .def(
          "split",
          static_cast<TensorView* (TensorView::*)(int64_t, int64_t, bool)>(
              &TensorView::split),
          nb::arg("axis"),
          nb::arg("factor"),
          nb::arg("inner_split") = true,
          nb::rv_policy::reference,
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
          "inner_split",
          [](TensorView* self, int64_t axis, int64_t factor) {
            return self->split(axis, factor, true);
          },
          nb::arg("axis"),
          nb::arg("factor"),
          nb::rv_policy::reference,
          R"(
Split an axis into inner-first order (alias of split with inner_split=True).

Parameters
----------
axis : int
    The axis to split.
factor : int
    The factor to split by (inner size).

Returns
-------
TensorView
    A TensorView with the split axes in its loop domain.
)")
      .def(
          "outer_split",
          [](TensorView* self, int64_t axis, int64_t factor) {
            return self->split(axis, factor, false);
          },
          nb::arg("axis"),
          nb::arg("factor"),
          nb::rv_policy::reference,
          R"(
Split an axis into outer-first order (alias of split with inner_split=False).

Parameters
----------
axis : int
    The axis to split.
factor : int
    The factor to split by (outer size).

Returns
-------
TensorView
    A TensorView with the split axes in its loop domain.
)")
      .def(
          "merge",
          static_cast<TensorView* (TensorView::*)(int64_t, int64_t)>(
              &TensorView::merge),
          nb::arg("axis_o"),
          nb::arg("axis_i"),
          nb::rv_policy::reference,
          R"(
Merge two axes into one axis.

Parameters
----------
axis_o : int
    The outer axis to merge.
axis_i : int
    The inner axis to merge.

Returns
-------
TensorView
    A TensorView with the merged axes in its loop domain.
)")
      .def(
          "reorder",
          [](TensorView* self, std::unordered_map<int64_t, int64_t>& old2new) {
            return self->reorder(old2new);
          },
          nb::arg("old2new") = nb::dict(),
          nb::rv_policy::reference,
          R"(
Reorder the axes of this tensor.

Parameters
----------
old2new : dict of int to int
    The new order of the axes.

Returns
-------
TensorView
    A TensorView with the reordered axes in its loop domain.
)")
      .def(
          "swizzle",
          [](TensorView* self, int64_t x, int64_t y) {
            return self->swizzle(SwizzleType::XOR, x, y);
          },
          nb::rv_policy::reference,
          nb::arg("x"),
          nb::arg("y"),
          R"(
Swizzle the axes of this tensor.

Parameters
----------
x : int
    The x axis to swizzle.
y : int
    The y axis to swizzle.

Returns
-------
TensorView
    A TensorView with the swizzled axes in its loop domain.
)")
      .def(
          "rfactor",
          static_cast<TensorView* (TensorView::*)(const std::vector<int64_t>&)>(
              &TensorView::rFactor),
          nb::arg("axes"),
          nb::rv_policy::reference,
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
)")
      .def(
          "set_allocation_domain",
          static_cast<void (TensorView::*)(std::vector<IterDomain*>, bool)>(
              &TensorView::setAllocationDomain),
          nb::arg("new_allocation_domain"),
          nb::arg("new_contiguity"),
          R"(
Set the allocation domain of this tensor.

Parameters
----------
new_allocation_domain : list of IterDomain
    The new allocation iteration domains.
new_contiguity : bool
    The new contiguity flag.
)")
      .def(
          "set_device_mesh",
          &TensorView::setDeviceMesh,
          nb::arg("mesh"),
          R"(
Set the device mesh of this tensor.

Parameters
----------
mesh : DeviceMesh
    The device mesh to set.
)")
      .def(
          "axis",
          &TensorView::axis,
          nb::arg("index"),
          nb::rv_policy::reference_internal,
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
)");
}

// Creates a new TensorView with the specified properties.
//
// This function creates a tensor with the given shape, contiguity, data type,
// and stride order. It handles both CPU and CUDA tensors, with special handling
// for CPU scalar tensors.
//
// Parameters
// ----------
// shape : vector<int64_t>
//     The shape of the tensor
// contiguity : vector<optional<bool>>
//     The contiguity flags for each dimension. None indicates a broadcast
//     dimension.
// dtype : PrimDataType
//     The data type of the tensor (e.g., Float, Half, Int)
// is_cpu : bool, optional
//     Whether this is a CPU tensor. Default is false.
// stride_order : vector<int64_t>, optional
//     The stride order of dimensions. Default is empty.
//
// Returns
// -------
// TensorView*
//     A pointer to the newly created TensorView.
//
// Notes
// -----
// - CPU tensors are only supported for scalar tensors (empty shape)
// - The stride order is normalized and validated before use
// - Expanded dimensions are automatically determined based on shape and
// contiguity
// - The tensor is created using the TensorViewBuilder pattern
TensorView* defineTensor(
    std::vector<int64_t> shape,
    std::vector<std::optional<bool>> contiguity,
    PrimDataType dtype,
    bool is_cpu = false,
    std::vector<int64_t> stride_order = {}) {
  normalizeStrideOrder(stride_order);

  TensorView* tv = TensorViewBuilder()
                       .contiguity(contiguity)
                       .shape(shape)
                       .dtype(dtype)
                       .expanded(getExpanded(shape, contiguity, stride_order))
                       .strideOrder(stride_order)
                       .build();

  if (shape.empty() && is_cpu) {
    tv->setCpuScalar(true);
  } else {
    NVF_CHECK(!is_cpu, "CPU non-scalar tensor is not supported!");
  }
  return tv;
}

// return the unpacked shape and dtype for a given packed dtype, where we need
// to double the size of the inner most dimension.
std::tuple<std::vector<int64_t>, PrimDataType> translatePackedDtype(
    const std::vector<int64_t>& shape,
    const PrimDataType dtype,
    const std::vector<int64_t>& stride_order) {
  // TODO: switch to isPackedType when the pack width is retrieved through
  // utility functions as well.
  NVF_CHECK(dtype == DataType::Float4_e2m1fn_x2);

  int fastest_dim = shape.size() - 1;
  for (const auto& [i, val] : enumerate(stride_order)) {
    if (val == 0) {
      fastest_dim = i;
      break;
    }
  }
  std::vector<int64_t> un_packed_shape = shape;
  un_packed_shape[fastest_dim] *= 2;
  return {un_packed_shape, DataType::Float4_e2m1fn};
}

namespace {

// Nanobind does not convert torch.Size automatically to std::vector. Instead,
// manually convert nb::sequence to a std::vector
//
// >>> import torch
// >>> type(torch.randn(10,10).shape)
// <class 'torch.Size'>
std::vector<int64_t> get_shape_from_pysequence(nb::sequence seq) {
  std::vector<int64_t> result;
  result.reserve(nb::len(seq));
  try {
    std::transform(
        seq.begin(), seq.end(), std::back_inserter(result), [](nb::handle obj) {
          // Get value from Thunder Proxy
          if (nb::hasattr(obj, "value")) {
            nb::object value = obj.attr("value");
            return nb::cast<int64_t>(value);
          }
          return nb::cast<int64_t>(obj);
        });
  } catch (...) {
    throw nb::type_error("define_tensor(): incompatible function arguments");
  }
  return result;
}

} // namespace

void bindDefineTensor(nb::module_& nvfuser) {
  nvfuser
      .def(
          "define_tensor",
          [](nb::sequence shape_py,
             const std::vector<std::optional<bool>>& contiguity,
             const PrimDataType dtype = DataType::Float,
             const bool is_cpu = false,
             const std::vector<int64_t>& stride_order = {}) -> TensorView* {
            auto shape = get_shape_from_pysequence(shape_py);
            verifyShape(shape);
            if (!isPackedType(dtype)) {
              return defineTensor(
                  shape, contiguity, dtype, is_cpu, stride_order);
            } else {
              auto&& [new_shape, new_dtype] =
                  translatePackedDtype(shape, dtype, stride_order);
              return defineTensor(
                  new_shape, contiguity, new_dtype, is_cpu, stride_order);
            }
          },
          nb::arg("shape"),
          nb::arg("contiguity"),
          nb::arg("dtype") = DataType::Float,
          nb::arg("is_cpu") = false,
          nb::arg("stride_order") = nb::list(),
          nb::rv_policy::reference)
      .def(
          "define_tensor",
          [](nb::sequence shape_py,
             // Contiguity for non-broadcast dimensions.
             const bool contiguity = false,
             const PrimDataType dtype = DataType::Float,
             const bool is_cpu = false,
             const std::vector<int64_t>& stride_order = {}) -> TensorView* {
            auto shape = get_shape_from_pysequence(shape_py);
            verifyShape(shape);
            if (!isPackedType(dtype)) {
              return defineTensor(
                  shape,
                  getContiguityVec(shape, stride_order, contiguity),
                  dtype,
                  is_cpu,
                  stride_order);
            } else {
              auto&& [new_shape, new_dtype] =
                  translatePackedDtype(shape, dtype, stride_order);
              return defineTensor(
                  new_shape,
                  getContiguityVec(new_shape, stride_order, contiguity),
                  new_dtype,
                  is_cpu,
                  stride_order);
            }
          },
          nb::arg("shape"),
          nb::arg("contiguity") = false,
          nb::arg("dtype") = DataType::Float,
          nb::arg("is_cpu") = false,
          nb::arg("stride_order") = nb::list(),
          nb::rv_policy::reference)
      .def(
          "define_tensor",
          [](const std::vector<int64_t>& sizes,
             const std::vector<int64_t>& strides,
             const PrimDataType dtype = DataType::Float,
             const bool static_sizes = false,
             const bool is_cpu = false) -> TensorView* {
            NVF_CHECK(
                sizes.size() == strides.size(),
                "The number of sizes does not match the number of strides.",
                sizes.size(),
                strides.size());
            std::vector<std::optional<bool>> contiguity;
            std::vector<int64_t> stride_order;
            std::tie(contiguity, stride_order) =
                computeTensorDescriptor(sizes, strides);
            if (!isPackedType(dtype)) {
              return defineTensor(
                  getTensorViewBuilderSizes(sizes, static_sizes),
                  contiguity,
                  dtype,
                  is_cpu,
                  stride_order);
            } else {
              auto&& [new_sizes, new_dtype] =
                  translatePackedDtype(sizes, dtype, stride_order);
              return defineTensor(
                  getTensorViewBuilderSizes(new_sizes, static_sizes),
                  contiguity,
                  new_dtype,
                  is_cpu,
                  stride_order);
            }
          },
          nb::arg("sizes"),
          nb::arg("strides"),
          nb::arg("dtype") = DataType::Float,
          nb::arg("static_sizes") = false,
          nb::arg("is_cpu") = false,
          nb::rv_policy::reference);
}

void bindDefineScalar(nb::module_& nvfuser) {
  // The symbolic define_scalar must come before the constant version because of
  // nanobind rules for overload resolution. Essentially, overload functions are
  // tried in the order they were registered with nanobind. If the order is
  // reversed, the PrimDataType is cast to its corresponding Enum integer and
  // used as a Fusion contant.
  //
  // Reference:
  // https://nanobind.readthedocs.io/en/latest/api_core.html#overload-resolution
  nvfuser.def(
      "define_scalar",
      [](PrimDataType dtype = DataType::Double) {
        return IrBuilder::create<Val>(std::monostate{}, dtype);
      },
      nb::arg("dtype") = DataType::Double,
      nb::rv_policy::reference);
  nvfuser.def(
      "define_scalar",
      [](PolymorphicValue::VariantType value,
         std::optional<PrimDataType> dtype) {
        PolymorphicValue cast_value(
            dtype.has_value() ? castToDtype(std::move(value), dtype.value())
                              : std::move(value));
        PrimDataType value_dtype(
            dtype.has_value()
                ? dtype.value()
                : std::get<PrimDataType>(getDataType(cast_value).type));
        return IrBuilder::create<Val>(cast_value, value_dtype);
      },
      nb::arg("value").none(),
      nb::arg("dtype") = std::nullopt,
      nb::rv_policy::reference);
}

} // namespace

void bindFusionIr(nb::module_& nvfuser) {
  bindBaseNodes(nvfuser);
  bindInternalBaseNodes(nvfuser);
  bindInterfaceNodes(nvfuser);
  bindDefineTensor(nvfuser);
  bindDefineScalar(nvfuser);
}

} // namespace nvfuser::python

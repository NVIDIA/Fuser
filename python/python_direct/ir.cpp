// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <python_utils.h>

// size and shape operations are a part of TensorView bindings but not a
// part of TensorView IR node.
#include <ops/arith.h>

#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>

namespace nvfuser::python {

// For all nodes, use multiple inheritance to disable destructor with
// std::unique_ptr<Statement, py::nodelete>. This class will
// disable memory management because it is handled automatically by IrContainer.

namespace {

void bindBaseNodes(py::module& nvfuser) {
  // Statement
  py::class_<Statement, std::unique_ptr<Statement, py::nodelete>>(
      nvfuser, "Statement")
      .def(
          "__str__",
          [](Statement* self) { return self->toString(); },
          R"(Get string representation of Statement.)");

  // Val
  py::class_<Val, Statement, std::unique_ptr<Val, py::nodelete>>(nvfuser, "Val")
      .def(
          "is_symbolic",
          &Val::isSymbolic,
          R"(
Check if this value is symbolic (not a concrete value).

Returns
-------
bool
    True if the value is symbolic, False otherwise.
)");

  // Expr
  py::class_<Expr, Statement, std::unique_ptr<Expr, py::nodelete>>(
      nvfuser, "Expr");
}

void bindInternalBaseNodes(py::module& nvfuser) {
  // IterDomain
  py::class_<IterDomain, Val, std::unique_ptr<IterDomain, py::nodelete>>(
      nvfuser, "IterDomain")
      .def(
          "__str__",
          [](IterDomain* self) { return self->toString(/*indent_size=*/0); },
          "Convert the IterDomain to a string representation.")
      .def(
          "extent",
          &IterDomain::extent,
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

  // TensorDomain
  py::class_<TensorDomain, Val, std::unique_ptr<TensorDomain, py::nodelete>>(
      nvfuser, "TensorDomain")
      .def(
          "__str__",
          [](TensorDomain* self) { return self->toString(/*indent_size=*/0); },
          "Convert the TensorDomain to a string representation.");
}

void bindInterfaceNodes(py::module& nvfuser) {
  py::class_<TensorView, Val, std::unique_ptr<TensorView, py::nodelete>>(
      nvfuser, "TensorView")
      .def(
          "__str__",
          [](TensorView* self) { return self->toString(/*indent_size=*/0); },
          "Convert the TensorView to a string representation.")
      .def(
          "num_dims",
          [](TensorView* self) {
            return TensorDomain::noReductions(self->getLogicalDomain()).size();
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
          py::arg("dim"),
          py::return_value_policy::reference,
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
          py::return_value_policy::reference,
          R"(
Get the shape of this tensor.

Returns
-------
list of Val
    The shape of this tensor.
)")
      .def(
          "domain",
          &TensorView::domain,
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
          "get_loop_domain",
          &TensorView::getLoopDomain,
          R"(
Get the loop domain of this tensor.

Returns
-------
list of IterDomain
    The loop iteration domains.
)")
      .def(
          "split",
          static_cast<TensorView* (TensorView::*)(int64_t, int64_t, bool)>(
              &TensorView::split),
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
          "set_allocation_domain",
          static_cast<void (TensorView::*)(std::vector<IterDomain*>, bool)>(
              &TensorView::setAllocationDomain),
          py::arg("new_allocation_domain"),
          py::arg("new_contiguity"),
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
          py::arg("mesh"),
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

void bindDefineTensor(py::module& nvfuser) {
  nvfuser
      .def(
          "define_tensor",
          [](const std::vector<int64_t>& shape,
             const std::vector<std::optional<bool>>& contiguity,
             const PrimDataType dtype = DataType::Float,
             const bool is_cpu = false,
             const std::vector<int64_t>& stride_order = {}) -> TensorView* {
            verifyShape(shape);
            return defineTensor(shape, contiguity, dtype, is_cpu, stride_order);
          },
          py::arg("shape"),
          py::arg("contiguity"),
          py::arg("dtype") = DataType::Float,
          py::arg("is_cpu") = false,
          py::arg("stride_order") = py::list(),
          py::return_value_policy::reference)
      .def(
          "define_tensor",
          [](const std::vector<int64_t>& shape,
             // Contiguity for non-broadcast dimensions.
             const bool contiguity = false,
             const PrimDataType dtype = DataType::Float,
             const bool is_cpu = false,
             const std::vector<int64_t>& stride_order = {}) -> TensorView* {
            verifyShape(shape);
            return defineTensor(
                shape,
                getContiguityVec(shape, stride_order, contiguity),
                dtype,
                is_cpu,
                stride_order);
          },
          py::arg("shape"),
          py::arg("contiguity") = false,
          py::arg("dtype") = DataType::Float,
          py::arg("is_cpu") = false,
          py::arg("stride_order") = py::list(),
          py::return_value_policy::reference)
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
            return defineTensor(
                getTensorViewBuilderSizes(sizes, static_sizes),
                contiguity,
                dtype,
                is_cpu,
                stride_order);
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = DataType::Float,
          py::arg("static_sizes") = false,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference);
}

void bindDefineScalar(py::module& nvfuser) {
  // The symbolic define_scalar must come before the constant version because of
  // pybind11 rules for overload resolution. Essentially, overload functions are
  // tried in the order they were registered with pybind11. If the order is
  // reversed, the PrimDataType is cast to its corresponding Enum integer and
  // used as a Fusion contant.
  //
  // Reference:
  // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#overload-resolution-order
  nvfuser.def(
      "define_scalar",
      [](PrimDataType dtype = DataType::Double) {
        return IrBuilder::create<Val>(std::monostate{}, dtype);
      },
      py::arg("dtype") = DataType::Double,
      py::return_value_policy::reference);
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
      py::arg("value"),
      py::arg("dtype") = std::nullopt,
      py::return_value_policy::reference);
}

} // namespace

void bindFusionIr(py::module& nvfuser) {
  bindBaseNodes(nvfuser);
  bindInternalBaseNodes(nvfuser);
  bindInterfaceNodes(nvfuser);
  bindDefineTensor(nvfuser);
  bindDefineScalar(nvfuser);
}

} // namespace nvfuser::python

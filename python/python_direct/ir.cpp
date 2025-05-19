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
  py::class_<Val, Statement, std::unique_ptr<Val, py::nodelete>>(
      nvfuser, "Val");

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
          &TensorView::nDims,
          R"(
Get the number of dimensions in this tensor.

Returns
-------
int
    The number of dimensions.
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

// Determines which dimensions of a tensor are expanded.
//
// A dimension is considered expanded if:
// 1. It is marked as a broadcast dimension (contiguity is None)
// 2. It has a non-broadcast size (shape != 1)
//
// Parameters
// ----------
// shape : vector<int64_t>
//     The shape of the tensor
// contiguity : vector<optional<bool>>
//     The contiguity flags for each dimension. None indicates a broadcast
//     dimension.
// stride_order : vector<int64_t>
//     The stride order of dimensions, mapping logical domain to allocation
//     domain.
//
// Returns
// -------
// vector<bool>
//     A vector of boolean flags indicating which dimensions are expanded.
//     The order of this vector corresponds to the logical domain of TensorView.
//
// Notes
// -----
// The function handles the mapping between logical domain and allocation domain
// using the stride order. For each dimension i:
// - If stride_order is empty, contiguity[i] corresponds to dimension i
// - Otherwise, contiguity[i] corresponds to dimension
// (rank - 1 - stride_order[i])
std::vector<bool> getExpanded(
    const std::vector<int64_t>& shape,
    const std::vector<std::optional<bool>>& contiguity,
    const std::vector<int64_t>& stride_order) {
  size_t rank = shape.size();
  std::vector<bool> is_expand(rank);
  for (size_t index : arange(rank)) {
    // since contiguity vector is given to the corresponding order in alloc
    // domain, while is_expand is given to root domain, we need to map it
    // correctly with `contig_index` and `index`.
    //
    // stride_order[i] indicates that:
    //   `logical_domain[i]` maps to `alloc_domain[rank - 1 - stride_order_[i]]`
    //
    // Hence `index` on root domain would be corresponding to the contiguity
    // index `contig_index = rank - 1 - stride_order[index]`
    const size_t contig_index = stride_order.empty()
        ? index
        : rank - 1 - static_cast<size_t>(stride_order[index]);
    const bool is_broadcast = !contiguity[contig_index].has_value();
    const bool has_non_broadcast_size = (shape[index] != 1);
    // A root dimension is expand dimension if:
    //   The dimension is marked a broadcast; and
    //   The dimension has an expanded extent.
    is_expand[index] = is_broadcast && has_non_broadcast_size;
  }
  return is_expand;
}

// Normalizes and validates the stride order for a tensor.
//
// This function processes the stride order vector to ensure it's valid and
// consistent:
// 1. Handles negative indices by converting them to positive indices
// 2. Validates that all indices are within the valid range
// 3. Ensures there are no duplicate entries
//
// Parameters
// ----------
// stride_order : vector<int64_t>
//     The stride order vector to normalize. Can be empty or contain indices
//     from -rank to rank-1, where rank is the size of the vector.
//
// Notes
// -----
// - Negative indices are converted to positive by adding the rank
// - All indices must be within the range [-rank, rank-1]
// - No duplicate indices are allowed
// - An empty stride_order vector is valid and will be left unchanged
void normalizeStrideOrder(std::vector<int64_t>& stride_order) {
  if (stride_order.empty()) {
    return;
  }
  int64_t rank = (int64_t)stride_order.size();
  std::unordered_set<int64_t> order_set;
  for (auto& order : stride_order) {
    order_set.insert(order);
    if (order < 0) {
      NVF_CHECK(
          order >= -rank,
          "defineTensor stride_order argument is out of range, expects >= ",
          -rank,
          ", but got: ",
          order);
      order += rank;
    } else {
      NVF_CHECK(
          order < rank,
          "defineTensor stride_order argument is out of range, expects < ",
          rank,
          ", but got: ",
          order);
    }
  }
  NVF_CHECK(
      order_set.size() == stride_order.size(),
      "defineTensor got duplicated stride_order entries: " +
          toDelimitedString(stride_order));
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

            // TensorViewBuilder assumes any dim with a compile time constant
            // size == 1 is a broadcast axis, symbolic sizes are
            // identified by -1, and size == 0 is not supported.

            // Translate to TensorViewBuilder's view of the world.
            std::vector<int64_t> dim_sizes;
            dim_sizes.reserve(sizes.size());
            for (size_t i : arange(sizes.size())) {
              NVF_ERROR(
                  sizes[i] >= 0,
                  "Size of ",
                  sizes[i],
                  " is not supported in nvFuser. Expected size >= 0.");
              if (static_sizes) {
                dim_sizes.push_back(sizes[i]);
              } else { // Symbolic defined tensor for dynamic shape usage
                if (sizes[i] == 1) {
                  dim_sizes.push_back(1);
                } else {
                  dim_sizes.push_back(-1);
                }
              }
            }

            std::vector<std::optional<bool>> contiguity;
            std::vector<int64_t> stride_order;
            std::tie(contiguity, stride_order) =
                computeTensorDescriptor(sizes, strides);

            return defineTensor(
                std::move(dim_sizes), contiguity, dtype, is_cpu, stride_order);
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = DataType::Float,
          py::arg("static_sizes") = false,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference);
}

} // namespace

void bindFusionIr(py::module& nvfuser) {
  bindBaseNodes(nvfuser);
  bindInternalBaseNodes(nvfuser);
  bindInterfaceNodes(nvfuser);
  bindDefineTensor(nvfuser);
}

} // namespace nvfuser::python

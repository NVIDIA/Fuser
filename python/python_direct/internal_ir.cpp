// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <ir/internal_nodes.h>

namespace nvfuser::python {

// For all nodes, use multiple inheritance to disable destructor with
// std::unique_ptr<Statement, py::nodelete>. This class will
// disable memory management because it is handled automatically by IrContainer.

namespace {

void bindInternalFusionIr(py::module& nvfuser) {
  py::class_<Split, Expr, std::unique_ptr<Split, py::nodelete>> split(
      nvfuser, "Split");
  split.def(
      "__str__",
      [](Split* self) { return self->toString(/*indent_size=*/0); },
      "Convert the Split to a string representation.");
  split.def(
      "outer",
      &Split::outer,
      R"(
Get the outer of this Split.

Returns
-------
IterDomain
    The outer of this Split.
)");
  split.def(
      "inner",
      &Split::inner,
      R"(
Get the inner of this Split.

Returns
-------
IterDomain
    The inner of this Split.
)");

  py::class_<Merge, Expr, std::unique_ptr<Merge, py::nodelete>> merge(
      nvfuser, "Merge");
  merge.def(
      "__str__",
      [](Merge* self) { return self->toString(/*indent_size=*/0); },
      "Convert the Merge to a string representation.");
  merge.def(
      "outer",
      &Merge::outer,
      R"(
Get the outer of this Merge.

Returns
-------
IterDomain
    The outer of this Merge.
)");
  merge.def(
      "inner",
      &Merge::inner,
      R"(
Get the inner of this Merge.

Returns
-------
IterDomain
    The inner of this Merge.
)");

  py::class_<BroadcastOp, Expr, std::unique_ptr<BroadcastOp, py::nodelete>>
      broadcast(nvfuser, "BroadcastOp");
  broadcast.def(
      "__str__",
      [](BroadcastOp* self) { return self->toString(/*indent_size=*/0); },
      "Convert the BroadcastOp to a string representation.");

  py::class_<ReshapeOp, Expr, std::unique_ptr<ReshapeOp, py::nodelete>> reshape(
      nvfuser, "ReshapeOp");
  reshape.def(
      "__str__",
      [](ReshapeOp* self) { return self->toString(/*indent_size=*/0); },
      "Convert the ReshapeOp to a string representation.");

  py::class_<SqueezeOp, Expr, std::unique_ptr<SqueezeOp, py::nodelete>> squeeze(
      nvfuser, "SqueezeOp");
  squeeze.def(
      "__str__",
      [](SqueezeOp* self) { return self->toString(/*indent_size=*/0); },
      "Convert the SqueezeOp to a string representation.");
}

} // namespace

void bindInternalIr(py::module& nvfuser) {
  bindInternalFusionIr(nvfuser);
}

} // namespace nvfuser::python

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

// For all nodes, memory management is handled automatically by IrContainer.

namespace {

void bindInternalFusionIr(nb::module_& nvfuser) {
  nb::class_<Split, Expr> split(nvfuser, "Split");
  split.def(
      "__str__",
      [](Split* self) { return self->toString(/*indent_size=*/0); },
      "Convert the Split to a string representation.");
  split.def(
      "outer",
      &Split::outer,
      nb::rv_policy::reference,
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
      nb::rv_policy::reference,
      R"(
Get the inner of this Split.

Returns
-------
IterDomain
    The inner of this Split.
)");

  nb::class_<Merge, Expr> merge(nvfuser, "Merge");
  merge.def(
      "__str__",
      [](Merge* self) { return self->toString(/*indent_size=*/0); },
      "Convert the Merge to a string representation.");
  merge.def(
      "outer",
      &Merge::outer,
      nb::rv_policy::reference,
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
      nb::rv_policy::reference,
      R"(
Get the inner of this Merge.

Returns
-------
IterDomain
    The inner of this Merge.
)");

  nb::class_<BroadcastOp, Expr> broadcast(nvfuser, "BroadcastOp");
  broadcast.def(
      "__str__",
      [](BroadcastOp* self) { return self->toString(/*indent_size=*/0); },
      "Convert the BroadcastOp to a string representation.");

  nb::class_<ReshapeOp, Expr> reshape(nvfuser, "ReshapeOp");
  reshape.def(
      "__str__",
      [](ReshapeOp* self) { return self->toString(/*indent_size=*/0); },
      "Convert the ReshapeOp to a string representation.");

  nb::class_<SqueezeOp, Expr> squeeze(nvfuser, "SqueezeOp");
  squeeze.def(
      "__str__",
      [](SqueezeOp* self) { return self->toString(/*indent_size=*/0); },
      "Convert the SqueezeOp to a string representation.");
}

} // namespace

void bindInternalIr(nb::module_& nvfuser) {
  bindInternalFusionIr(nvfuser);
}

} // namespace nvfuser::python

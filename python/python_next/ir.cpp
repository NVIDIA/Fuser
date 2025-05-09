// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>

#include <fusion.h>
#include <ir/base_nodes.h>

namespace python {

// For all nodes, use multiple inheritance to disable destructor with
// std::unique_ptr<Statement, py::nodelete>. This class will
// disable memory management because it is handled automatically by IrContainer.

namespace {

using namespace nvfuser;

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

} // namespace

void bindFusionIr(py::module& nvfuser) {
  bindBaseNodes(nvfuser);
}

} // namespace python

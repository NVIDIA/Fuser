// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/all_ops.h>
#include <python_frontend/python_bindings.h>

namespace nvfuser::python_frontend {

void bindOperations(py::module& fusion) {
  py::module ops = fusion.def_submodule("ops", "CPP Fusion Operations");
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

} // namespace nvfuser::python_frontend

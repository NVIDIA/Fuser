// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <direct_bindings.h>

namespace direct_bindings {

void initNvFuserDirectBindings(PyObject* module) {
  auto direct_bindings = py::handle(module).cast<py::module>();
  bindFusionIr(direct_bindings);
  bindRuntime(direct_bindings);
}

} // namespace direct_bindings

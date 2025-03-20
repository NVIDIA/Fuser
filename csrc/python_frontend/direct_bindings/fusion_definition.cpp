// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/all_ops.h>
#include <python_frontend/direct_bindings/fusion_definition.h>

namespace nvfuser::python_frontend {

void bindDirectFusionDefinition(py::module& fusion) {
  py::class_<DirectFusionDefinition> fusion_def(
      fusion, "_DirectFusionDefinition");
  fusion_def.def(py::init<>())
      .def_readwrite("ops", &DirectFusionDefinition::ops)
      .def_readwrite("schedule", &DirectFusionDefinition::sched);
  bindDirectOperations(fusion_def);
  bindDirectScheduleOperators(fusion_def);
}

} // namespace nvfuser::python_frontend

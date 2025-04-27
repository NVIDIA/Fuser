// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <fusion_definition.h>
#include <ops/all_ops.h>

namespace nvfuser::python {

void bindFusionDefinition(py::module& nvfuser) {
  py::class_<FusionDefinition> fusion_def(nvfuser, "_FusionDefinition");
  fusion_def.def(py::init<>())
      .def_readwrite("ops", &FusionDefinition::ops)
      .def_readwrite("schedule", &FusionDefinition::sched);
  bindOperations(fusion_def);
  bindScheduleOperators(fusion_def);
}

} // namespace nvfuser::python

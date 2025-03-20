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

namespace {

void bindScheduleOps(
    py::class_<DirectFusionDefinition::ScheduleOperators>& sched) {
  // TODO: Implement
}

} // namespace

void bindDirectScheduleOperators(
    py::class_<DirectFusionDefinition>& fusion_def) {
  //! The ScheduleOperators class is a nested class of DirectFusionDefinition to
  //! allow the user to query the class for the list of schedule operators.
  //!
  //! Example:
  //!   help(DirectFusionDefinition.ScheduleOperators)
  py::class_<DirectFusionDefinition::ScheduleOperators> nvf_sched(
      fusion_def, "ScheduleOperators");
  nvf_sched.def(py::init<DirectFusionDefinition*>());
  bindScheduleOps(nvf_sched);
}

} // namespace nvfuser::python_frontend

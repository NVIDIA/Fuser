// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace nvfuser::python_frontend {

//! The DirectFusionDefinition class exposes the NvFuser operations to mimic the
//! interface of the FusionDefinition class.
class DirectFusionDefinition {
 public:
  DirectFusionDefinition() : ops(this), sched(this) {}

  //! The Operators define what operations are fused. They are not directly
  //! defined here but in the python bindings through lambda functions.
  struct Operators {
    Operators(DirectFusionDefinition* fd) : fusion_definition(fd) {}
    DirectFusionDefinition* fusion_definition;
  };

  //! The ScheduleOperators are not directly defined in this header but in the
  //! python bindings through lambda functions so the user only needs to define
  //! new operators in one place. ScheduleOperators allow the user to define how
  //! a fusion should be scheduled.
  struct ScheduleOperators {
    ScheduleOperators(DirectFusionDefinition* fd) : fusion_definition(fd) {}
    DirectFusionDefinition* fusion_definition;
  };

  Operators ops;
  ScheduleOperators sched;
};

// Add direct bindings for CPP Fusion Operations
void bindDirectOperations(py::class_<DirectFusionDefinition>& fusion_def);

// Add direct bindings for CPP Schedule Operators
void bindDirectScheduleOperators(
    py::class_<DirectFusionDefinition>& fusion_def);

} // namespace nvfuser::python_frontend

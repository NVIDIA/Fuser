// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace nvfuser::python {

//! The FusionDefinition class exposes the NvFuser operations to mimic the
//! interface of the FusionDefinition class.
class FusionDefinition {
 public:
  FusionDefinition() : ops(this), sched(this) {}

  //! The Operators define what operations are fused. They are not directly
  //! defined here but in the python bindings through lambda functions.
  struct Operators {
    Operators(FusionDefinition* fd) : fusion_definition(fd) {}
    FusionDefinition* fusion_definition;
  };

  //! The ScheduleOperators are not directly defined in this header but in the
  //! python bindings through lambda functions so the user only needs to define
  //! new operators in one place. ScheduleOperators allow the user to define how
  //! a fusion should be scheduled.
  struct ScheduleOperators {
    ScheduleOperators(FusionDefinition* fd) : fusion_definition(fd) {}
    FusionDefinition* fusion_definition;
  };

  Operators ops;
  ScheduleOperators sched;
};

} // namespace nvfuser::python

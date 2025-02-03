// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <iostream>
#include <optional>
#include <vector>

#include <runtime/fusion_executor_cache.h>
#include <type.h>
#include <visibility.h>

namespace nvfuser::python {

struct UserSchedule;

//! This is helper function used to print a python formated
//! Fusion IR DataType when printing a fusion definition.

NVF_API const char* dtypeToPyString(PrimDataType t);

//! FusionDefinition defines the C++ side of a Python Context manager to
//! encapsulate the definition of fusion operations.
//!
//! The FusionDefinition records the state definitions and operations prior
//! to exiting the context manager.  Upon exit, the operations are queried
//! in a cache and the recorded records are used to build an nvFuser Fusion
//! object if the definition missed in the cache.
//!
//! The nested Operators class was designed to allow the user to query all the
//! available Operators in the FusionDefinition via python help.
//!
//! Example:
//!   help(FusionDefinition.Operators)
class NVF_API FusionDefinition {
 public:
  FusionDefinition(std::optional<size_t> id, size_t max_length = 256);

  // The copy/move/assign constructors/operators are removed
  FusionDefinition(const FusionDefinition& fd) = delete;
  FusionDefinition(FusionDefinition&& fd) = delete;
  FusionDefinition& operator=(const FusionDefinition& fd) = delete;
  FusionDefinition& operator=(FusionDefinition&& fd) = delete;

  //! Enter Python Context Manager -- Reset trie for new cache lookup
  NVF_API FusionDefinition* enter();
  //! Exit Python Context Manager -- Triggers Fusion IR build if it is not
  //! cached
  NVF_API void exit();
  //! Prints a python function representing the definition
  NVF_API void print(std::ostream& os) const;
  //! Executes a fusion if a valid definition occurred prior.
  NVF_API at::ArrayRef<c10::IValue> execute(
      const at::ArrayRef<c10::IValue>& inputs,
      std::optional<int8_t> device,
      bool override_user_schedule,
      bool capture_debug_output,
      bool profile,
      std::vector<std::string> _enable_options,
      std::vector<std::string> _disable_options) const;
 private:
  // Book keeping data members for user created schedules

  //! Data member for holding previous fusion container when manually setting
  //! the fusion guard.
  Fusion* prev_fusion_;
  //! Data member for holding the current user schedule object
  UserSchedule* user_sched_;
 public:
  //! The Operators are not directly defined in this header.  They are defined
  //! in the python bindings through lambda functions so the user only needs to
  //! define new operators in one place.
  //! Operators define what operations are fused.
  struct Operators {
    Operators(FusionDefinition* fd) : fusion_definition(fd) {}
    FusionDefinition* fusion_definition;
  };

  //! The SchedOperators are not directly defined in this header.  They are
  //! defined in the python bindings through lambda functions so the user only
  //! needs to define new operators in one place.
  //! SchedOperators allow the user to define how a fusion should be blocked
  //! for execution.
  struct SchedOperators {
    SchedOperators(FusionDefinition* fd) : fusion_definition(fd) {}
    FusionDefinition* fusion_definition;
  };

  Operators ops;
  SchedOperators sched;
};

} // namespace nvfuser::python

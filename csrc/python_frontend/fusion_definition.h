// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <exceptions.h>
#include <iostream>

#include <c10/macros/Export.h>
#include <kernel_cache.h>
#include <python_frontend/fusion_state.h>

namespace nvfuser::python_frontend {

class FusionCache;
class FusionDefinition;
class FusionInterface;
class FusionState;
struct RecordFunctor;
struct UserSchedule;
struct TrieNode;

//! This is helper function used to print a python formated
//! Fusion IR DataType when printing a fusion definition.

const char* dtypeToPyString(PrimDataType t);

//! The Tensor and Scalar classes are used to define separate function signtures
//! in the FusionDefinition to identify the appropriate Operator function.
//!
//! Example:
//!
//!   add(Tensor* arg1, Tensor* arg2) -> Tensor*
//!   add(Tensor* arg1, Val* arg2) -> Tensor*
//!   add(Val* arg1, Val* arg2) -> Val*
struct Tensor {
  Tensor(size_t _index, size_t _dims, FusionDefinition* _fd)
      : index(_index), dims(_dims), fusion_definition(_fd) {}

  size_t operator()() const {
    return index;
  }

  //! A unique index to identifiy each recorded state item.
  size_t index;
  size_t dims;

  //! Pointer to the FusionDefinition used to create this tensor
  //! The FusionDefinition pointer is necessary to enable special
  //! dunder operations (ie __add__()) from the python API.
  FusionDefinition* fusion_definition;
};

struct Scalar {
  Scalar(size_t _index, FusionDefinition* _fd)
      : index(_index), fusion_definition(_fd) {}

  size_t operator()() const {
    return index;
  }

  //! A unique index to identifiy each recorded state item.
  size_t index;

  //! Pointer to the FusionDefinition used to create this scalar
  //! The FusionDefinition pointer is necessary to enable special
  //! dunder operations (ie __add__()) from the python API.
  FusionDefinition* fusion_definition;
};

struct Vector {
  Vector(size_t _index, size_t _size, FusionDefinition* _fd)
      : index(_index), size(_size), fusion_definition(_fd) {}

  size_t operator()() const {
    return index;
  }

  //! A unique index to identifiy each recorded state item.
  size_t index;
  //! Elements in the vector
  size_t size;

  //! Pointer to the FusionDefinition used to create this scalar
  FusionDefinition* fusion_definition;
};

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
class FusionDefinition : public FusionState {
 public:
  FusionDefinition(std::optional<size_t> id, size_t max_length = 256);

  // The copy/move/assign constructors/operators are removed
  FusionDefinition(const FusionDefinition& fd) = delete;
  FusionDefinition(FusionDefinition&& fd) = delete;
  FusionDefinition& operator=(const FusionDefinition& fd) = delete;
  FusionDefinition& operator=(FusionDefinition&& fd) = delete;

  //! Enter Python Context Manager -- Reset trie for new cache lookup
  FusionDefinition* setupDefinition();
  //! Exit Python Context Manager -- Triggers Fusion IR build if it is not
  //! cached
  void finalizeDefinition();
  //! Setup user scheduling of a fusion
  //! Copies fusion object and sets up FusionGuard
  void setupSchedule(const at::ArrayRef<c10::IValue>& inputs);
  //! Finalized use scheduling of a fusion
  //! resets FusionGuard, lowers IR to a kernel, compiles kernel
  void finalizeSchedule(const at::ArrayRef<c10::IValue>& inputs);
  //! Prints a python function representing the definition
  void print(std::ostream& os) const;
  //! Executes a fusion if a valid definition or cache lookup occurred prior
  std::vector<at::Tensor> execute(
      const at::ArrayRef<c10::IValue>& inputs,
      bool override_user_schedule,
      bool capture_debug_output,
      std::optional<int8_t> device) const;
  //! Return debugging output captured through exeuction with
  //! capture_debug_output=true
  std::optional<std::string> getDebugOutput() const {
    return debug_output_;
  }
  //! Return the unscheduled Fusion IR
  std::string fusionIr();
  //! Return the Cuda code for the last executed set of inputs
  std::string lastCudaCode(bool intrinsic_code, bool override_user_schedule)
      const;
  //! Return the Cuda code for the given inputs
  std::string cudaCodeFor(
      const at::ArrayRef<c10::IValue>& inputs,
      bool intrinsic_code,
      bool override_user_schedule) const;
  //! Return the Cuda code for the last executed set of inputs
  std::string lastScheduledFusionIr(
      bool tensor_transforms,
      bool override_user_schedule) const;
  //! Return the Cuda code for the given inputs
  std::string scheduledFusionIrFor(
      const at::ArrayRef<c10::IValue>& inputs,
      bool tensor_transforms,
      bool override_user_schedule) const;
  //! Return fusion id of defined FusionDefinition
  std::optional<size_t> id() const;
  //! Prints the Prescheduled Fusion IR representation
  void printMathIr();

  bool completed() {
    return id().has_value();
  }

  //! These methods are used to record the FusionDefinition for cache lookup

  //! Defines a Scalar State Record
  Scalar defineScalar();
  //! Defines a Tensor State Record
  Tensor defineTensor(size_t dims);
  //! Defines a Vector State Record
  Vector defineVector(size_t size);
  //! Defines a Record that records the operation required to
  //! build the corresponding Fusion IR operation on cache miss.
  void defineRecord(RecordFunctor* record);
  //! Gets a Record State object
  State recordingState(size_t index) const;

 private:
  //! Returns the FusionCache Ptr that holds the cache of Fusions
  FusionCache* fusionCache() const;
  //! Return a prescheduled Fusion object
  Fusion* preschedFusion();

  //! Holds the defined maximum length of a FusionDefinition in order to
  //! prevent a run away error. The user should feel free to increase this
  //! number as appropriate.
  size_t max_length_;
  //! Fusion Cache Id for Scheduled Fusion.
  std::optional<size_t> fusion_id_;
  //! A pointer to the FusionCache.
  FusionCache* fusion_cache_;
  //! Current pointer to node in FusionCache.
  TrieNode* trie_node_;

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
    bool validUse() const {
      return !fusion_definition->completed();
    }

    FusionDefinition* fusion_definition;
  };

  //! The SchedOperators are not directly defined in this header.  They are
  //! defined in the python bindings through lambda functions so the user only
  //! needs to define new operators in one place.
  //! SchedOperators allow the user to define how a fusion should be blocked
  //! for execution.
  struct SchedOperators {
    SchedOperators(FusionDefinition* fd) : fusion_definition(fd) {}
    bool validUse() const {
      return fusion_definition->completed();
    }

    FusionDefinition* fusion_definition;
  };

  Operators ops;
  SchedOperators sched;

 private:
  mutable std::optional<std::string> debug_output_ = std::nullopt;
};

} // namespace nvfuser::python_frontend

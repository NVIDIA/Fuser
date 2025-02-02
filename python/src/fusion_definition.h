// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <iostream>
#include <unordered_map>
#include <exceptions.h>
#include <python_frontend/distributed_tensor.h>
#include <python_frontend/fusion_state.h>
#include <python_frontend/segmentation.h>
#include <visibility.h>

namespace nvfuser::python {

class FusionCache;
class FusionDefinition;
class FusionInterface;
class FusionState;
struct RecordFunctor;
class SegmentationState;
struct TrieNode;
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
class NVF_API FusionDefinition : public FusionState {
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
  //! Executes a fusion if a valid definition or cache lookup occurred prior.
  //!
  //! This method returns a list of `DistributedTensor`s. Each
  //! `DistributedTensor` is either the local view of a distributed tensor
  //! (when the mesh is non-empty) or a non-distributed tensor
  //! (when the mesh is empty).
  //!
  //! Alternatives considered:
  //! 1. Return std::vector<std::variant<at::Tensor, DistributedTensor>>.
  //! Because DistributedTensor can also represent a non-distributed tensor, I
  //! chose the current API for simplicity -- C++ is more verbose than Python
  //! when dealing with dynamic types.
  //! 2. Return std::variant<std::vector<at::Tensor>,
  //! std::vector<DistributedTensor>>. Same reason.
  //! 3. Store output shardings (i.e. the mesh and the mesh-to-tenseor-axis
  //! mapping) to a field of FusionDefinition and retrieve it using another
  //! method. This would be similar to getDebugOutput. I didn't choose that
  //! because it introduced a new state in the class that could get out of sync.
  NVF_API std::vector<DistributedTensor> execute(
      const at::ArrayRef<c10::IValue>& inputs,
      std::optional<int8_t> device,
      bool override_user_schedule,
      bool capture_debug_output,
      bool profile,
      std::vector<std::string> _enable_options,
      std::vector<std::string> _disable_options) const;
 private:
  //! Returns the FusionCache Ptr that holds the cache of Fusions
  FusionCache* fusionCache() const;
  //! Composite operations can create hidden TensorViews in the CPP fusion
  //! These TensorViews are not visible from python definition. This function
  //! finds and adds them to FusionDefinition
  void findHiddenTensorViews(Fusion* fusion);
  //! Update Symbolic FusionStates after DynamicTransform pass
  void updateSymbolicStates(
      const std::unordered_map<Val*, Val*>& symbolic_to_concretized_map);
  // Check that the NvFuser TensorView and the Python Tensor dimensions match.
  // Apply after buildFusionIr
  void verifyTensorDimensions();

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
  //! Number of recording_states_ before applying user schedule
  int64_t num_recording_states_presched_ = 0;
  //! Data member that creates SegmentedFusion from cloned, prescheduled Fusion
  //! then translates the segments to python FusionDefinitions.
  std::unique_ptr<SegmentationState> segmentation_state_;

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
};

} // namespace nvfuser::python

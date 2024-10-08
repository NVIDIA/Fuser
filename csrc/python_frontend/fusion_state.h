// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <exceptions.h>
#include <ir/interface_nodes.h>
#include <serde/fusion_cache_generated.h>
#include <visibility.h>

namespace nvfuser::python_frontend {

struct RecordFunctor;

struct State {
  State()
      : index(0),
        stype(serde::StateType::None),
        inline_def_record_(std::nullopt) {}
  State(
      size_t _index,
      serde::StateType _stype,
      std::optional<const RecordFunctor*> inline_def_record = std::nullopt)
      : index(_index), stype(_stype), inline_def_record_(inline_def_record) {}

  bool inlineDef() const;
  void setInlineDefRecord(const RecordFunctor* record);
  const RecordFunctor* inlineDefRecord() const;

  bool operator==(const State& other) const;
  bool operator!=(const State& other) const;

  //! A unique index to identifiy each recorded state item.
  size_t index;
  //! StateType is either: Tensor, Scalar, or Vector
  serde::StateType stype;

 private:
  // This data member is only set if this state is inline defined!
  std::optional<const RecordFunctor*> inline_def_record_;
};

NVF_API std::ostream& operator<<(std::ostream& os, const State& state);

//! FusionState contains the information used to build a new cpp Fusion object.
//! Unlike FusionDefinition, it does not modify the FusionCache Trie structure.
class FusionState {
 public:
  FusionState();

  // The copy/move/assign constructors/operators are removed
  FusionState(const FusionState& other) = delete;
  FusionState(FusionState&& other) noexcept = delete;
  FusionState& operator=(const FusionState& other) = delete;
  FusionState& operator=(FusionState&& other) noexcept = delete;
  virtual ~FusionState() = default;

  //! Get fusion object
  Fusion* fusion();
  //! Prints the Fusion IR representation
  void printIr() const;

  //! Adds a Fusion IR Tensor/Scalar object
  NVF_API void addFusionState(Val* val);
  //! Adds a Fusion IR Vector of Scalars
  void addFusionStateVector(std::vector<Val*> val);
  //! Gets a Fusion IR Tensor/Scalar object
  NVF_API Val* getFusionState(size_t index) const;
  //! Gets a Fusion IR Vector of Scalars
  NVF_API const std::vector<Val*>& getFusionStateVector(size_t index) const;
  //! Number of fusion states
  NVF_API size_t numFusionStates() const;
  //! Sets a Fusion IR Tensor/Scalar object
  NVF_API void setFusionState(size_t index, Val* val);
  //! Sets a Fusion IR Vector of Scalars
  NVF_API void setFusionStateVector(size_t index, std::vector<Val*> val);

  //! Adds a Tensor/Scalar input to the Fusion object
  NVF_API void addInput(Val* input);
  //! Adds a Tensor/Scalar output to the Fusion object
  NVF_API void addOutput(Val* output);
  //! Alias an Output to Input in the Fusion object
  NVF_API void aliasOutputToInput(Val* output, Val* input);

  //! Add a Record
  void addRecord(RecordFunctor* record);
  //! Builds an nvFuser Fusion IR object
  void buildFusionIr(Fusion* fusion);

  //! Create clone of FusionState
  std::unique_ptr<FusionState> clone();

 private:
  //! Change the fusion ptr and reset its state
  void resetFusionState(Fusion* fusion, size_t size);

 protected:
  //! Holds an End Record
  std::unique_ptr<RecordFunctor> end_record_;
  //! A vector of record operations in the FusionDefintion
  std::vector<std::unique_ptr<RecordFunctor>> recording_;
  //! A vector of state that represents Tensors/Vectors/Scalars
  std::vector<State> recording_state_;

 private:
  //! A ptr to the container used when building the Fusion IR from a definition
  Fusion* fusion_;
  //! A vector of nvFuser Fusion IR TensorViews/Vectors/Scalars for building the
  //! Fusion IR graph.
  //! NOTE: Vectors are represented by a vector<Val*>.  This could
  //! be another child class of Val in the IR, similar to TensorView.
  std::vector<std::vector<Val*>> fusion_state_;
  //! The number of states in Fusion Container
  //! A sum of all outputs for each RecordFunctor
  size_t num_recording_states_;
};

} // namespace nvfuser::python_frontend

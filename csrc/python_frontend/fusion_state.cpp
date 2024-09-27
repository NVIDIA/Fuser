// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <python_frontend/fusion_record.h>
#include <python_frontend/fusion_state.h>
#include <utils.h>

// Require namespace for perf scope instrumentation
using namespace nvfuser::inst;

namespace nvfuser::python_frontend {

bool State::inlineDef() const {
  return inline_def_record_.has_value();
}
void State::setInlineDefRecord(const RecordFunctor* record) {
  NVF_CHECK(
      record, "Attemped to set the record for an inline definition as Null!");
  inline_def_record_ = std::optional<const RecordFunctor*>(record);
}
const RecordFunctor* State::inlineDefRecord() const {
  NVF_CHECK(
      inlineDef(),
      "Attempting to query the inline definition Record State that is not inline defined!");
  NVF_CHECK(inline_def_record_.value(), "Inline definition Record is Null!");
  return inline_def_record_.value();
}

bool State::operator==(const State& other) const {
  NVF_ERROR(
      (index == other.index ? (stype == other.stype) : true),
      "State indices should not match with different State Types!");
  return (index == other.index) && (stype == other.stype);
}

bool State::operator!=(const State& other) const {
  NVF_ERROR(
      (index == other.index ? (stype == other.stype) : true),
      "State indices should not match with different State Types!");
  return (index != other.index) || (stype != other.stype);
}

// Generalized printing of State
std::ostream& operator<<(std::ostream& os, const State& state) {
  if (state.inlineDef()) {
    NVF_CHECK(
        state.inlineDefRecord()->inlineDef(),
        "The State Object's definition record is not set with an inline definition!");
    state.inlineDefRecord()->print(os);
  } else {
    if (state.stype == serde::StateType::Scalar) {
      os << "S" << state.index;
    } else if (state.stype == serde::StateType::Tensor) {
      os << "T" << state.index;
    } else if (state.stype == serde::StateType::Vector) {
      os << "V" << state.index;
    } else if (state.stype == serde::StateType::None) {
      os << "None";
    } else {
      NVF_THROW("Unsupported StateType");
    }
  }
  return os;
}

FusionState::FusionState()
    : end_record_(new EndRecord()),
      recording_(),
      recording_state_(),
      fusion_(nullptr),
      fusion_state_(),
      num_recording_states_(0) {}

std::unique_ptr<FusionState> FusionState::clone() {
  auto state = std::make_unique<FusionState>();
  for (auto&& rf : recording_) {
    state->recording_.emplace_back(rf->clone());
  }
  state->fusion_ = fusion_;
  state->fusion_state_.insert(
      state->fusion_state_.end(), fusion_state_.begin(), fusion_state_.end());
  state->num_recording_states_ = num_recording_states_;
  return state;
}

void FusionState::buildFusionIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("FusionContainer::buildFusionIr");
  NVF_CHECK(fusion != nullptr, "Fusion is undefined.");
  resetFusionState(fusion, num_recording_states_);
  auto fusion_guard = FusionGuard(fusion);
  for (auto& record : recording_) {
    auto functor = record.get();
    (*functor)(*this);
  }
}

void FusionState::addRecord(RecordFunctor* record) {
  FUSER_PERF_SCOPE("FusionContainer::addRecord");
  recording_.emplace_back(record);
  num_recording_states_ += record->numOutputs();
  RecordFunctor* state_record = recording_.back().get();

  // NOTE: when the outputs are added to the Record constructor,
  // the Record is not constructed.  Therefore, the information has to be
  // propagated when the Record is added to the FusionState.
  for (const auto& out : state_record->outputs()) {
    if (state_record->inlineDef()) {
      NVF_CHECK(
          out.index < recording_state_.size(),
          "Output state is not found in recording_state! Index: ",
          out.index,
          " Size: ",
          recording_state_.size());
      recording_state_.at(out.index).setInlineDefRecord(state_record);
    }
  }
}

Fusion* FusionState::fusion() {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  return fusion_;
}

void FusionState::printIr() const {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->printMath();
}

void FusionState::resetFusionState(Fusion* fusion, size_t size) {
  NVF_CHECK(fusion != nullptr, "Fusion is undefined.");
  fusion_ = fusion;
  fusion_state_.clear();
  fusion_state_.resize(size, {});
}

void FusionState::addFusionState(Val* val) {
  fusion_state_.push_back({val});
}

void FusionState::addFusionStateVector(std::vector<Val*> val) {
  for (auto v : val) {
    NVF_CHECK(
        !v->isA<TensorView>(),
        "TensorViews should not be added to State Vectors!");
  }
  fusion_state_.push_back(val);
}

Val* FusionState::getFusionState(size_t index) const {
  const auto& ret = fusion_state_.at(index);
  NVF_CHECK(ret.size() == 1, "Expecting to return only one Val*.");
  return ret.front();
}

const std::vector<Val*>& FusionState::getFusionStateVector(size_t index) const {
  return fusion_state_.at(index);
}

size_t FusionState::numFusionStates() const {
  return fusion_state_.size();
}

void FusionState::setFusionState(size_t index, Val* val) {
  fusion_state_.at(index) = {val};
}

void FusionState::setFusionStateVector(size_t index, std::vector<Val*> val) {
  for (auto v : val) {
    NVF_CHECK(
        !v->isA<TensorView>(),
        "TensorViews should not be added to State Vectors!");
  }
  fusion_state_.at(index) = {val};
}

void FusionState::addInput(Val* input) {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->addInput(input);
}

void FusionState::addOutput(Val* output) {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->addOutput(output);
}

void FusionState::aliasOutputToInput(Val* output, Val* input) {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  // We haven't exposed AllocationType to Python API. For now, use
  // ReuseBuffer to preserve the old behavior.
  fusion_->aliasOutputToInput(output, input, AllocationType::ReuseBuffer);
}

} // namespace nvfuser::python_frontend

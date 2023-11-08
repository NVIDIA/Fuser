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
  if (state.stype == serde::StateType_Scalar) {
    os << "S";
  } else if (state.stype == serde::StateType_Tensor) {
    os << "T";
  } else if (state.stype == serde::StateType_Vector) {
    os << "V";
  } else if (state.stype == serde::StateType_None) {
    os << "None";
  } else {
    NVF_ERROR(false, "Unsupported StateType");
  }
  os << state.index;
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
  return ret[0];
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

void FusionState::addOutput(
    Val* output,
    const std::vector<int64_t>& permutation) {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->addOutput(output);
  fusion_->setPermutationOnOutput(
      (int)fusion_->outputs().size() - 1, permutation);
}

void FusionState::aliasOutputToInput(Val* output, Val* input) {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  // We haven't exposed AliasType to Python API. For now, use
  // InplaceUpdate to preserve the old behavior.
  fusion_->aliasOutputToInput(output, input, AliasType::InplaceUpdate);
}

} // namespace nvfuser::python_frontend

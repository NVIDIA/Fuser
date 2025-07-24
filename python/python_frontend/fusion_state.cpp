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
      "Attempting to query the inline definition Record State that is not "
      "inline defined!");
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
        "The State Object's definition record is not set with an inline "
        "definition!");
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

std::vector<Val*> getExtents(Fusion* fusion) {
  NVF_CHECK(fusion != nullptr, "Fusion is undefined.");

  std::vector<Val*> extents;
  for (Val* v : fusion->inputs()) {
    // short-circuit: skip if not TensorView
    if (!v->isA<TensorView>()) {
      continue;
    }
    TensorView* tv = v->as<TensorView>();
    std::vector<IterDomain*> logical_dom =
        TensorDomain::noReductions(tv->getLogicalDomain());
    std::transform(
        logical_dom.begin(),
        logical_dom.end(),
        std::back_inserter(extents),
        [](IterDomain* id) { return id->getMaybeExpandedExtent(); });
  }
  return extents;
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
  std::copy(
      inputs_fid_.begin(),
      inputs_fid_.end(),
      std::back_inserter(state->inputs_fid_));
  std::copy(
      outputs_fid_.begin(),
      outputs_fid_.end(),
      std::back_inserter(state->outputs_fid_));
  std::copy(
      extents_fid_.begin(),
      extents_fid_.end(),
      std::back_inserter(state->extents_fid_));
  std::copy(
      map_value_to_fid_.begin(),
      map_value_to_fid_.end(),
      std::inserter(state->map_value_to_fid_, state->map_value_to_fid_.end()));
  return state;
}

void FusionState::buildFusionIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("FusionContainer::buildFusionIr");
  NVF_CHECK(fusion != nullptr, "Fusion is undefined.");
  resetFusionState(fusion, num_recording_states_);
  auto fusion_guard = FusionGuard(fusion);
  for (auto& record : recording_) {
    auto functor = record.get();
    try {
      (*functor)(*this);
    } catch (const std::exception& e) {
      std::stringstream ss;
      record->print(ss);

      NVF_THROW(
          "\nDetected exception while building Fusion Ir. The failing "
          "RecordFunctor is: ",
          ss.str(),
          "\nNvFuser error message is: ",
          e.what());
    }
  }
  addExtents();
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
  inputs_fid_.clear();
  outputs_fid_.clear();
  extents_fid_.clear();
  map_value_to_fid_.clear();
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
  map_value_to_fid_.emplace(val, (int64_t)index);
}

void FusionState::setFusionStateVector(size_t index, std::vector<Val*> val) {
  for (auto v : val) {
    NVF_CHECK(
        !v->isA<TensorView>(),
        "TensorViews should not be added to State Vectors!");
  }
  fusion_state_.at(index) = {val};
}

void FusionState::addInput(Val* input, size_t index) {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->addInput(input);
  map_value_to_fid_.emplace(input, (int64_t)index);
  inputs_fid_.push_back((int64_t)index);
}

void FusionState::addOutput(Val* output, size_t index) {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->addOutput(output);
  map_value_to_fid_.emplace(output, (int64_t)index);
  outputs_fid_.push_back((int64_t)index);
}

void FusionState::aliasOutputToInput(Val* output, Val* input) {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  // We haven't exposed AllocationType to Python API. For now, use
  // ReuseBuffer to preserve the old behavior.
  fusion_->aliasOutputToInput(output, input, AllocationType::ReuseBuffer);
}

const std::unordered_map<const Val*, int64_t>& FusionState::getValueMap()
    const {
  return map_value_to_fid_;
}

const std::vector<int64_t>& FusionState::inputs() const {
  return inputs_fid_;
}

const std::vector<int64_t>& FusionState::outputs() const {
  return outputs_fid_;
}

const std::vector<int64_t>& FusionState::extents() const {
  return extents_fid_;
}

void FusionState::addExtents() {
  NVF_CHECK(fusion_ != nullptr, "Fusion is undefined.");

  // The size of the tensor dimensions can be used as an input of the
  // segments. NvFuser does not support returning scalar values. Segmentation
  // must pass those sizes as segment arguments manually.
  std::vector<Val*> extents = getExtents(fusion_);
  for (Val* extent : extents) {
    int64_t num_extents = (int64_t)extents_fid_.size();
    // Use negative numbers to represent extent of iterDomains to avoid conflict
    // with non-negative numbers used for scalars, vectors, and tensors.
    // The extents are ordered based on the order of the fusion's inputs.
    int64_t extent_fid = -num_extents - 1;
    extents_fid_.push_back(extent_fid);
    // The extent can already exist in the fusion. However, since scalars cannot
    // be passed between segments, always overwrited existing fids. The original
    // fusion definition will provide scalar extents.
    map_value_to_fid_[extent] = extent_fid;
  }
}

} // namespace nvfuser::python_frontend

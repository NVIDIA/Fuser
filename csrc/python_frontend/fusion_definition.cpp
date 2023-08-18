// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <instrumentation.h>
#include <options.h>
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_definition.h>
#include <utils.h>

// Require namespace for perf scope instrumentation
using namespace nvfuser::inst;

namespace nvfuser::python_frontend {

const char* dtypeToPyString(PrimDataType t) {
  switch (t) {
    case DataType::Bool:
      return "DataType.Bool";
    case DataType::Double:
      return "DataType.Double";
    case DataType::Float:
      return "DataType.Float";
    case DataType::Half:
      return "DataType.Half";
    case DataType::BFloat16:
      return "DataType.BFloat16";
    case DataType::Int:
      return "DataType.Int";
    case DataType::Int32:
      return "DataType.Int32";
    case DataType::ComplexFloat:
      return "DataType.ComplexFloat";
    case DataType::ComplexDouble:
      return "DataType.ComplexDouble";
    case DataType::Null:
      return "DataType.Null";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
  return nullptr;
}

FusionDefinition::FusionDefinition(std::optional<size_t> id, size_t max_length)
    : FusionState(),
      max_length_(max_length),
      fusion_id_(id),
      fusion_cache_(FusionCache::get()),
      trie_node_(nullptr),
      prev_fusion_(nullptr),
      user_sched_(nullptr),
      ops(this),
      sched(this) {}

FusionCache* FusionDefinition::fusionCache() const {
  TORCH_INTERNAL_ASSERT(
      fusion_cache_ != nullptr, "FusionCache pointer is null!");
  return fusion_cache_;
}

FusionDefinition* FusionDefinition::setupDefinition() {
  TORCH_CHECK(max_length_ > 0, "Can't make a FusionDefinition with 0 records!");
  TORCH_CHECK(!id().has_value(), "Fusion Schedule is already found!");
  trie_node_ = fusionCache()->rootTriePtr();
  return this;
}

void FusionDefinition::finalizeDefinition() {
  FUSER_PERF_SCOPE("FusionDefinition::finalizeDefinition");
  auto child_node = fusionCache()->queryChildren(trie_node_, end_record_.get());
  if (!child_node.has_value()) {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      debug() << "\nFusionDefinition: Terminal Node not found.\n";
    }
    trie_node_ = fusionCache()->createChild(trie_node_, end_record_.get());
    fusion_id_ = std::optional<size_t>(trie_node_->fusion_id);
    TORCH_CHECK(id().has_value(), "Invalid fusion id!");

    if (isDebugDumpEnabled(DebugDumpOption::PythonDefinition)) {
      print(debug());
    }

    buildFusionIr(preschedFusion());

    if (isDebugDumpEnabled(DebugDumpOption::FusionIrPresched)) {
      printIr();
    }
  } else {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      debug() << "\nFusionDefinition: Terminal Node found!\n";
    }
    trie_node_ = child_node.value();
    fusion_id_ = std::optional<size_t>(trie_node_->fusion_id);
  }
}

void FusionDefinition::setupSchedule(const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionDefinition::setupSchedule");
  TORCH_CHECK(id().has_value(), "FusionDefinition definition does not exist!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());
  auto device = getCommonDeviceCUDA(inputs);
  TORCH_CHECK(
      inputs.empty() || device > -1, "Inputs are not all on the same device!");
  TORCH_CHECK(user_sched_ == nullptr, "Expected User Scheduler to be null!");
  user_sched_ = fusionCache()->createUserSchedule(scheds, inputs, device);

  // Building a new Fusion container for scheduling with definition such that
  // the definition's tensor data members refer to the corresponding IR objects
  // needed for scheduling. A simple copy of the container would mean the data
  // members that represent tensors would refer to the IR objects in the
  // original and not the copy needed for scheduling.
  buildFusionIr(user_sched_->schedule.get());

  // Manually setting the fusion guard as there is not a good way of using a
  // guard in a local scope across the schedule function
  prev_fusion_ = FusionGuard::getCurFusion();
  FusionGuard::setCurFusion(user_sched_->schedule.get());
}

void FusionDefinition::finalizeSchedule(
    const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionDefinition::finalizeSchedule");
  FusionGuard::setCurFusion(prev_fusion_);
  prev_fusion_ = nullptr;

  user_sched_->executor->compileFusion(user_sched_->schedule.get(), inputs);
  user_sched_ = nullptr;
}

void FusionDefinition::print(std::ostream& os) const {
  if (id().has_value()) {
    os << "\ndef nvfuser_fusion_id" << id().value();
  } else {
    os << "\ndef nvfuser_incomplete_fusion";
  }
  os << "(fd : FusionDefinition) -> None :\n";
  os << std::dec;
  for (auto& rec : recording_) {
    os << "    ";
    rec->print(os);
    os << "\n";
  }
  os << std::endl;
}

std::vector<at::Tensor> FusionDefinition::execute(
    const at::ArrayRef<c10::IValue>& inputs,
    bool override_user_schedule,
    std::optional<int8_t> selected_device) const {
  TORCH_CHECK(id().has_value(), "Valid fusion schedule is not available!");

  auto scheds = fusionCache()->queryFusionSchedules(id().value());

  if (!override_user_schedule) {
    auto device = getCommonDeviceCUDA(inputs, selected_device);
    TORCH_CHECK(
        inputs.empty() || device > -1,
        "Inputs are not all on the same device or don't match selection!");
    auto user_sched_id = fusionCache()->queryUserScheduleId(scheds, inputs);
    if (user_sched_id.has_value()) {
      auto& user_sched = fusionCache()->queryUserSchedule(
          scheds, user_sched_id.value(), device);
      scheds->last_user_def_scheduled_ir = user_sched.schedule.get();
      scheds->last_user_def_executor = user_sched.executor.get();
      return user_sched.executor->runFusion(inputs);
    }
  }

  return scheds->auto_gen_schedules->runFusionWithInputs(
      inputs, std::nullopt, selected_device);
}

std::string FusionDefinition::fusionIr() {
  TORCH_CHECK(id().has_value(), "Invalid fusion definition!");
  std::stringstream ss;
  preschedFusion()->print(ss, false);
  return ss.str();
}

std::string FusionDefinition::lastCudaCode(
    bool intrinsic_code,
    bool override_user_schedule) const {
  std::string result;
  TORCH_CHECK(id().has_value(), "Invalid fusion definition!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());
  auto user_exec = scheds->last_user_def_executor;

  if (!override_user_schedule && (user_exec != nullptr)) {
    if (intrinsic_code) {
      result = user_exec->getStructuredCode(
          user_exec->kernelString(), user_exec->kernel()->indexType());
    } else {
      result = user_exec->kernelString();
    }
  } else {
    result = scheds->auto_gen_schedules->getMostRecentCode(intrinsic_code);
  }
  return result;
}

std::string FusionDefinition::cudaCodeFor(
    const at::ArrayRef<c10::IValue>& inputs,
    bool intrinsic_code,
    bool override_user_schedule) const {
  TORCH_CHECK(id().has_value(), "Invalid fusion definition!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());

  if (!override_user_schedule) {
    auto device = getCommonDeviceCUDA(inputs);
    TORCH_CHECK(
        inputs.empty() || device > -1,
        "Inputs are not all on the same device!");
    auto user_sched_id = fusionCache()->queryUserScheduleId(scheds, inputs);
    if (user_sched_id.has_value()) {
      auto& user_sched = fusionCache()->queryUserSchedule(
          scheds, user_sched_id.value(), device);
      auto user_exec = user_sched.executor.get();
      if (intrinsic_code) {
        return user_exec->getStructuredCode(
            user_exec->kernelString(), user_exec->kernel()->indexType());
      } else {
        return user_exec->kernelString();
      }
    }
  }
  return scheds->auto_gen_schedules->getCodeFor(inputs, intrinsic_code);
}

std::string FusionDefinition::lastScheduledFusionIr(
    bool tensor_transforms,
    bool override_user_schedule) const {
  std::string result;
  TORCH_CHECK(id().has_value(), "Invalid fusion definition!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());
  auto user_sched_ir = scheds->last_user_def_scheduled_ir;

  if (!override_user_schedule && (user_sched_ir != nullptr)) {
    std::stringstream ss;
    user_sched_ir->print(ss, tensor_transforms);
    result = ss.str();
  } else {
    result =
        scheds->auto_gen_schedules->getMostRecentScheduledIr(tensor_transforms);
  }
  return result;
}

std::string FusionDefinition::scheduledFusionIrFor(
    const at::ArrayRef<c10::IValue>& inputs,
    bool tensor_transforms,
    bool override_user_schedule) const {
  TORCH_CHECK(id().has_value(), "Invalid fusion definition!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());

  if (!override_user_schedule) {
    auto device = getCommonDeviceCUDA(inputs);
    TORCH_CHECK(
        inputs.empty() || device > -1,
        "Inputs are not all on the same device!");
    auto user_sched_id = fusionCache()->queryUserScheduleId(scheds, inputs);
    if (user_sched_id.has_value()) {
      auto& user_sched = fusionCache()->queryUserSchedule(
          scheds, user_sched_id.value(), device);
      auto user_sched_ir = user_sched.schedule.get();
      std::stringstream ss;
      user_sched_ir->print(ss, tensor_transforms);
      return ss.str();
    }
  }
  return scheds->auto_gen_schedules->getScheduledIrFor(
      inputs, tensor_transforms);
}

std::optional<size_t> FusionDefinition::id() const {
  return fusion_id_;
}

Scalar FusionDefinition::defineScalar() {
  FUSER_PERF_SCOPE("FusionDefinition::defineScalar");
  Scalar out(recording_state_.size(), this);
  recording_state_.emplace_back(out(), serde::StateType_Scalar);
  return out;
}

Tensor FusionDefinition::defineTensor(size_t dims) {
  FUSER_PERF_SCOPE("FusionDefinition::defineTensor");
  Tensor out(recording_state_.size(), dims, this);
  recording_state_.emplace_back(out(), serde::StateType_Tensor);
  return out;
}

Vector FusionDefinition::defineVector(size_t size) {
  FUSER_PERF_SCOPE("FusionDefinition::defineVector");
  Vector out(recording_state_.size(), size, this);
  recording_state_.emplace_back(out(), serde::StateType_Vector);
  return out;
}

void FusionDefinition::defineRecord(RecordFunctor* record) {
  FUSER_PERF_SCOPE("FusionDefinition::defineRecord");
  TORCH_CHECK(
      (recording_.size() + 1) <= max_length_,
      "The fusion definition has exceeded ",
      max_length_,
      "operations.  The max_length for FusionDefintion's might need to be ",
      "increased if the definition is created as expected.");
  addRecord(record);
  auto child_node =
      fusionCache()->queryChildren(trie_node_, recording_.back().get());
  // If the Record is found in the cache, the FusionDefinition and the Cache
  // will not share Record given the Record had to be created in order to
  // match it but it also already existed in the cache.
  if (child_node.has_value()) {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      debug() << "\nFusionDefinition: Record (hash: 0x" << std::hex
              << record->hash() << ") hit in Fusion Cache.\n";
    }
    trie_node_ = child_node.value();
    // The FusionDefinition and the Cache will share the Record
  } else {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      debug() << "\nFusionDefinition: Record (hash: 0x" << std::hex
              << record->hash() << ") missed in Fusion Cache.\n";
    }
    trie_node_ =
        fusionCache()->createChild(trie_node_, recording_.back().get());
  }
}

Fusion* FusionDefinition::preschedFusion() {
  TORCH_CHECK(
      fusion_id_.has_value(),
      "FusionDefinition does not contain a definition, yet!");
  return fusionCache()
      ->queryFusionSchedules(fusion_id_.value())
      ->preschedFusion();
}

void FusionDefinition::printMathIr() {
  return preschedFusion()->printMath();
}

State FusionDefinition::recordingState(size_t index) const {
  return recording_state_.at(index);
}

} // namespace nvfuser::python_frontend

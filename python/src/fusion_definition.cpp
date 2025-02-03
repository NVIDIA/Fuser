// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion_definition.h>
#include <instrumentation.h>
#include <options.h>
#include <utils.h>

// Require namespace for perf scope instrumentation
using namespace nvfuser::inst;

namespace nvfuser::python {

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
    case DataType::Float8_e4m3fn:
      return "DataType.Float8_e4m3fn";
    case DataType::Float8_e5m2:
      return "DataType.Float8_e5m2";
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
  NVF_THROW("No string found for data type.");
  return nullptr;
}

FusionDefinition::FusionDefinition(std::optional<size_t> id, size_t max_length)
    : FusionState(),
      prev_fusion_(nullptr),
      user_sched_(nullptr),
      ops(this),
      sched(this) {}

FusionDefinition* FusionDefinition::enter() {
  FUSER_PERF_SCOPE("FusionDefinition::enter");
  return this;
}

void FusionDefinition::exit() {
  FUSER_PERF_SCOPE("FusionDefinition::exit");
}

std::vector<DistributedTensor> FusionDefinition::execute(
    const at::ArrayRef<c10::IValue>& inputs,
    std::optional<int8_t> selected_device,
    bool override_user_schedule,
    bool capture_debug_output,
    bool profile,
    std::vector<std::string> _enable_options,
    std::vector<std::string> _disable_options) const {
  // Convert `at::Tensor`s to `DistributedTensor`s.
  std::vector<DistributedTensor> out_dtensors;
  return out_dtensors;
}

} // namespace nvfuser::python

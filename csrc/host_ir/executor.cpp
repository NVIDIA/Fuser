// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/executor.h>
#include <ir/utils.h>

namespace nvfuser {

namespace hir {

HostIrExecutor::HostIrExecutor(
    std::unique_ptr<HostIrContainer> container,
    Communicator* communicator,
    HostIrExecutorParams params)
    : container_(std::move(container)),
      communicator_(communicator),
      params_(params) {}

std::vector<at::Tensor> HostIrExecutor::runWithInput(
    std::unordered_map<Val*, c10::IValue> val_to_IValue) {
  // process input values
  val_to_IValue_ = std::move(val_to_IValue);

  // Interpret each instruction in an "eager" way by iterate over the Host Ir
  // Container's top level expression list
  for (auto expr : container_->topLevelExprs()) {
    dispatch(expr);
  }

  // Collect global outputs
  std::vector<at::Tensor> outputs;
  for (auto output_val : container_->outputs()) {
    auto output = val_to_IValue_.at(output_val).toTensor();
    outputs.push_back(output);
  }

  return outputs;
}

void HostIrExecutor::handle(SetCurrentStream* set_current_stream) {
  Stream* stream = set_current_stream->stream();
  if (streams_.find(stream) == streams_.end()) {
    auto i = (communicator_ != nullptr && communicator_->is_available())
        ? communicator_->deviceId()
        : 0;
    streams_.insert(
        {stream,
         c10::cuda::getStreamFromPool(
             /*isHighPriority=*/true, static_cast<c10::DeviceIndex>(i))});
  }
  setCurrentCUDAStream(streams_.at(stream));
}

void HostIrExecutor::handle(PostOnStream* post_ir) {
  std::vector<c10::IValue> input_IValues;
  for (auto& input : post_ir->inputs()) {
    NVF_ERROR(
        val_to_IValue_.find(input) != val_to_IValue_.end(),
        "No buffer associated with Val ",
        input,
        " for handling ",
        post_ir->toString());
    input_IValues.push_back(val_to_IValue_.at(input));
  }

  // placeholder for storing the outputs
  std::vector<at::Tensor> outputs;

  NVF_ERROR(
      post_ir->hostOpToPost()->isA<HostUnit>(),
      "op must be a HostUnit: ",
      post_ir->hostOpToPost());
  auto hu = post_ir->hostOpToPost()->as<HostUnit>();
  // Compile the fusion and execute it with FusionExecutor(Cache)
  // Check if the executor has been cached. If not, create and cache it
  if (params_.use_fusion_executor_cache) {
    if (!fec_.count(hu)) {
      fec_.try_emplace(
          hu,
          std::make_unique<Fusion>(*hu->fusion_to_execute()),
          0,
          !params_.skip_auto_scheduling);
    }
    outputs = fec_.at(hu).runFusionWithInputs(input_IValues);
  } else {
    auto [it, has_emplaced] = fe_.try_emplace(hu);
    auto& fe = it->second;
    if (has_emplaced) {
      fe.compileFusion(hu->fusion_to_execute(), input_IValues);
    }
    outputs = fe.runFusion(input_IValues);
    if (!params_.cache_fusion_executor) {
      fe_.erase(hu);
    }
  }

  // Store the outputs in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    val_to_IValue_[post_ir->outputs().at(output_idx)] = outputs.at(output_idx);
  }
}

void HostIrExecutor::handle(Communication* communication) {
  NVF_ERROR(
      communicator_ != nullptr && communicator_->is_available(),
      "A valid communicator must be provided");
  NVF_ERROR(
      std::find(
          communication->team().begin(),
          communication->team().end(),
          communicator_->deviceId()) != communication->team().end(),
      "current device index ",
      communicator_->deviceId(),
      " must be present in the communication's team");

  Val* input_val = communication->input(0);
  Val* output_val = communication->output(0);
  at::Tensor input_tensor;
  if (val_to_IValue_.find(input_val) != val_to_IValue_.end()) {
    input_tensor = val_to_IValue_.at(input_val).toTensor();
  }
  at::Tensor output_tensor;
  if (val_to_IValue_.find(output_val) != val_to_IValue_.end()) {
    output_tensor = val_to_IValue_.at(output_val).toTensor();
  }

  c10d::Backend* backend =
      communicator_->getBackendForTeam(communication->team(), std::nullopt);
  works_[communication] = postSingleCommunication(
      communication,
      communicator_->deviceId(),
      backend,
      input_tensor,
      output_tensor);
}

void HostIrExecutor::handle(Wait* wait) {
  Communication* communication = wait->communication();
  NVF_ERROR(works_.find(communication) != works_.end(), "no wait req");
  auto& work = works_.at(communication);
  if (work != nullptr) {
    work->wait();
  }
  works_.erase(communication);
}

} // namespace hir

} // namespace nvfuser

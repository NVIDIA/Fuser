// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ir/utils.h>
#include <multidevice/executor.h>
#include <multidevice/pipeline.h>

namespace nvfuser {

bool PipelineExecutor::shouldRun(PipelineStage* stage) {
  if (should_run_.find(stage) == should_run_.end()) {
    should_run_.emplace(
        stage, stage->descriptor()->mesh.has(runtime_.comm_.deviceId()));
  }
  return should_run_[stage];
}

void PipelineExecutor::handle(PipelineStage* stage) {
  // get the IValues corresponding to the stage's input
  std::vector<c10::IValue> stage_input_IValues;
  for (auto& input_val : stage->inputs()) {
    stage_input_IValues.push_back(val_to_IValue_[input_val]);
  }

  std::vector<at::Tensor> outputs;

  // Compile the stage and either execute it or allocate output buffers
  // if the stage is configured to be autoscheduled, use FusionExecutorCache,
  // otherwise use FusionExecutor
  if (stage->descriptor()->auto_schedule) {
    // Check if the executor has been cached. If not, create and cache it
    if (fec_.find(stage) == fec_.end()) {
      fec_.emplace(
          stage,
          std::make_unique<FusionExecutorCache>(
              runtime_.pipeline_->stageToFusion(stage)));
    }
    // Run the stage to get concrete outputs or placeholders
    // TODO: reimplement allocOutputSpace
    // TODO: allocate output space only if strictly necessary,
    //       and move the allocation to
    //       PipelineExecutor::handle(PipelineCommunication*)
    outputs = shouldRun(stage)
        ? fec_[stage]->runFusionWithInputs(stage_input_IValues)
        : fec_[stage]->allocOutputSpace(stage_input_IValues);

  } else {
    // Check if the executor has been cached. If not, create and cache it
    if (fe_.find(stage) == fe_.end()) {
      fe_.emplace(stage, std::make_unique<FusionExecutor>());
      fe_[stage]->compileFusion(
          runtime_.pipeline_->stageToFusion(stage).get(), stage_input_IValues);
    }
    // Run the stage to get concrete outputs or placeholders
    outputs = shouldRun(stage)
        ? fe_[stage]->runFusion(stage_input_IValues)
        : fe_[stage]->allocOutputSpace(stage_input_IValues);
  }

  // Store the outputs or placeholders in the context
  for (auto output_idx : c10::irange(stage->outputs().size())) {
    val_to_IValue_[stage->outputs().at(output_idx)] = outputs.at(output_idx);
  }
}

struct SendRecvDescriptor {
  Team team;
  DeviceIdxType root = 0;
};

void PipelineExecutor::handle(PipelineCommunication* c) {
  /* Lower the communication into several SendRecvDescriptor
     The idea is to evenly split the destinations accross the sources
     TODO: ensure that the srcs send to the receivers that are the closest in
     the topology. */
  std::vector<SendRecvDescriptor> communications;
  {
    Team senders;
    for (auto& d_id :
         c->in()->as<PipelineVal>()->getStage()->descriptor()->mesh.vector()) {
      senders.push_back(d_id);
    }

    Team receivers;
    for (auto& d_id :
         c->out()->as<PipelineVal>()->getStage()->descriptor()->mesh.vector()) {
      receivers.push_back(d_id);
    }

    auto nbr_srcs = senders.size();
    auto nbr_dests_per_comm = receivers.size() / nbr_srcs;
    auto remainder = receivers.size() % nbr_srcs;
    auto j = 0;
    for (size_t i : c10::irange(nbr_srcs)) {
      SendRecvDescriptor communication;
      auto src = senders.at(i);
      communication.team = {src};
      communication.root = src;
      for (size_t counter = 0; counter < nbr_dests_per_comm + (i < remainder);
           counter++, j++) {
        auto dst = receivers.at(j);
        communication.team.push_back(dst);
      }
      communications.push_back(communication);
    }
  }

  auto input_val = c->in();
  auto output_val = c->out();
  std::vector<at::Tensor> tensor = {val_to_IValue_.at(input_val).toTensor()};

  /* perform the needed communications. For now everything is translated as
     send/recv.
     TODO: sending from one src to multiple dsts could be lowered as a
     broadcast, in which case we should create a new communictor backend (and
     cache it)*/
  for (auto& communication : communications) {
    auto sender_rank = communication.root;
    for (auto receiver_rank : communication.team) {
      if ((sender_rank == receiver_rank) ||
          !(runtime_.comm_.deviceId() == sender_rank ||
            runtime_.comm_.deviceId() == receiver_rank)) {
        continue;
      }
      runtime_.comm_.sendRecv(receiver_rank, sender_rank, tensor);
    }
  }
  val_to_IValue_[output_val] = (c10::IValue)(tensor[0]);
}

std::vector<at::Tensor> PipelineExecutor::runWithInput(
    const std::vector<c10::IValue>& inputs) {
  // Make sure inputs align at global boundary.
  NVF_ERROR(
      inputs.size() == runtime_.pipeline_->inputs().size(),
      "Wrong number of inputs");

  // process input values input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue_[runtime_.pipeline_->inputs().at(input_idx)] =
        inputs.at(input_idx);
  }

  // Run through the stages to launch kernel
  traverseTo(runtime_.pipeline_, runtime_.pipeline_->outputs());

  // Collect global outputs from context
  std::vector<at::Tensor> outputs;
  for (auto output_val : runtime_.pipeline_->outputs()) {
    outputs.push_back(val_to_IValue_[output_val].toTensor());
  }

  return outputs;
}

} // namespace nvfuser

#endif

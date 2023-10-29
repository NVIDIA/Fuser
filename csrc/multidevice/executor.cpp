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
#include <multidevice/lower_communication.h>
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
    // TODO: allocate output space only if strictly necessary
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

void PipelineExecutor::handle(PipelineCommunication* c) {
  at::Tensor input_tensor =
      val_to_IValue_.at(c->in())
          .toTensor();

  /* Allocation of output buffer.
    TODO: revise to avoid garbage allocation. Indeed, for now we use the same buffer
    for the input and output of the Communication. The input has always
    been allocated previously since we systematically allocate the output of every
    PipelineStage. This is valid but induces a lot of garbage allocation:
    1) some PipelineStage's outputs could be ignore on certain devices
    2) some buffers are overallocated, e.g., if the communication pattern is 
       the one of a "Scatter", the destination buffer's size only need to be a fraction
       of the source buffer.
  */ 
  val_to_IValue_[c->out()] = val_to_IValue_.at(c->in());
  at::Tensor output_tensor =
      val_to_IValue_.at(c->out()).toTensor(); // shallow copy

  // Lower the Communication into a vector of Communications
  if (communications_.find(c) == communications_.end()) { // check if cached
    communications_.emplace(
        c,
        lowerCommunication(
            runtime_.comm_.deviceId(), c, input_tensor, output_tensor));
  }
  auto& communications = communications_[c];

  // post and wait communications
  for (auto& communication : communications) {
    auto work = communication->post(runtime_.comm_);
    if (work) {
      work->wait();
    }
  }
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

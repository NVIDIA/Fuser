// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <fusion_segmenter.h>
#include <ir/utils.h>
#include <multidevice/executor.h>

namespace nvfuser {

// Update launch parameters if scheduler needs to set the launch params.
void updateLaunchParamsFromScheduler(
    SchedulerEntry* scheduler,
    LaunchParams& lparams) {
  // Set launch parameters form scheduler.
  if (scheduler->params()->isA<ReductionParams>()) {
    lparams = scheduler->reductionParams().lparams;
  } else {
    TORCH_INTERNAL_ASSERT(scheduler->params()->isA<PointwiseParams>());
    lparams = scheduler->pointwiseParams().lparams;
  }
}

PipelineExecutor::CompiledKernelPtr PipelineExecutor::compileStage(
    PipelineStage* stage,
    std::vector<c10::IValue> stage_inputs) {
  // convert the stage to a Fusion
  auto fusion_from_stage = runtime_.pipeline_->stageToFusion(stage);
  // Placeholder for auto schedule parameters if any.
  c10::optional<SchedulerEntry*> maybe_scheduler_entry = c10::nullopt;

  // Auto schedule if requested
  if (stage->descriptor()->auto_schedule) {
    // Get runtime info from fusion graph and concrete tensor inputs.
    SchedulerRuntimeInfo runtime_info(fusion_from_stage.get(), stage_inputs);

    // Get heuristic tag that applies to the given fusion and input info.
    auto heuristic = SchedulerEntry::proposeHeuristics(
        fusion_from_stage.get(), runtime_info);
    TORCH_INTERNAL_ASSERT(heuristic.has_value(), "cannot auto schedule fusion");

    // Generate scheduler parameters from tag.
    auto scheduler = SchedulerEntry::makeEntry(
        heuristic.value(), fusion_from_stage.get(), runtime_info);

    // Apply schedule to fusion graph.
    scheduler->schedule(fusion_from_stage.get());

    maybe_scheduler_entry = scheduler.get();
    // Cache scheduler in registry to retrieve launch parameters.
    auto_scheduler_registry_[stage] = std::move(scheduler);
  }

  auto executor_ptr = std::make_unique<FusionExecutor>();

  // Infer which device this fusion runs from input device ids.
  const auto device_index = getCommonDeviceCUDA(stage_inputs);
  TORCH_CHECK(device_index >= 0, "All inputs must be on the same device");

  // Set launch parameters
  LaunchParams launch_params;

  // Set compile options
  CompileOptions options;
  options.device = c10::Device(c10::DeviceType::CUDA, device_index);

  auto args = KernelArgumentHolder::createKernelArgumentHolder(stage_inputs);
  // Set parameters inferred by auto scheduler.
  if (maybe_scheduler_entry.has_value()) {
    auto scheduler_entry = maybe_scheduler_entry.value();
    // Set launch parameters with auto scheduler.
    updateLaunchParamsFromScheduler(scheduler_entry, launch_params);
  }

  args.setDeviceIndex(device_index);
  // Lower the fusion and compile the generated kernel.
  executor_ptr->compileFusion(fusion_from_stage.get(), args, launch_params, {});

  return executor_ptr;
}

void PipelineExecutor::handle(PipelineStage* stage) {
  // get the IValues corresponding to the stage's input
  std::vector<c10::IValue> stage_input_IValues;
  for (auto& input_val : stage->inputs()) {
    stage_input_IValues.push_back(val_to_IValue_[input_val]);
  }

  // Run the lowering and compilation step if we haven't compiled yet.
  if (!compiled_kernels_.count(stage)) {
    compiled_kernels_[stage] = compileStage(stage, stage_input_IValues);
  }
  auto& executor = compiled_kernels_[stage];

  // Launch kernel and record the kernel output into current context
  std::vector<at::Tensor> outputs;

  // bool indicating whether the current rank should run the current stage
  bool shouldRun = std::count(
      stage->descriptor()->mesh.deviceIndices().begin(),
      stage->descriptor()->mesh.deviceIndices().end(),
      runtime_.rankToDeviceIdx(runtime_.comm_.rank()));

  if (shouldRun) {
    // Use default launch parameters.
    LaunchParams launch_params;
    // If the kernel was auto-scheduled, we need to
    //  pull the launch parameters from the scheduler.
    auto scheduler_it = auto_scheduler_registry_.find(stage);
    if (scheduler_it != auto_scheduler_registry_.end()) {
      updateLaunchParamsFromScheduler(
          scheduler_it->second.get(), launch_params);
    }
    outputs = executor->runFusion(stage_input_IValues, launch_params, {});
  } else {
    // Allocate space for kernel outputs.
    // TODO: allocate only if strictly necessary
    outputs = executor->allocOutputSpace(stage_input_IValues);
  }
  // Store the outputs or placeholders in the context
  // Bind context tensors to the actual kernel outputs:
  for (auto output_idx : c10::irange(stage->outputs().size())) {
    // Fill tensor data or placeholder to context.
    val_to_IValue_[stage->outputs().at(output_idx)] = outputs.at(output_idx);
  }
}

struct SendRecvDescriptor {
  std::vector<RankType> team;
  RankType root = 0;
};

void PipelineExecutor::handle(PipelineCommunication* c) {
  /* Lower the communication into several SendRecvDescriptor
     The idea is to evenly split the destinations accross the sources
     TODO: ensure that the srcs send to the receivers that are the closest in
     the topology. */
  std::vector<SendRecvDescriptor> communications;
  {
    std::vector<RankType> sender_ranks;
    for (auto& dId : c->in()
                         ->as<PipelineVal>()
                         ->getStage()
                         ->descriptor()
                         ->mesh.deviceIndices()) {
      sender_ranks.push_back(runtime_.deviceIdxToRank(dId));
    }

    std::vector<RankType> receiver_ranks;
    for (auto& dId : c->out()
                         ->as<PipelineVal>()
                         ->getStage()
                         ->descriptor()
                         ->mesh.deviceIndices()) {
      receiver_ranks.push_back(runtime_.deviceIdxToRank(dId));
    }

    auto nbr_srcs = sender_ranks.size();
    auto nbr_dests_per_comm = receiver_ranks.size() / nbr_srcs;
    auto remainder = receiver_ranks.size() % nbr_srcs;
    auto j = 0;
    for (size_t i : c10::irange(nbr_srcs)) {
      SendRecvDescriptor communication;
      auto src = sender_ranks.at(i);
      communication.team = {src};
      communication.root = src;
      for (size_t counter = 0; counter < nbr_dests_per_comm + (i < remainder);
           counter++, j++) {
        auto dst = receiver_ranks.at(j);
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
      runtime_.comm_.sendRecv(receiver_rank, sender_rank, tensor);
    }
  }
  val_to_IValue_[output_val] = (c10::IValue)(tensor[0]);
}

std::vector<at::Tensor> PipelineExecutor::runWithInput(
    const std::vector<c10::IValue>& inputs) {
  // Make sure inputs align at global boundary.
  TORCH_INTERNAL_ASSERT(
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

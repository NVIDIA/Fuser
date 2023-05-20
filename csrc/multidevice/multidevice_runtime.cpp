// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <device_lower/utils.h>
#include <fusion_segmenter.h>
#include <ir/utils.h>
#include <multidevice/multicluster_fusion.h>
#include <multidevice/multidevice_runtime.h>

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

MultiDeviceRuntime::CompiledKernelPtr MultiDeviceRuntime::compileCluster(
    ClusterPtr cluster,
    std::vector<c10::IValue> cluster_inputs) {
  // Make a copy of the fusion graph we want to generate
  //  CUDA kernel and compile.
  auto fusion_from_cluster = cluster->toFusion();
  // Placeholder for auto schedule parameters if any.
  c10::optional<SchedulerEntry*> maybe_scheduler_entry = c10::nullopt;

  // Auto schedule if requested
  if (cluster->params().auto_schedule) {
    // Get runtime info from fusion graph and concrete tensor inputs.
    SchedulerRuntimeInfo runtime_info(
        fusion_from_cluster.get(), cluster_inputs);

    // Get heuristic tag that applies to the given fusion and input info.
    auto heuristic = SchedulerEntry::proposeHeuristics(
        fusion_from_cluster.get(), runtime_info);
    TORCH_INTERNAL_ASSERT(heuristic.has_value(), "cannot auto schedule fusion");

    // Generate scheduler parameters from tag.
    auto scheduler = SchedulerEntry::makeEntry(
        heuristic.value(), fusion_from_cluster.get(), runtime_info);

    // Apply schedule to fusion graph.
    scheduler->schedule(fusion_from_cluster.get());

    maybe_scheduler_entry = scheduler.get();
    // Cache scheduler in registry to retrieve launch parameters.
    auto_scheduler_registry_[cluster] = std::move(scheduler);
  }

  auto executor_ptr = std::make_unique<FusionExecutor>();

  // Infer which device this fusion runs from input device ids.
  // TODO: fix should bind device with cluster?
  const auto device_index = getCommonDeviceCUDA(cluster_inputs);
  TORCH_CHECK(device_index >= 0, "All inputs must be on the same device");

  // Set launch parameters
  LaunchParams launch_params;

  // Set compile options
  CompileOptions options;
  options.device = c10::Device(c10::DeviceType::CUDA, device_index);

  auto args = KernelArgumentHolder::createKernelArgumentHolder(cluster_inputs);
  // Set parameters inferred by auto scheduler.
  if (maybe_scheduler_entry.has_value()) {
    auto scheduler_entry = maybe_scheduler_entry.value();
    // TODO: I don't remember what happened with index mode. Need to follow up.
    // args.setIndexMode(scheduler_entry->indexMode());
    // Set launch parameters with auto scheduler.
    updateLaunchParamsFromScheduler(scheduler_entry, launch_params);
  }

  args.setDeviceIndex(device_index);
  // Lower the fusion and compile the generated kernel.
  executor_ptr->compileFusion(
      fusion_from_cluster.get(), args, launch_params, {});

  return executor_ptr;
}

void MultiDeviceRuntime::handle(SendRecv* sr) {
  auto sender_cluster = sr->in()->getCluster();
  auto receiver_cluster = sr->out()->getCluster();

  // check if current process receives
  bool is_sender = shouldRun(sender_cluster);
  // check if current process sends
  bool is_receiver = shouldRun(receiver_cluster);

  int sender_rank = sender_cluster->params().process_rank;
  int receiver_rank = receiver_cluster->params().process_rank;

  // Val corresponding to the sent tensor
  auto val = sr->in()->getOriginalVal();

  // container for the sent/received tensor
  std::vector<at::Tensor> tensor = {val_to_IValue.at(val).toTensor()};

  if (is_sender) {
    // sending the tensor
    process_cluster_->send(tensor, receiver_rank, 0);
  }
  if (is_receiver) {
    // receiving the tensor
    auto work = process_cluster_->recv(tensor, sender_rank, 0);
    // wait for completion
    while (!work->isCompleted())
      ;
    // store the receive tensor
    val_to_IValue[val] = (c10::IValue)(tensor[0]);
  }
}

void MultiDeviceRuntime::handle(AggregateExpr* aExpr) {
  auto cluster = aExpr->getCluster();

  // get the c10::IValues corresponding to the cluster's input
  std::vector<c10::IValue> cluster_input;
  std::transform(
      cluster->inputs().begin(),
      cluster->inputs().end(),
      std::back_inserter(cluster_input),
      [this](auto input_val) { return val_to_IValue[input_val]; });

  // Run the lowering and compilation step if we haven't compiled yet.
  if (!compiled_kernels_.count(cluster)) {
    compiled_kernels_[cluster] = compileCluster(cluster, cluster_input);
  }
  auto& executor = compiled_kernels_[cluster];

  // Launch kernel and record the kernel output into current context
  std::vector<at::Tensor> outputs;
  if (shouldRun(cluster)) {
    // Use default launch parameters.
    LaunchParams launch_params;
    // If the kernel was auto-scheduled, we need to
    //  pull the launch parameters from the scheduler.
    auto scheduler_it = auto_scheduler_registry_.find(cluster);
    if (scheduler_it != auto_scheduler_registry_.end()) {
      updateLaunchParamsFromScheduler(
          scheduler_it->second.get(), launch_params);
    }
    outputs = executor->runFusion(cluster_input, launch_params, {});
  } else {
    // Allocate space for kernel outputs.
    // TODO: allocate only if necessary
    outputs = executor->allocOutputSpace(cluster_input);
  }

  // Store the outputs or place holders in the context
  // Bind context tensors to the actual kernel outputs:
  for (auto output_idx : c10::irange(cluster->outputs().size())) {
    // Fill tensor data or placeholder to context.
    val_to_IValue[cluster->outputs().vector().at(output_idx)] =
        outputs.at(output_idx);
  }
}

std::vector<at::Tensor> MultiDeviceRuntime::runWithInput(
    std::vector<c10::IValue> inputs) {
  // Make sure inputs align at global boundary.
  TORCH_INTERNAL_ASSERT(
      inputs.size() == a_dag_->MCFusionInputs().size(),
      "Wrong number of inputs");

  // Make initial context with input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue[a_dag_->MCFusionInputs().at(input_idx)] =
        inputs.at(input_idx);
  }

  // Run through the clusters to launch kernel
  traverseTo(a_dag_.get(), a_dag_->outputs());

  // Collect global outputs from context
  std::vector<at::Tensor> outputs;
  std::transform(
      a_dag_->MCFusionOutputs().begin(),
      a_dag_->MCFusionOutputs().end(),
      std::back_inserter(outputs),
      [this](auto output_val) { return val_to_IValue[output_val].toTensor(); });

  // Clear life time of intermediate tensors.
  val_to_IValue.clear();

  return outputs;
}

} // namespace nvfuser

#endif

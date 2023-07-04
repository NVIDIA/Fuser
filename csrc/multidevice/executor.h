// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

// #include <disjoint_set.h>
#include <evaluator_common.h>
#include <executor.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <multidevice/pipeline.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/registry.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include <multidevice/communicator.h>
#include <multidevice/runtime.h>

namespace nvfuser {

// Runtime Executor for Pipelines
// This class inherits from IterVisitor because the execution
// is ordered by the traversal of the Pipeline seen as a DAG
class PipelineExecutor : public IterVisitor {
 public:
  using CompiledKernelPtr = std::unique_ptr<FusionExecutor>;

  explicit PipelineExecutor(MultiDeviceRuntime& runtime)
      : IterVisitor(), runtime_(runtime) {}

  // Run the Pipelined Fusion with the given global inputs
  std::vector<at::Tensor> runWithInput(const std::vector<c10::IValue>& inputs);

 private:
  // Implement the execution of exprs of the Pipeline
  // Each PipelineStage will be compiled and executed on a GPU
  // Each PipelineCommunication will invoke the ProgressStage to perform the
  // comm
  using IterVisitor::handle;
  void handle(PipelineStage* pipelineStage) override;
  void handle(PipelineCommunication* sr) override;

  // Generate and compile cuda kernel corresponding to the given Stage
  CompiledKernelPtr compileStage(
      PipelineStage* stage,
      std::vector<c10::IValue> stage_input);

  // Returns whether the current process should run the stage
  bool shouldRun(PipelineStage* stage);

  // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue_;

  // Compiled kernels from multi_stage_fusion_
  std::unordered_map<PipelineStage*, CompiledKernelPtr> compiled_kernels_;

  // Cache results of shouldRun method
  std::unordered_map<PipelineStage*, bool> should_run_;

  // Keeps track of heuristics that are used to schedule
  //  the auto-scheduled kernels.
  std::unordered_map<PipelineStage*, std::unique_ptr<SchedulerEntry>>
      auto_scheduler_registry_;

  MultiDeviceRuntime& runtime_;
};

} // namespace nvfuser

#endif

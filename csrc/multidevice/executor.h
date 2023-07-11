// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <iter_visitor.h>
#include <kernel_cache.h>
#include <multidevice/pipeline_ir.h>
#include <multidevice/runtime.h>

namespace nvfuser {

// Runtime Executor for Pipelines
// This class inherits from IterVisitor because the execution
// is ordered by the traversal of the Pipeline seen as a DAG
class PipelineExecutor : public IterVisitor {
 public:
  explicit PipelineExecutor(MultiDeviceRuntime& runtime)
      : IterVisitor(), runtime_(runtime) {}

  // Run the Pipelined Fusion with the given global inputs
  std::vector<at::Tensor> runWithInput(const std::vector<c10::IValue>& inputs);

 private:
  // Implement the execution of exprs of the Pipeline
  // Each PipelineStage will be compiled and executed on a GPU
  // Each PipelineCommunication will invoke the communicator's process group
  // to perform the communication
  using IterVisitor::handle;
  void handle(PipelineStage* pipelineStage) override;
  void handle(PipelineCommunication* sr) override;

  // Returns whether the current process should run the stage
  bool shouldRun(PipelineStage* stage);

  // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue_;

  // Stores FusionExecutorCache for each PipelineStage
  std::unordered_map<PipelineStage*, std::unique_ptr<FusionExecutorCache>> fec_;

  // Cache results of shouldRun method
  std::unordered_map<PipelineStage*, bool> should_run_;

  // MultiDeviceRuntime to be executed
  MultiDeviceRuntime& runtime_;
};

} // namespace nvfuser

#endif

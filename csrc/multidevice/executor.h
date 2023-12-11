// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <exceptions.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <multidevice/communication.h>
#include <multidevice/pipeline_ir.h>
#include <multidevice/runtime.h>

namespace nvfuser {

// Runtime Executor for Pipelines
// This class inherits from IterVisitor because the execution
// is ordered by the traversal of the Pipeline seen as a DAG
class PipelineExecutor : public IterVisitor {
 public:
  explicit PipelineExecutor(MultiDeviceRuntime& runtime)
      : runtime_(runtime) {}

  // Run the Pipelined Fusion with the given global inputs
  std::vector<at::Tensor> runWithInput(const std::vector<c10::IValue>& inputs);

 private:
  // Returns whether the current process should run the stage
  bool shouldRun(SegmentedGroup* stage);
  void executeKernel(SegmentedGroup* group);
  void executeCommunication(SegmentedGroup* group);


  // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue_;

  // Stores FusionExecutor(Cache) for each PipelineStage
  std::unordered_map<SegmentedGroup*, std::unique_ptr<FusionExecutor>> fe_;
  std::unordered_map<SegmentedGroup*, std::unique_ptr<Fusion>> fusions_;
  // Stores the resulting Communications after lowering each
  // PipelineCommunication
  std::unordered_map<
      SegmentedGroup*,
      std::vector<std::shared_ptr<Communication>>>
      communications_;

  // Cache results of shouldRun method
  std::unordered_map<SegmentedGroup*, bool> should_run_;

  // MultiDeviceRuntime to be executed
  MultiDeviceRuntime& runtime_;

  RuntimeWorkSpace workspace_;
};

} // namespace nvfuser

#endif

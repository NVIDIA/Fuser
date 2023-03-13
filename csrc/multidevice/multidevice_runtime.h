// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <disjoint_set.h>
#include <evaluator_common.h>
#include <executor.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <multidevice/aggregate_dag.h>
#include <multidevice/multicluster_fusion.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/registry.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <multidevice/ProcessGroupBuilder.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace nvfuser {

// Runtime for multi_cluster_fusion.
// This class inherits from IterVisitor because the runtime executor
// is ordered by the traversal of the aggregate dag
class TORCH_CUDA_CU_API MultiDeviceRuntime : public IterVisitor {
 public:
  using CompiledKernelPtr = std::unique_ptr<FusionExecutor>;

  MultiDeviceRuntime(
      std::unique_ptr<AggregateDag> a_dag,
      c10::intrusive_ptr<c10d::Backend> process_cluster)
      : IterVisitor(),
        process_cluster_(process_cluster),
        a_dag_(std::move(a_dag)) {}

  // Run the multidevice fusion with the given global inputs
  std::vector<at::Tensor> runWithInput(std::vector<c10::IValue> inputs);

 private:
  // Implement the execution of exprs of the AggregateDag
  // Each AggregateExpr will be compiled and executed on a GPU
  // Each SendRecv will invoke the ProgressCluster to perform the comm
  using IterVisitor::handle;
  void handle(AggregateExpr* aExpr) override;
  void handle(SendRecv* sr) override;

  // Check if the current process should run a Cluster
  bool shouldRun(ClusterPtr cluster) {
    return cluster->params().process_rank == process_cluster_->getRank();
  }
  // Generate and compile cuda kernel corresponding to the given Cluster
  CompiledKernelPtr compileCluster(
      ClusterPtr cluster,
      std::vector<c10::IValue> cluster_input);

  // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue;

  // Compiled kernels from multi_cluster_fusion_
  std::unordered_map<ClusterPtr, CompiledKernelPtr> compiled_kernels_;

  // Keeps track of heuristics that are used to schedule
  //  the auto-scheduled kernels.
  std::unordered_map<ClusterPtr, std::unique_ptr<SchedulerEntry>>
      auto_scheduler_registry_;

  // Process cluster. Interface for inter-process collectives
  c10::intrusive_ptr<c10d::Backend> process_cluster_;

  // AggregateDag built from the MultiClusterFusion whose traversal
  // defines the runtime execution.
  std::unique_ptr<AggregateDag> a_dag_;
};

} // namespace nvfuser

#endif

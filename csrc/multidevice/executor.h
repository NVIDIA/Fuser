// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#pragma once

#include <c10/core/DeviceType.h>
#include <exceptions.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

/*
  The MultiDeviceExecutor executes a Fusion on a multi-device setting.
  It is instantiated from a Fusion and a Communicator.

  Here is a summary of the different steps.
  I. At instantiation:
  - resharding "Set" exprs are automatically inserted in the fusion where a
    network communication is needed. See the function insertReshardings.
  - the Fusion is segmented into segments which can be of two types:
      1) compute segments, composed of non-Resharding expressions only,
         that can be purely execute on a single device
      or
      2) communication, composed of exactly one resharding expression, which
         can be either a "Set" or "Reduce" Exprs.
  - the runtime order of execution of the different segments is computed in
    prepareRuntimeOrder

  II. At runtime, through the method runWithInput:
  - allocateRecvBuffers allocates on each device the necessary buffers to
    store the data received from network communications
  - Each (compute or comm) segment is executed separately, in order:
    1) each compute segment is transformed into a fusion, compiled and executed
       on a single device, see postKernel
    2) each comm segment is lowered into a series of communications (defined in
       multidevice/communications.h) and are posted on the stream.
       "Wait" primitives are also posted on the stream.

  Later, the MultiDeviceExecutor should be integrated into FusionExecutorCache.
  Also, the steps described above should be divided into compilation,
  allocation, runtime etc.
  This will be done along the way when we will have a better symbolic
  representation of the multidevice module 
*/
class MultiDeviceExecutor {
 public:
  MultiDeviceExecutor(std::unique_ptr<Fusion> fusion, Communicator& comm);

  // Run the fusion on several devices with the given global inputs
  std::vector<at::Tensor> runWithInput(const std::vector<c10::IValue>& inputs);

  // Returns the Communicator
  Communicator* comm() const {
    return &comm_;
  }

  // Returns the Fusion
  auto completeFusion() const {
    return staged_fusion_->completeFusion();
  }

  // check if the runtime is valid returns an error msg.
  // An empty message means that the runtime is valid
  std::string validate() const;

 private:
  // execute locally a SegmentedGroup that does not involve inter-device
  // communication
  void postKernel(SegmentedGroup* group);
  // execute a SegmentedGroup representing inter-device communication
  void postCommunication(SegmentedGroup* group);

  // allocate inter-device communication recv buffers
  std::unordered_map<Val*, c10::IValue> allocateRecvBuffers(
      std::vector<c10::IValue> global_inputs_IValues);

  // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue_;

  // holds the Communicator to be used for execution
  Communicator& comm_;
  // holds the fusion after segmentation at the inter-device communications
  // Each SegmentedGroup represents a pipeline's stage, and can be either
  // 1) a Fusion which doesn't involve inter-device communication
  // 2) a Fusion comprised of one Expr, representing inter-device communication
  std::unique_ptr<SegmentedFusion> staged_fusion_;
  // Stores the order in which the pipeline's stage should be executed
  RuntimeWorkSpace workspace;
  // Cache Fusions, FusionExecutors, and Communications
  std::unordered_map<SegmentedGroup*, std::unique_ptr<FusionExecutor>> fe_;
  std::unordered_map<SegmentedGroup*, std::unique_ptr<Fusion>> fusions_;
  // Cache whether a SegmentedGroup should be run by the current device
  std::unordered_map<SegmentedGroup*, bool> should_run_;
  // Cache whether a SegmentedGroup requires inter-device communication
  std::unordered_map<SegmentedGroup*, bool> is_resharding_;
};

} // namespace nvfuser

#endif

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <c10/core/DeviceType.h>
#include <exceptions.h>
#include <multidevice/communicator.h>
#include <multidevice/communication.h>
#include <multidevice/multidevice.h>
#include <fusion_segmenter.h>
#include <fusion.h>

namespace nvfuser {

/*
  The MultiDeviceExecutor class gather all what is needed for executing a
  Pipeline on a multi-device setting. It is instantiated from a Pipeline and a
  Communicator (a default Communicator is built at initialization if none is
  provided).
*/
class MultiDeviceExecutor {
 public:
  MultiDeviceExecutor(std::unique_ptr<Fusion> fusion, Communicator& comm);

  // Run the multidevice fusion with the given global inputs
  std::vector<at::Tensor> runWithInput(const std::vector<c10::IValue>& inputs);

  // Returns the Communicator
  Communicator* comm() const {
    return &comm_;
  }

  // Returns the current device index
  DeviceIdxType dId() const {
    return comm_.deviceId();
  }

  // Returns the Fusion
  auto fusion() const {
    return pipeline_->completeFusion();
  }

  // Returns the Pipeline
  auto pipeline() const {
    return pipeline_.get();
  }

  bool isResharding(SegmentedGroup* group) const {
    return is_resharding_.at(group);
  }

  bool shouldRun(SegmentedGroup* group) const {
    return should_run_.at(group);
  }

  // check if the runtime is valid returns an error msg.
  // An empty message means that the runtime is valid
  std::string validate() const;

 private:
  std::unique_ptr<SegmentedFusion> pipeline_;
  Communicator& comm_;

  // Returns whether the current process should run the stage
  void postKernel(SegmentedGroup* group);
  void postCommunication(SegmentedGroup* group);

  std::unordered_map<Val*, c10::IValue> allocateRecvBuffers(std::vector<c10::IValue> global_inputs_IValues);

  // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue_;

  // Stores Fusions and FusionExecutors
  std::unordered_map<SegmentedGroup*, std::unique_ptr<FusionExecutor>> fe_;
  std::unordered_map<SegmentedGroup*, std::unique_ptr<Fusion>> fusions_;
  // Stores the resulting Communications after lowering each
  // resharding segmented group
  std::unordered_map<
      SegmentedGroup*,
      std::vector<std::shared_ptr<Communication>>>
      communications_;

  // indicate whether a SegmentedGroup should be run by the current device
  std::unordered_map<SegmentedGroup*, bool> should_run_;
  // indicate whether a SegmentedGroup requires inter-device communication
  std::map<SegmentedGroup*, bool> is_resharding_;
};

} // namespace nvfuser

#endif

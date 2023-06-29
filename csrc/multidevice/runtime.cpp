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
#include <multidevice/executor.h>
#include <multidevice/runtime.h>

namespace nvfuser {

std::vector<at::Tensor> MultiDeviceRuntime::runWithInput(
    std::vector<c10::IValue> inputs) {
  PipelineExecutor executor(*this);
  return executor.runWithInput(inputs);
}

void MultiDeviceRuntime::validate() const {
  // stores all the device indices present in the pipeline accross all stages
  std::unordered_set<DeviceIdxType> device_indices;
  for (auto& stage_desc : pipeline_->descriptor().stageDescriptors) {
    for (auto dId : stage_desc->mesh.deviceIndices()) {
      device_indices.insert(dId);
    }
  }

  // Gather all the device indices covered by the communicator
  std::unordered_set<DeviceIdxType> device_indices_in_communicator;
  for (RankType rank : comm_.ranks()) {
    device_indices_in_communicator.insert(rankToDeviceIdx(rank));
  }

  // Checks if all the devices indices involved in the pipeline are
  // associated with a rank in the communicator
  for (auto dId : device_indices) {
    TORCH_INTERNAL_ASSERT(
        device_indices_in_communicator.count(dId),
        "device index " + std::to_string(dId) +
            " is present in the pipeline but no process in the communicator runs it");
  }

  // Checks that the device index of the current process corresponds to a valid
  // concrete device (if invovled in the pipeline)
  auto currentDeviceIdx = rankToDeviceIdx(comm_.rank());
  if (device_indices.count(currentDeviceIdx)) {
    TORCH_INTERNAL_ASSERT(
        true, // TODO
        "device index " + deviceIdxToDevice(currentDeviceIdx).str() +
            " is present in the pipeline but does not correspond to a valid cuda device");
  }
}

} // namespace nvfuser

#endif

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
  for (auto& stage_desc : pipeline_->descriptor().stage_descriptors) {
    for (auto d_id : stage_desc.mesh.deviceIndices()) {
      device_indices.insert(d_id);
    }
  }

  // Gather all the device indices covered by the communicator
  std::unordered_set<DeviceIdxType> device_indices_in_communicator;
  for (RankType rank : comm_.ranks()) {
    device_indices_in_communicator.insert(rankToDeviceIdx(rank));
  }

  // Checks if all the devices indices involved in the pipeline are
  // associated with a rank in the communicator
  for (auto d_id : device_indices) {
    TORCH_INTERNAL_ASSERT(
        device_indices_in_communicator.count(d_id),
        "device index " + std::to_string(d_id) +
            " is present in the pipeline but no process in the communicator runs it");
  }

  // Checks that the device index of the current process corresponds to a valid
  // concrete device (if invovled in the pipeline)
  auto current_device_idx = rankToDeviceIdx(comm_.rank());
  if (device_indices.count(current_device_idx)) {
    TORCH_INTERNAL_ASSERT(
        true, // TODO
        "device index " + deviceIdxToDevice(current_device_idx).str() +
            " is present in the pipeline but does not correspond to a valid cuda device");
  }
}

} // namespace nvfuser

#endif

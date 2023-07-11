// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ATen/cuda/CUDAContext.h>
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

  // Checks if all the devices indices involved in the pipeline are
  // associated with a rank in the communicator
  for (auto d_id : device_indices) {
    TORCH_INTERNAL_ASSERT(
        d_id < comm_.size(),
        "device index " + std::to_string(d_id) +
            " is present in the pipeline but no process in the communicator runs it");
  }

  // Checks that the number of processes within the node is less or equal
  // to the number of available GPUs.
  TORCH_INTERNAL_ASSERT(
      comm_.local_size() <= at::cuda::getNumGPUs(),
      std::to_string(comm_.local_size()) + " processes are spawn but only " +
          std::to_string(at::cuda::getNumGPUs()) + " GPUs are available");
}

} // namespace nvfuser

#endif

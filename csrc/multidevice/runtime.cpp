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
  auto error_msg = validate();
  NVF_ERROR(error_msg.empty(), error_msg);
  PipelineExecutor executor(*this);
  return executor.runWithInput(inputs);
}

std::string MultiDeviceRuntime::validate() const {
  if (!comm_.is_available()) {
    return "distributed configuration required";
  }

  if (pipeline_->requestedNumberOfDevices() > comm_.size()) {
    return "the pipeline requests " +
        std::to_string(pipeline_->requestedNumberOfDevices()) +
        " GPUs to run, but there are only " + std::to_string(comm_.size()) +
        " ranks in the communicator";
  }

  if (comm_.local_size() > static_cast<uint64_t>(at::cuda::getNumGPUs())) {
    return std::to_string(comm_.local_size()) +
        " processes are spawn on the node but only " +
        std::to_string(at::cuda::getNumGPUs()) + " GPUs are available";
  }

  return "";
}

} // namespace nvfuser

#endif

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
#include <multidevice/pipeline.h>
#include <multidevice/pipeline_ir.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/registry.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/cuda/CUDAContext.h>
#include <multidevice/communicator.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#define COMM_BACKEND_DEFAULT CommunicatorBackend::nccl
#define COMM_SERVER_RANK_DEFAULT 0


namespace nvfuser {

// default canonical mappings rank -> device ID -> concrete devices used in runtime class
static auto rankToDeviceIdx_default(RankType rank) {
    return static_cast<DeviceIdxType>(rank);
}
static auto deviceIdxToDevice_default(DeviceIdxType device_id) {
    return at::Device("cuda:" + std::to_string(device_id % at::cuda::getNumGPUs()));
}

/*
  The MultiDeviceRuntime class gather all what is needed for executing a Pipeline on a
  multi-device setting. It is instantiated from a Pipeline and a Communicator (a default
  Communicator is built at initialization if none is provided).
  It also holds mappings rank -> device ID -> concrete devices which
  are defined as the canonical mappings for now.
*/
class TORCH_CUDA_CU_API MultiDeviceRuntime {
 public:
  explicit MultiDeviceRuntime(Pipeline* pipeline, Communicator comm = {COMM_BACKEND_DEFAULT, COMM_SERVER_RANK_DEFAULT})
  : pipeline_(pipeline), comm_(comm) {
    validate();
  }

  // Run the multidevice fusion with the given global inputs
  std::vector<at::Tensor> runWithInput(std::vector<c10::IValue> inputs);

  // Returns the Communicator
  auto& comm() {
    return comm_;
  }

  // Returns the Communicator
  auto pipeline() const {
    return pipeline_;
  }

  // Returns the rank of the process
  auto rank() const {
    return comm_.rank();
  }

  // mappings: rank -> virtual device id -> concrete device.
  // for now, assuming canonical mappings but could consider other one-to-one mappings in the future
  auto rankToDeviceIdx(RankType rank) const {
    return rankToDeviceIdx_default(rank);
  }

  auto deviceIdxToDevice(DeviceIdxType device_id) const {
    return deviceIdxToDevice_default(device_id);
  }

  auto device() const {
      return deviceIdxToDevice(rankToDeviceIdx(comm_.rank()));
  }

  auto deviceIdxToRank(DeviceIdxType dId) const {
    for (auto rank: comm_.ranks()) {
      if (rankToDeviceIdx(rank) == dId) {
        return rank;
      }
    }
    TORCH_INTERNAL_ASSERT(
      false, "No rank is associated with device index " + std::to_string(dId));
  }

//  private:
  friend class PipelineExecutor; // could remove friendship by passing pipeline_ and comm_ to PipelineExecutor
  // test if the runtime is valid and satisfies our assumptions
  void validate() const;

  Pipeline* pipeline_;
  Communicator comm_;//TODO: put in private
};

} // namespace nvfuser

#endif

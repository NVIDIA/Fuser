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
#include <multidevice/communicator.h>
#include <multidevice/pipeline.h>
#include <multidevice/pipeline_ir.h>

#define COMM_BACKEND_DEFAULT CommunicatorBackend::nccl
#define COMM_SERVER_RANK_DEFAULT 0

namespace nvfuser {

/*
  The MultiDeviceRuntime class gather all what is needed for executing a
  Pipeline on a multi-device setting. It is instantiated from a Pipeline and a
  Communicator (a default Communicator is built at initialization if none is
  provided). It also holds mappings ranks <-> device IDs, and associate a device
  to the current node
*/
class TORCH_CUDA_CU_API MultiDeviceRuntime {
 public:
  explicit MultiDeviceRuntime(
      Pipeline* pipeline,
      Communicator comm = // NOLINT: pass by value and use std::move
      {COMM_BACKEND_DEFAULT, // NOLINT: pass by value and use std::move
       COMM_SERVER_RANK_DEFAULT}) // NOLINT: pass by value and use std::move
      : pipeline_(pipeline), comm_(comm) {
    validate();
  }

  // Run the multidevice fusion with the given global inputs
  std::vector<at::Tensor> runWithInput(std::vector<c10::IValue> inputs);

  // Returns the Communicator
  auto& comm() {
    return comm_;
  }

  // Returns the Pipeline
  auto pipeline() const {
    return pipeline_;
  }

  // Returns the rank of the process
  auto rank() const {
    return comm_.rank();
  }

  // returns the device associated with the process
  auto device() const {
    return at::Device("cuda:" + std::to_string(comm_.local_rank()));
  }

  // returns the rank corresponding to device index
  auto dIdToRank(DeviceIdxType d_id) const {
    return static_cast<RankType>(d_id);
  }

  // returns the device index corresponding to the rank
  auto rankToDiD(RankType rank) const {
    return static_cast<DeviceIdxType>(rank);
  }

 private:
  friend class PipelineExecutor; // could remove friendship by passing pipeline_
                                 // and comm_ to PipelineExecutor
  // test if the runtime is valid and satisfies our assumptions
  void validate() const;

  Pipeline* pipeline_;
  Communicator comm_;
};

} // namespace nvfuser

#endif

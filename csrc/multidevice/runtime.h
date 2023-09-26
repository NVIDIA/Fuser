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
#include <multidevice/pipeline.h>
#include <multidevice/pipeline_ir.h>

namespace nvfuser {

/*
  The MultiDeviceRuntime class gather all what is needed for executing a
  Pipeline on a multi-device setting. It is instantiated from a Pipeline and a
  Communicator (a default Communicator is built at initialization if none is
  provided).
*/
class MultiDeviceRuntime {
 public:
  explicit MultiDeviceRuntime(Pipeline* pipeline, Communicator& comm)
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

 private:
  friend class PipelineExecutor; // could remove friendship by passing pipeline_
                                 // and comm_ to PipelineExecutor
  // test if the runtime is valid and satisfies our assumptions
  void validate() const;

  Pipeline* pipeline_;
  Communicator& comm_;
};

} // namespace nvfuser

#endif

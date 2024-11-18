// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/core/DeviceType.h>
#include <exceptions.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <host_ir/executor.h>
#include <ir/cloner.h>
#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

/*
  The MultiDeviceExecutor executes a Fusion on a multi-device setting.
  It is instantiated from a Fusion and a Communicator.

  The Fusion must be scheduled prior to the instantiation of the
  MultiDeviceExecutor. One can use the multidevice scheduling API to specify
  the desired tensor sharding. It is composed of two aspects:
    *) Set each tensor's DeviceMesh, through TensorView::setDeviceMesh
    *) parallelize each tensor axis, possibly with the multidevice sharding
       parallel type ParallelType::DIDx

  We make the following assumptions on the Fusion:
  - Only one (non-reduction) axis is allowed to be parallelized
    with ParallelType::DIDx. Moreover, this axis cannot be split/merged.
  - We only support 1D device meshes for now
  - We only support TensorViews in communication segments.

  Summary of the different steps performed by the MultiDeviceExecutor:
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

  TODOS:
  *) the MultiDeviceExecutor should be integrated into FusionExecutorCache.
  *) The different steps should be divided into compilation, allocation,
     runtime etc. This will be done along the way when we will have better
     symbolic representation of the multidevice modules
  *) Allocation of buffers needs to be reimplemented
  *) Need to work on auto-scheduling, in particular, to combine inter-/intra-
     device scheduling.
*/

class MultiDeviceExecutor {
 public:
  MultiDeviceExecutor(
      std::unique_ptr<Fusion> fusion,
      Communicator& comm,
      hir::HostIrEvaluatorParams params = hir::HostIrEvaluatorParams());

  // Run the fusion on several devices with the given global inputs
  std::vector<at::Tensor> runWithInput(const std::vector<c10::IValue>& inputs);

  // Returns the Communicator
  Communicator* comm() const {
    return &comm_;
  }

  // Returns the Fusion
  auto completeFusion() const {
    return complete_fusion_.get();
  }

  // check if the runtime is valid returns an error msg.
  // An empty message means that the runtime is valid
  std::string validate() const;

  //! Print to default debugging output stream
  std::ostream& print(std::ostream& os = debug());

  const auto& getFusionExecutorCaches() {
    return host_ir_executor_->getFusionExecutorCaches();
  };

 private:
  // holds the Communicator to be used for execution
  Communicator& comm_;
  // holds the original complete fusion
  std::unique_ptr<Fusion> complete_fusion_;
  // holds the HostIrEvaluator used for execution
  std::unique_ptr<hir::HostIrEvaluator> host_ir_executor_;
  // Cached objects used for MultiDevice allocation
  // TODO: remove and handle the allocation through Host Irs
  std::unique_ptr<Fusion> allocator_fusion_;
  // Cache the tensors that need to be allocated at runtime, which correspond to
  // the destination buffers of interdevice communications.
  std::vector<Val*> vals_to_allocate_;
};

} // namespace nvfuser

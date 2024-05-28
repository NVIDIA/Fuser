// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dispatch.h>
#include <executor.h>
#include <host_ir/container.h>
#include <host_ir/host_ir.h>
#include <kernel_cache.h>
#include <multidevice/communicator.h>

#include <c10/cuda/CUDAStream.h>

namespace nvfuser {

namespace hir {

/*
a HostIrExecutor executes a host programs represented through a HostIrContainer
It is instantiated with the desired HostIrContainer, and runs the Host program
with concrete inputs by calling the method runWithInput.

For now HostIrExecutor is an interpreter; later we could rather compile host
code.

Note: most of the implementation is copy pasted for MultiDeviceExecutor. This
duplication will be resolved in the future.
*/

// Set of parameters that control the behavior of HostIrExecutor
struct HostIrExecutorParams {
  // Experimental: whether to use FusionExecutorCache rather than
  // FusionExecutor.
  bool use_fusion_executor_cache = false;
  // Experimental: whether to apply auto-scheduling in FusionExecutorCache if
  // use_fusion_executor_cache=true. WAR: temporary hack mainly use for
  // development
  bool skip_auto_scheduling = false;
  // Experimental: whether to cache fusion executor. WAR: avoid recompilation
  // but implicitely assumes that the input shape don't change over iterations
  bool cache_fusion_executor = false;
};

class HostIrExecutor final : public OptInDispatch {
 public:
  HostIrExecutor(
      std::unique_ptr<HostIrContainer> container,
      Communicator* communicator = nullptr,
      HostIrExecutorParams = HostIrExecutorParams());
  std::vector<at::Tensor> runWithInput(
      std::unordered_map<Val*, c10::IValue> val_to_IValue);

 private:
  using OptInDispatch::handle;
  void handle(SetCurrentStream* set_current_stream);
  void handle(PostOnStream* post) override;
  void postCompute(PostOnStream* post);
  void postCommunication(PostOnStream* post);

  std::unique_ptr<HostIrContainer> container_;
  Communicator* communicator_;
  HostIrExecutorParams params_;
  // Stores concrete computed values
  std::unordered_map<Val*, c10::IValue> val_to_IValue_;
  // Cache Fusions, FusionExecutors
  std::unordered_map<HostUnit*, FusionExecutor> fe_;
  std::unordered_map<HostUnit*, FusionExecutorCache> fec_;
  std::unordered_map<StreamIr*, c10::cuda::CUDAStream> streams_;
};

} // namespace hir

} // namespace nvfuser

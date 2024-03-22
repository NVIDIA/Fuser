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
#include <host_ir_container.h>
#include <ir/host_ir.h>
#include <kernel_cache.h>

namespace nvfuser {

namespace hir {

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
  HostIrExecutor(std::unique_ptr<HostIrContainer> container, HostIrExecutorParams = HostIrExecutorParams());
  std::vector<at::Tensor> runWithInput(const std::vector<c10::IValue>& inputs);

 private:
  using OptInDispatch::handle;
  void handle(PostOnStream* post) override;

  std::unique_ptr<HostIrContainer> container_;
  HostIrExecutorParams params_;
    // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue_;
  // Cache Fusions, FusionExecutors
  std::unordered_map<PostOnStream*, FusionExecutor> fe_;
  std::unordered_map<PostOnStream*, FusionExecutorCache> fec_;

};

} // namespace hir

} // namespace nvfuser

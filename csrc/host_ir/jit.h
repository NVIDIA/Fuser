// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h>
#include <host_ir/container.h>
#include <multidevice/communicator.h>
#include <utils.h>
#include <memory>
namespace nvfuser {

class HostIrJit {
 public:
  struct LaunchKernelResult {
    KernelArgumentHolder args;
    KernelArgumentHolder outputs;
  };

  at::Tensor allocate(
      const kir::Allocate* allocate,
      const std::vector<int64_t>& input_sizes,
      const std::vector<int64_t>& input_strides);

  at::Tensor allocate(
      const kir::Allocate* allocate);

  std::vector<at::Tensor> runFullGraph(
      const hir::HostIrContainer* container,
      const std::unordered_map<Val*, PolymorphicValue>& val_to_PValue);

LaunchKernelResult launchKernel(
      const hir::LaunchKernel* launch_kernel,
      int64_t cache_id,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs);

  HostIrJit(
      hir::HostIrContainer* container,
      int num_threads = kHostIrJitCompileThreads);

  ~HostIrJit();

 private:
  struct LlvmJitImpl;
  std::unique_ptr<LlvmJitImpl> pimpl_;
};
} // namespace nvfuser

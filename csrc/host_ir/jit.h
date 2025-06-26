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
#include <memory>
namespace nvfuser {

constexpr int64_t kHostIrJitCompileThreads = 4;

constexpr std::string_view kHostIrJitEmptyStridedCudaFuncName = "empty_strided_cuda";
class HostIrJit {
 public:
  at::Tensor allocate(
      const kir::Allocate* allocate,
      const std::vector<int64_t>& input_sizes,
      const std::vector<int64_t>& input_strides);

  HostIrJit(
      hir::HostIrContainer* container,
      int num_threads = kHostIrJitCompileThreads);

  ~HostIrJit();

 private:
 struct LlvmJitImpl;
  std::unique_ptr<LlvmJitImpl> pimpl_;
};
} // namespace nvfuser

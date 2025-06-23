// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h> // For TensorView and at::Tensor
#include <host_ir/container.h> // For HostIrContainer
#include <host_ir/executor.h> // For HostIrEvaluatorParams
#include <multidevice/communicator.h> // For Communicator
#include <memory> // For std::unique_ptr

namespace nvfuser {

class HostIrJit {
 public:

  // Run with the given input tensors.
  at::Tensor allocate(
      const kir::Allocate* allocate,
      const std::vector<int64_t>& input_sizes,
      const std::vector<int64_t>& input_strides);

  HostIrJit(
      hir::HostIrContainer* container,
      Communicator* communicator = &Communicator::getInstance(),
      hir::HostIrEvaluatorParams params = hir::HostIrEvaluatorParams(),
      int num_threads = 4);
  ~HostIrJit();

 private:
  struct LlvmJitImpl;
  std::unique_ptr<LlvmJitImpl> pimpl_;
};
} // namespace nvfuser

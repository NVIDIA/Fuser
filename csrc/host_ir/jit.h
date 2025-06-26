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
#include <multidevice/communicator.h> // For Communicator
#include <memory> // For std::unique_ptr
#include <utils.h>
namespace nvfuser {


class HostIrJit {
 public:

  // Run with the given input tensors.
  at::Tensor allocate(
      const kir::Allocate* allocate,
      const std::vector<int64_t>& input_sizes,
      const std::vector<int64_t>& input_strides);

  // Overloaded constructor for default params
  HostIrJit(
      hir::HostIrContainer* container,
      Communicator* communicator = &Communicator::getInstance(),
      const hir::HostIrEvaluatorParams& evaluator_params = hir::HostIrEvaluatorParams(),
      int num_threads = hostIrJitCompileThreads);

  ~HostIrJit() = default;

 private:
  struct LlvmJitImpl;
  std::unique_ptr<LlvmJitImpl> pimpl_;
};
} // namespace nvfuser

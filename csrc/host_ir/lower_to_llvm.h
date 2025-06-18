// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h> // For TensorView and at::Tensor
#include <memory> // For std::unique_ptr

namespace nvfuser {

class HostIrLlvmJit {
 public:

  // Compile a fusion associated with the given output TensorView.
  void compile(
      const hir::HostIrContainer* container);

  // Run with the given input tensors.
  at::Tensor allocate(
      const kir::Allocate* allocate,
      const std::vector<int64_t>& input_sizes);

  HostIrLlvmJit(int num_threads = 4);
  ~HostIrLlvmJit();

 private:
  struct LlvmJitImpl;
  std::unique_ptr<LlvmJitImpl> pimpl_;
};
} // namespace nvfuser

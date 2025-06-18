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
  // Get singleton instance
  static HostIrLlvmJit& getInstance(int num_threads = 4);

  // Compile a fusion associated with the given output TensorView.
  void compile(
      const hir::HostIrContainer* container);

  // Run with the given input tensors.
  at::Tensor allocate(
      const kir::Allocate* allocate,
      const std::vector<int64_t>& input_sizes);

 private:
  explicit HostIrLlvmJit(int num_threads = 4);
  ~HostIrLlvmJit();
  HostIrLlvmJit(HostIrLlvmJit&&) noexcept;
  HostIrLlvmJit& operator=(HostIrLlvmJit&&) noexcept;
  struct LlvmJitImpl;
  std::unique_ptr<LlvmJitImpl> pimpl_;
};
} // namespace nvfuser

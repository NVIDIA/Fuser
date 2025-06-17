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

  // Delete copy constructor and assignment operator
  HostIrLlvmJit(const HostIrLlvmJit&) = delete;
  HostIrLlvmJit& operator=(const HostIrLlvmJit&) = delete;

  // Compile a fusion associated with the given output TensorView.
  void compile(const hir::HostIrContainer* container);

 private:
  explicit HostIrLlvmJit(int num_threads = 4);
  ~HostIrLlvmJit();
  HostIrLlvmJit(HostIrLlvmJit&&) noexcept;
  HostIrLlvmJit& operator=(HostIrLlvmJit&&) noexcept;
  struct LlvmJitImpl;
  std::unique_ptr<LlvmJitImpl> pimpl_;
  std::vector<at::Tensor> input_tensors_;
};
} // namespace nvfuser

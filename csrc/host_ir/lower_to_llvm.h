// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h> // For TensorView and at::Tensor
#include <memory>  // For std::unique_ptr

namespace nvfuser {

class HostIrLlvmJit {
 public:
  // Constructor initializes the JIT
  explicit HostIrLlvmJit(int num_threads = 0);
  // Destructor is required for PIMPL with std::unique_ptr
  ~HostIrLlvmJit();

  // Enable move semantics
  HostIrLlvmJit(HostIrLlvmJit&&) noexcept;
  HostIrLlvmJit& operator=(HostIrLlvmJit&&) noexcept;

  // Disable copy
  HostIrLlvmJit(const HostIrLlvmJit&) = delete;
  HostIrLlvmJit& operator=(const HostIrLlvmJit&) = delete;

  // Compile a fusion associated with the given output TensorView.
  void compile(TensorView* output_tv);

  // Execute the compiled functions to allocate and return an output tensor.
  at::Tensor allocateOutputTensor(const std::vector<at::Tensor>& input_tensors);

 private:
  struct LlvmJitImpl; // The PIMPL forward declaration
  std::unique_ptr<LlvmJitImpl> pimpl_;

  // Store info from compile to run
  int64_t output_tensor_dim_ = 0;
};

} // namespace nvfuser
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <filesystem>
#include <string>

#include <runtime/compiled_kernel.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>

namespace nvfuser {

class Fusion;

class CutlassCompiledKernel : public CompiledKernelBase {
 public:
  // This constructor does not attempt to lower the fusion
  NVF_API CutlassCompiledKernel(
      Fusion* fusion,
      c10::Device device = c10::Device(c10::DeviceType::CUDA, 0),
      SchedulerType scheduler_type = SchedulerType::None,
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

  inline bool isCompiled() const {
    return compiled_;
  }

  void compile();

  void run(const KernelArgumentHolder& args, cudaStream_t stream) const;

 private:
  // Generate CUTLASS kernel code from fusion
  void generateCode();

  // Compile using nvcc (generate .so and dlopen)
  void compileWithNVCC();

  // Load compiled module/function
  void loadKernel();

 private:
  Fusion* fusion_;

  std::string cutlass_code_;
  std::string compilation_log_;

  // Temporary directory for nvcc compilation
  std::filesystem::path temp_dir_;

  bool compiled_ = false;

  // CUDA resources
  CUfunction cuda_function_ = nullptr;
  void* shared_library_handle_ = nullptr; // For nvcc/dlopen approach
};

} // namespace nvfuser

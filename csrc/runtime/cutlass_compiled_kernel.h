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
#include <scheduler/cutlass.h>

namespace nvfuser {

class Fusion;

class CutlassCompiledKernel : public CompiledKernelBase {
 public:
  // This constructor does not attempt to lower the fusion
  NVF_API CutlassCompiledKernel(
      Fusion* fusion,
      const CutlassParams& params,
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

  ~CutlassCompiledKernel();

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

  const CutlassParams params_;

  std::string cutlass_code_;
  std::string compilation_log_;

  // Temporary directory for nvcc compilation
  std::filesystem::path temp_dir_;

  bool compiled_ = false;

  //! Note that this must match the expected type in the generated kernel
  struct TensorArg {
    void* data_ptr;
    int64_t dim;
    int64_t* sizes;
    int64_t* strides = nullptr;
  };

  //! Running the jitted CUTLASS kernel is a four-step process:
  //!   1) Query the size of temporary tensors needed using the dlsym'ed
  //!   temp_sizes function
  //!   2) Allocate those temporary tensors via ATen and place them in a vector
  //!   3) Run the dlsym'ed init_kernel function to initialize the temporary
  //!   tensors
  //!   4) Run the kernel using the dlsym'ed run_kernel function

  // This is set when generating the kernel
  int64_t num_temp_tensors_ = 32;

  //! Signature for the function for getting the workspace size
  using WorkspaceSizeFunc = size_t (*)(void*);
  WorkspaceSizeFunc workspace_size_function_ = nullptr;

  //! Signature for the function to run the kernel
  using RunKernelFunc =
      void (*)(const std::vector<TensorArg>&, uint8_t*, cudaStream_t);
  RunKernelFunc run_kernel_function_ = nullptr;

  void* shared_library_handle_ = nullptr;
};

} // namespace nvfuser

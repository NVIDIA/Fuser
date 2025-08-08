// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <runtime/compiled_kernel.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>
#include <scheduler/cutlass.h>

namespace nvfuser {

class Fusion;

// Compilation options specific to CUTLASS kernels
struct CutlassCompileOptions {
  // Whether to use NVRTC or nvcc
  bool use_nvrtc = true;

  // CUTLASS include paths
  std::vector<std::string> include_paths;

  // Optimization level
  int optimization_level = 3;

  // Target architecture
  int compute_capability = 0; // 0 means auto-detect

  // Enable debug mode
  bool debug = false;

  // CUTLASS specific defines
  std::vector<std::string> defines;
};

// CUTLASS kernel descriptor containing kernel configuration
struct CutlassKernelDescriptor {
  // Kernel name
  std::string kernel_name;

  // CUTLASS operation type (e.g., "cutlass::gemm::device::Gemm")
  std::string operation_type;

  // Template parameters for the CUTLASS kernel
  std::string template_params;

  // Launch configuration
  dim3 grid_dim;
  dim3 block_dim;
  int shared_memory_size = 0;

  // Kernel arguments info
  std::vector<std::string> argument_types;
  std::vector<size_t> argument_sizes;
};

// Compiled CUTLASS kernel similar to CompiledKernel but for CUTLASS
class CutlassCompiledKernel : public NonCopyable {
 public:
  CutlassCompiledKernel() = delete;

  ~CutlassCompiledKernel();

  CutlassCompiledKernel(
      Fusion* fusion,
      const CutlassParams& cutlass_params,
      const CutlassCompileOptions& compile_options = CutlassCompileOptions());

  // Compile the CUTLASS kernel
  void compile();

  // Check if kernel is compiled
  bool isCompiled() const {
    return compiled_ && (cuda_module_ != nullptr || cuda_function_ != nullptr);
  }

  // Run the kernel with given arguments
  float run(const KernelArgumentHolder& args, cudaStream_t stream = nullptr);

  // Get the generated CUTLASS code
  const std::string& getCode() const {
    return cutlass_code_;
  }

  // Get kernel descriptor
  const CutlassKernelDescriptor& getDescriptor() const {
    return descriptor_;
  }

  // Get compilation log
  const std::string& getCompilationLog() const {
    return compilation_log_;
  }

  // Get PTX/CUBIN if available
  const std::vector<char>& getBinary() const {
    return binary_;
  }

 private:
  // Generate CUTLASS kernel code from fusion
  void generateCode();

  // Compile using NVRTC
  void compileWithNVRTC();

  // Compile using nvcc (generate .so and dlopen)
  void compileWithNVCC();

  // Load compiled module/function
  void loadKernel();

  // Generate kernel arguments structure
  void generateKernelArguments();

  // Create launch parameters
  void createLaunchParams();

  // Member variables
  Fusion* fusion_ = nullptr;
  CutlassParams cutlass_params_;
  CutlassCompileOptions compile_options_;
  CutlassKernelDescriptor descriptor_;

  bool compiled_ = false;
  std::string cutlass_code_;
  std::string compilation_log_;
  std::vector<char> binary_; // PTX or CUBIN

  // CUDA resources
  CUmodule cuda_module_ = nullptr;
  CUfunction cuda_function_ = nullptr;
  void* shared_library_handle_ = nullptr; // For nvcc/dlopen approach

  // Kernel arguments buffer
  std::vector<uint8_t> kernel_args_buffer_;
  std::vector<void*> kernel_arg_pointers_;

  // Launch configuration
  LaunchParams launch_params_;

  // Temporary directory for nvcc compilation
  std::string temp_dir_;
};

// Code generator for CUTLASS kernels
class CutlassCodeGenerator {
 public:
  // Generate CUTLASS C++ code for a fusion
  static std::string generateCode(
      Fusion* fusion,
      const CutlassParams& params,
      CutlassKernelDescriptor& descriptor);

 private:
  // Generate includes
  static std::string generateIncludes();

  // Generate kernel definition
  static std::string generateKernelDefinition(
      const CutlassKernelDescriptor& descriptor);

  // Generate epilogue visitor tree (EVT) code
  static std::string generateEpilogueVisitorTree(
      Fusion* fusion,
      const CutlassParams& params);

  // Generate kernel launch wrapper
  static std::string generateLaunchWrapper(
      const CutlassKernelDescriptor& descriptor);

  // Map nvfuser types to CUTLASS types
  static std::string mapDataTypeToCutlass(DataType dtype);

  // Map nvfuser layout to CUTLASS layout
  static std::string mapLayoutToCutlass(const TensorView* tv);
};

} // namespace nvfuser

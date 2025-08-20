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
#include <vector>

#include <runtime/compiled_kernel.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>
#include <scheduler/cutlass.h>

namespace nvfuser {

class Fusion;

// Compilation options specific to CUTLASS kernels
struct CutlassCompileOptions {
  // CUTLASS include paths
  std::vector<std::string> include_paths;

  // Optimization level
  int optimization_level = 3;

  // Target architecture
  int compute_capability = 0; // 0 means auto-detect

  // CUTLASS specific defines
  std::vector<std::string> defines;
};

// CUTLASS kernel descriptor containing kernel configuration
struct CutlassKernelDescriptor {
  // Kernel name
  std::string kernel_name;

  // Host function that launches the kernel
  std::string launch_function_name;

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
      const CutlassCompileOptions& compile_options = CutlassCompileOptions(),
      c10::Device device = c10::Device(c10::DeviceType::CUDA, 0),
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

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

  std::string& kernelId() {
    return kernel_id_;
  }
  const std::string& kernelId() const {
    return kernel_id_;
  }

  bool validKernelId() const {
    return !kernel_id_.empty();
  }

  void createKernelId();

  std::string kernelName() const;

 private:
  // Generate CUTLASS kernel code from fusion
  void generateCode();
  void generateCutlassCode();

  // Compile using NVRTC
  void compileWithNVRTC();

  // Compile using nvcc (generate .so and dlopen)
  void compileWithNVCC();

  // Load compiled module/function
  void loadKernel();

  // Create launch parameters
  void createLaunchParams();

 private:
  Fusion* fusion_ = nullptr;

  CutlassParams cutlass_params_;

  c10::Device device_;

  // ID of fusion in python frontend fusion cache, which maps to a single
  // CompiledKernelCache.
  const int64_t fusion_id_ = -1;

  // ID of (device, concrete_info) key in CompiledKernelCache
  const int64_t concrete_id_ = -1;

  // ID of FusionKernelRuntime given (device, concrete_info) key
  const int64_t runtime_id_ = -1;

  // ID of segment in FusionKernelRuntime
  const int64_t group_id_ = -1;

  // Note that this is separate from CompiledKernel::global_fusion_count_
  inline static std::atomic<int64_t> global_cutlass_fusion_count_;

  // Kernel name for fusion executor
  std::string kernel_id_;

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

  // Launch configuration
  LaunchParams launch_params_;

  // Temporary directory for nvcc compilation
  std::filesystem::path temp_dir_;
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

  // Generate nvfp4 scaled matmul kernel
  static std::string generateNvfp4ScaledMmKernel(
      Fusion* fusion,
      const CutlassParams& params,
      CutlassKernelDescriptor& descriptor);
};

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <atomic>
#include <functional>

#include <c10/core/DeviceType.h>

#include <exceptions.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <ir/graphviz.h>
#include <ir/printer.h>
#include <multidevice/communicator.h>
#include <runtime/allocations.h>
#include <runtime/executor_params.h>
#include <runtime/executor_utils.h>
#include <scheduler/scheduler_types.h>
#include <utils.h>

namespace nvfuser {

class GpuLower;

//! Internal tests only. Compiles CUDA code with NVRTC directly from
//! string. This util provides a path to test runtime code, i.e. the resource
//! strings.
class RtcKernel : public NonCopyable {
 public:
  NVF_API void compile(
      const std::string& code,
      const std::string& name,
      bool structured,
      PrimDataType index_type,
      int64_t device_index = 0);

  //! Internal tests only. Runs the compiled CUDA kernel. Returns elapsed
  //! milliseconds.
  NVF_API float run(
      const LaunchParams& launch_params,
      const KernelArgumentHolder& args,
      PrimDataType indextype);

 private:
  std::unique_ptr<executor_utils::CudaExecutable> compiled_kernel_;
  int64_t device_index_;
};

//! Class for compilation logic through nvRTC. It shouldn't hold any logic
//! associated with how to run a kernel, but how to compile it. It should also
//! contain any information about the kernel itself.
class CompiledKernel : public NonCopyable {
 public:
  // NVF_API was added for nvfuser_extension. See examples/sinh_extension.
  CompiledKernel() = delete;

  NVF_API ~CompiledKernel();

  NVF_API CompiledKernel(
      Fusion* fusion,
      CompileParams compile_params,
      c10::Device device,
      SchedulerType scheduler_type,
      int64_t fusion_id,
      int64_t concrete_id,
      int64_t runtime_id,
      int64_t group_id,
      const std::vector<std::function<void(GpuLower*)>>& pre_lowering_hooks,
      const std::vector<std::function<void(kir::Kernel*)>>&
          post_lowering_hooks);

  NVF_API CompiledKernel(
      Fusion* fusion,
      CompileParams compile_params,
      c10::Device device,
      SchedulerType scheduler_type = SchedulerType::None,
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

  CompiledKernel(const CompiledKernel& other) = delete;
  CompiledKernel& operator=(const CompiledKernel& other) = delete;
  CompiledKernel(CompiledKernel&& other) noexcept = delete;
  CompiledKernel& operator=(CompiledKernel&& other) noexcept = delete;

  //! To compile a fusion with the 32-bit index type, CompileParams
  //! must be passed in. There used to be an index type associated
  //! with KernelArgumentHolder, but it is no longer the case.
  NVF_API virtual void compile(const LaunchParams& lparams);

  // Function to query whether a `CompiledKernel` has a compiled kernel to
  // execute
  virtual bool isCompiled() const {
    if (lowered_ == nullptr) {
      return false;
    }
    if (compiled_kernel_ == nullptr) {
      return false;
    }
    NVF_ERROR(compiled_kernel_->function != nullptr);
    NVF_ERROR(validKernelId(), "Problem detected with compiled kernel ID.");
    return true;
  };

  static void setGlobalFusionCount(int64_t new_fusion_count) {
    global_fusion_count_.store(new_fusion_count);
  }

  static int64_t getGlobalFusionCount() {
    return global_fusion_count_.load();
  }

  static std::string kernelNamespace() {
    return "nvf";
  }

  // Public methods needed by KernelExecutor
  kir::Kernel* kernel() const;

  //! Returns the string of the compiled kernel
  NVF_API std::string kernelString() const {
    NVF_ERROR(!kernel_code_.empty(), "Kernel code not generated");
    return kernel_code_;
  }

  NVF_API std::string getStructuredCode() const;

  //! Returns a const reference to the latest compiled kernel.
  const std::unique_ptr<executor_utils::CudaExecutable>& cudaExecutable()
      const {
    return compiled_kernel_;
  }
  std::unique_ptr<executor_utils::CudaExecutable>& cudaExecutable() {
    return compiled_kernel_;
  }

  const c10::Device& device() const {
    return device_;
  }

  std::unique_ptr<GpuLower>& lowered() {
    return lowered_;
  }

  SchedulerType& schedulerType() {
    return scheduler_type_;
  }

  std::string& kernelCode() {
    return kernel_code_;
  }

  int64_t blockSizeHighWatermark() {
    return block_size_high_watermark_;
  }

  int64_t maxrregcountHighWatermark() {
    return maxrregcount_high_watermark_;
  }

  bool launchParamCacheDisabled() const {
    return launch_param_cache_disabled_;
  }

  void disableLaunchParamCache() {
    launch_param_cache_disabled_ = true;
  }

  void recompileKernel(
      const LaunchParams& lparams,
      const CompileParams& cparams);

  void deserialize(const serde::KernelExecutor* buffer);

  // Additional public methods needed by executor
  void setUsedTVs();

  const std::vector<TensorView*>& getUsedTVs() const {
    return used_tvs_;
  }

 protected:
  // Virtual method to generate kernel code - can be overridden by subclasses
  virtual std::string generateKernelCode();

  // Virtual method to compile the kernel - can be overridden by subclasses
  virtual void compileKernel();

  using ExecutorCompileTimeInfoCache =
      executor_utils::caching::ExecutorCompileTimeInfoCache;

  //! Returns the disassembled latest compiled binary
  NVF_API std::string disassembledKernelSASS() const;

  const int64_t& groupId() const {
    return group_id_;
  }

  bool validKernelId() const {
    return !kernel_id_.empty();
  }

  void createKernelId();

  std::string kernelName() const {
    NVF_ERROR(!kernel_id_.empty(), "Invalid kernel name for fusion executor.");
    std::stringstream ss;
    ss << "nvfuser_" << kernel_id_;
    return ss.str();
  }





 private:
  CompileParams compile_params_;
  // Assuming sm70 or above:
  //  limit of statically allocated smem is 48 KB:
  // See:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
  const int64_t max_static_smem_ = 48 << 10;

  int64_t warp_size_ = 0;
  std::unique_ptr<executor_utils::CudaExecutable> compiled_kernel_;

  // TensorViews actually used in the kernel.
  std::vector<TensorView*> used_tvs_;

  // Scheduling Heuristic for this Fusion
  SchedulerType scheduler_type_ = SchedulerType::None;

  // ID of fusion in python frontend fusion cache, which maps to a single
  // CompiledKernelCache.
  const int64_t fusion_id_ = -1;

  // ID of (device, concrete_info) key in CompiledKernelCache
  const int64_t concrete_id_ = -1;

  // ID of FusionKernelRuntime given (device, concrete_info) key
  const int64_t runtime_id_ = -1;

  // ID of segment in FusionKernelRuntime
  const int64_t group_id_ = -1;

  inline static std::atomic<int64_t> global_fusion_count_;

  // Kernel name for fusion executor
  std::string kernel_id_;

  std::unique_ptr<GpuLower> lowered_;

  // Track the block size this kernel was compiled with. If the block size
  // increases, recompile to adjust maxregister count.
  int64_t block_size_high_watermark_ = 1;
  int64_t maxrregcount_high_watermark_ = 255;

  // Profiling support: disable caching of launch params and output allocation
  // output allocation is also disable when output sizes are dependent on
  // runtime scalar inputs, such as for the case of tensor factory. see
  // https://github.com/csarofeen/pytorch/issues/2002
  bool launch_param_cache_disabled_ = false;

  // Profiling support: kept copy of the cuda kernel
  std::string kernel_code_;

  const c10::Device device_ = c10::Device(c10::DeviceType::CUDA, 0);
};

} // namespace nvfuser

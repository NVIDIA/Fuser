// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <device_lower/lower2device.h>
#include <exceptions.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_executor/allocations.h>
#include <fusion_executor/executor_params.h>
#include <fusion_executor/executor_utils.h>
#include <host_ir/container.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <multidevice/communicator.h>
#include <scheduler/scheduler_types.h>
// #include <serde/fusion_cache_generated.h>
#include <utils.h>
#include <atomic>

#include <c10/core/DeviceType.h>

#include <functional>

namespace nvfuser {

// TODO: Should this actually be in launch params?
struct CompileOptions {
  c10::Device device = c10::Device(c10::DeviceType::CUDA, 0);
};

class CompiledKernel : public NonCopyable {
 public:
  // NVF_API was added for nvfuser_extension. See examples/sinh_extension.
  NVF_API CompiledKernel() = default;

  //! To compile a fusion with the 32-bit index type, CompileParams
  //! must be passed in. There used to be an index type associated
  //! with KernelArgumentHolder, but it is no longer the case.
  NVF_API void compileFusion(
      Fusion* fusion,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      CompileParams compile_params,
      SchedulerType sceduler_type = SchedulerType::None,
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

  // TODO: merge it with the overload above.
  //! This API is merely here so we don't have to go back and update all cpp
  //! tests.
  void compileFusion(
      Fusion* fusion,
      const at::ArrayRef<c10::IValue>& inputs = {},
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams()) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    compileFusion(fusion, args, launch_constraints, compile_params);
  }

  //! Used by user defined schedules in python frontend
  void compileFusion(
      Fusion* fusion,
      const at::ArrayRef<c10::IValue>& inputs,
      int64_t fusion_id,
      int64_t concrete_id) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    compileFusion(
        fusion,
        args,
        LaunchParams(),
        CompileParams(),
        SchedulerType::None,
        fusion_id,
        concrete_id);
  }

  // Register a lowering hooks that are called to modify the GpuLower object
  // before running lowering passes. The main use case is for unit tests to
  // modify the lowering process.
  void registerLoweringHook(std::function<void(GpuLower*)> hook) {
    lowering_hooks_.push_back(std::move(hook));
  }

  // Register a post-lowering hooks that are called to modify the kernel after
  // lowering. The main use case is for unit tests to modify the kernel.
  void registerPostLoweringHook(std::function<void(kir::Kernel*)> hook) {
    post_lowering_hooks_.push_back(std::move(hook));
  }

  // Function to query whether compilation was attempted for a `CompiledKernel`
  bool isCompiled() const {
    int num_compiled_artifacts = (fusion_ != nullptr) + (lowered_ != nullptr);
    NVF_ERROR(num_compiled_artifacts <= 1);
    return num_compiled_artifacts == 1;
  };

  // function to query whether a `CompiledKernel` has a compiled kernel to
  // execute
  bool hasCompiledKernel() const {
    if (compiled_kernel_ != nullptr) {
      NVF_ERROR(compiled_kernel_->function != nullptr);
      NVF_ERROR(
          fusion_ == nullptr,
          "fusion_ should only be initialized when using expression evaluator.");
    }
    return validKernelId() && lowered_ && compiled_kernel_ != nullptr;
  };

  using ExecutorCompileTimeInfoCache =
      executor_utils::caching::ExecutorCompileTimeInfoCache;

  kir::Kernel* kernel() const {
    NVF_ERROR(lowered_);
    return lowered_->kernel();
  }

  Fusion* fusion() const {
    NVF_ERROR(isCompiled());
    if (fusion_ != nullptr) {
      return fusion_.get();
    }
    if (lowered_ != nullptr) {
      return lowered_->kernel()->as<Fusion>();
    }
    NVF_THROW("unreachable because of the isCompiled check");
  }

  const ThreadPredicateMap& threadPredMap() const {
    return lowered_->threadPredMap();
  }

  //! get register spills (load + store) of the compiled kernel
  int getKernelRegisterSpills() const {
    return compiled_kernel_->register_spills;
  }

  //! Returns the string of the compiled kernel
  NVF_API std::string kernelString() const {
    NVF_ERROR(!kernel_code_.empty(), "Kernel code not generated");
    return kernel_code_;
  }

  // Add preamble and wrap in namespace
  NVF_API std::string getStructuredCode(
      const std::string& kernel,
      PrimDataType index_type) const;

  NVF_API std::string getStructuredCode() const;

  //! Returns a const reference to the latest compiled kernel.
  const executor_utils::CompiledKernel& compiledKernel() const {
    return *compiled_kernel_;
  }

  //! Returns the disassembled latest compiled binary
  NVF_API std::string disassembledBinary(
      const std::string& nvdisasm_args = "") const {
    return executor_utils::disassembleBinary(
        compiled_kernel_->cubin, nvdisasm_args);
  }

  //! Returns the disassembled latest compiled binary
  NVF_API std::string disassembledKernelSASS() const {
    return executor_utils::disassembleBinary(
        compiled_kernel_->cubin, "-fun 1 -c");
  }

  static void setGlobalFusionCount(int64_t new_fusion_count) {
    global_fusion_count_.store(new_fusion_count);
  }

  static int64_t getGlobalFusionCount() {
    return global_fusion_count_.load();
  }

  int64_t groupId() const {
    return group_id_;
  }
  void setGroupId(int64_t gid) {
    group_id_ = gid;
  }

  bool validKernelId() const {
    return !kernel_id_.empty();
  }

  void createKernelId(
      SchedulerType scheduler_type = SchedulerType::None,
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0) {
    NVF_ERROR(fusion_id > -1, "Invalid fusion_id.");
    NVF_ERROR(concrete_id > -1, "Invalid concrete_id.");
    NVF_ERROR(runtime_id > -1, "Invalid runtime_id.");
    NVF_ERROR(group_id > -1, "Invalid group_id");

    scheduler_type_ = scheduler_type;
    fusion_id_ = fusion_id;
    concrete_id_ = concrete_id;
    runtime_id_ = runtime_id;
    group_id_ = group_id;
    ++global_fusion_count_;

    std::stringstream ss;
    if (isOptionEnabled(EnableOption::StaticFusionCount)) {
      ss << global_fusion_count_.load();
    } else {
      ss << toString(scheduler_type_);
      ss << "_f" << fusion_id_;
      ss << "_c" << concrete_id_;
      ss << "_r" << runtime_id_;
      ss << "_g" << group_id_;
    }
    kernel_id_ = ss.str();
  }

  std::string kernelName() const {
    NVF_ERROR(!kernel_id_.empty(), "Invalid kernel name for fusion executor.");
    std::stringstream ss;
    ss << "nvfuser_" << kernel_id_;
    return ss.str();
  }

  //! Internal tests only. Compiles CUDA code with NVRTC directly from
  //! string. This util provides a path to test runtime code, i.e. the resource
  //! strings.
  // TODO: Consider split out compileRtc and runRtc to a different
  //! class. Not much code is shared with the normal path.
  NVF_API void compileRtc(
      const std::string& code,
      const std::string& name,
      bool structured,
      PrimDataType index_type);

  //! Internal tests only. Runs the compiled CUDA kernel from
  //! compileRtc. Return the elapsed milliseconds.
  NVF_API float runRtc(
      const LaunchParams& launch_params,
      const std::vector<at::Tensor>& args,
      PrimDataType indextype);

  //! Internal knob used for debugging/profiling only
  void disableLaunchParamCache() {
    disable_parameter_cache_ = true;
  }

  // //! Serialize Fusion Executor using flatbuffers
  // flatbuffers::Offset<serde::CompiledKernel> serialize(
  //     flatbuffers::FlatBufferBuilder& builder) const;

  // //! Serialize CompiledKernel using flatbuffers
  // flatbuffers::Offset<serde::CudaKernel> serialize(
  //     flatbuffers::FlatBufferBuilder& builder,
  //     const executor_utils::CompiledKernel* kernel) const;

  // //! Deserialize Fusion Executor using flatbuffers
  // void deserialize(
  //     const serde::CompiledKernel* buffer,
  //     Fusion* fusion,
  //     int8_t device_index,
  //     CompileParams compile_params,
  //     SchedulerType scheduler_type,
  //     int64_t fusion_id,
  //     int64_t concrete_id,
  //     int64_t runtime_id,
  //     int64_t group_id);

 private:
  void setUsedTVs();

  const std::vector<TensorView*>& getUsedTVs() const {
    return used_tvs_;
  };

  // Recompile the kernel if the number of threads in the block has increased
  // or maxrregcount has changed
  void recompileKernel(
      const LaunchParams& new_launch_params,
      const CompileParams& new_compile_params);

 private:
  CompileOptions options_;

  // Assuming sm70 or above:
  //  limit of statically allocated smem is 48 KB:
  // See:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
  const int64_t max_static_smem_ = 48 << 10;

  int64_t warp_size_ = 0;
  std::unique_ptr<executor_utils::CompiledKernel> compiled_kernel_;

  // TensorViews actually used in the kernel.
  std::vector<TensorView*> used_tvs_;

  // ID of fusion in python frontend fusion cache, which maps to a single
  // CompiledKernelCache.
  int64_t fusion_id_ = -1;

  // ID of (device, concrete_info) key in CompiledKernelCache
  int64_t concrete_id_ = -1;

  // ID of FusionKernelRuntime given (device, concrete_info) key
  int64_t runtime_id_ = -1;

  // ID of segment in FusionKernelRuntime
  int64_t group_id_ = -1;

  inline static std::atomic<int64_t> global_fusion_count_;

  // Scheduling Heuristic for this Fusion
  SchedulerType scheduler_type_ = SchedulerType::None;

  // Kernel name for fusion executor
  std::string kernel_id_;

  std::unique_ptr<GpuLower> lowered_;

  // Initialized for non-compiled fusions
  std::unique_ptr<Fusion> fusion_;

  // Track the block size this kernel was compiled with. If the block size
  // increases, recompile to adjust maxregister count.
  int64_t block_size_high_water_mark_ = 1;
  int64_t maxrregcount_high_water_mark_ = 255;

  // Profiling support: disable caching of launch params and output allocation
  // output allocation is also disable when output sizes are dependent on
  // runtime scalar inputs, such as for the case of tensor factory. see
  // https://github.com/csarofeen/pytorch/issues/2002
  bool disable_parameter_cache_ = false;

  // Profiling support: kept copy of the cuda kernel
  std::string kernel_code_;

  // Lowering hooks that are called after the GpuLower instance is created
  // before running lowering passes.
  // The main use case is for unit tests to modify the lowering process.
  std::vector<std::function<void(GpuLower*)>> lowering_hooks_;

  // Post-lowering hooks that are called to modify the kernel after lowering.
  // The main use case is for unit tests to modify the kernel.
  std::vector<std::function<void(kir::Kernel*)>> post_lowering_hooks_;

  Communicator* communicator_;
};

} // namespace nvfuser

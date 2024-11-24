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
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <runtime/allocations.h>
#include <runtime/compiled_kernel.h>
#include <runtime/executor_abstract.h>
#include <runtime/executor_params.h>
#include <runtime/executor_utils.h>
#include <scheduler/scheduler_types.h>
#include <serde/fusion_cache_generated.h>
#include <utils.h>
#include <atomic>

#include <c10/core/DeviceType.h>

#include <functional>

namespace nvfuser {

class ExprEvalExecutor : public ExecutorAbstract {
 public:
  ExprEvalExecutor(
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0)
      : ExecutorAbstract(fusion_id, concrete_id, runtime_id, group_id) {}

  // Returns true if all fusion outputs are expression evaluated.
  static bool supported(Fusion* fusion);

  void compile(Fusion* fusion);

  bool isCompiled() const override;

  NVF_API std::vector<at::Tensor> run(
      KernelArgumentHolder& args,
      std::vector<at::Tensor> outputs = {});

  const std::unique_ptr<Fusion>& fusion() {
    return fusion_;
  }

 private:
  // TODO: Set properly
  std::unique_ptr<Fusion> fusion_;
};

class KernelExecutor : public ExecutorAbstract {
 public:
  // NVF_API was added for nvfuser_extension. See examples/sinh_extension.
  NVF_API KernelExecutor(
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0)
      : ExecutorAbstract(fusion_id, concrete_id, runtime_id, group_id) {}

  // TODO: What rules should be in this check?
  static bool supported(Fusion* fusion);

  //! To compile a fusion with the 32-bit index type, CompileParams
  //! must be passed in. There used to be an index type associated
  //! with KernelArgumentHolder, but it is no longer the case.
  NVF_API void compile(
      Fusion* fusion,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      CompileParams compile_params,
      SchedulerType sceduler_type = SchedulerType::None);

  // TODO: merge it with the overload above.
  //! This API is merely here so we don't have to go back and update all cpp
  //! tests.
  void compile(
      Fusion* fusion,
      const at::ArrayRef<c10::IValue>& inputs = {},
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams()) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    compile(fusion, args, launch_constraints, compile_params);
  }

  // TODO: args shouldn't come in a reference here because we will append the
  // outputs to be able to send it to the kernel. For now none of the users are
  // reconsuming the args, so it is okay. It isn't done now because changing it
  // from a reference makes a call as run({}) ambiguous, and that is used
  // in some places in the codebase.
  NVF_API std::vector<at::Tensor> run(
      KernelArgumentHolder& args,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      std::vector<at::Tensor> outputs = {});

  std::vector<at::Tensor> run(
      const at::ArrayRef<c10::IValue>& inputs,
      const std::vector<at::Tensor>& outputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      const std::optional<size_t>& opt_code = std::nullopt) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    if (opt_code.has_value()) {
      args.setCacheId(*opt_code);
    }
    return run(args, launch_constraints, compile_params, outputs);
  }

  std::vector<at::Tensor> run(
      const at::ArrayRef<c10::IValue>& inputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      const std::optional<size_t>& opt_code = std::nullopt) {
    return run(inputs, {}, launch_constraints, compile_params, opt_code);
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

  // Function to query whether compilation was attempted for a `KernelExecutor`
  bool isCompiled() const override {
    if (compiledKernel()) {
      return true;
    }
    return fusion_ != nullptr;
  };

  void evictCache(size_t cache_id) {
    executor_entry_lookup_.erase(cache_id);
  }

  // struct used to hold necessary information to launch compiled kernel on a
  // given input set.
  //
  // TODO: strides would also be important when we handle permutations in
  //       codegen.
  //
  struct ExecutorEntry {
    bool init = false;
    LaunchParams launch_params;
    std::vector<GlobalBufferInfo> outputs;
    // Temporary work buffers and intemediate global-memory tensors
    std::vector<GlobalBufferInfo> intermediates;
    // The arguments to the kernel. These are configured in computeArgs and
    // recomputeArgs.
    // For the common case of a tensor argument, these correspond to the
    // `struct Tensor` data in runtime/tensor.cu. That means each tensor
    // element in `args` would be a sizeof(void*) + len(shape)*sizeof(int) +
    // len(shape)*sizeof(int) byte array (here "int" is used in place of the
    // index type, which varies in practice).
    std::vector<std::vector<std::byte>> args;
    // This is just the data() pointers to the above `args`; cuLaunchKernel
    // requires an array of this form.
    std::vector<void*> arg_ptrs;
  };

  using ExecutorCompileTimeInfoCache =
      executor_utils::caching::ExecutorCompileTimeInfoCache;

  const std::unique_ptr<Fusion>& fusion() {
    return fusion_;
  }
  //! Internal knob used for debugging/profiling only
  void setExecuteKernelFlag(bool execute_kernel) {
    execute_kernel_ = execute_kernel;
  }

  //! get occupancy of the last kernel execution
  float getKernelOccupancy() const {
    NVF_ERROR(
        kernel_occupancy_ > 0,
        "Occupancy unknown, should run with dump occupancy or perf_debug_verbose");
    return kernel_occupancy_;
  }

  void setKernelOccupancy(float occupancy) {
    kernel_occupancy_ = occupancy;
  }

  //! Returns the launch parameters from the last kernel execution
  LaunchParams lastLaunchParams() const {
    return launch_params_;
  }

  static void setGlobalFusionCount(int64_t new_fusion_count) {
    CompiledKernel::setGlobalFusionCount(new_fusion_count);
  }

  static int64_t getGlobalFusionCount() {
    return CompiledKernel::getGlobalFusionCount();
  }

  void setGroupId(int64_t gid) {
    group_id_ = gid;
  }

  //! Serialize Fusion Executor using flatbuffers
  flatbuffers::Offset<serde::KernelExecutor> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  //! Deserialize Fusion Executor using flatbuffers
  void deserialize(
      const serde::KernelExecutor* buffer,
      Fusion* fusion,
      int8_t device_index,
      CompileParams compile_params,
      SchedulerType scheduler_type,
      int64_t fusion_id,
      int64_t concrete_id,
      int64_t runtime_id,
      int64_t group_id);

  const std::unique_ptr<CompiledKernel>& compiledKernel() const {
    return compiled_kernel_2_;
  }

  const std::unique_ptr<CompiledKernel>& initCompiledKernel() {
    compiledKernel_() = std::make_unique<CompiledKernel>();
    return compiledKernel();
  }

 private:
  LaunchParams computeLaunchParams(
      const LaunchParams& launch_constraints,
      ExpressionEvaluator& expr_eval,
      const int64_t warp_size,
      DataType index_dtype);

  //! Return information necessay for allocating intermediate tensors,
  //! including temporary work buffers as well as intermediate
  //! global-memory tensors
  // TODO: Move to allocations.h/cpp
  std::vector<GlobalBufferInfo> getIntermediateBufferInfo(
      ExpressionEvaluator& expr_eval,
      DataType index_dtype);

  ExecutorCompileTimeInfoCache* compileTimeDataCache() {
    return &compile_time_info_cache_;
  }

  //! TODO: Consider changing this to a constructor of ExecutorEntry
  void initializeExecutorEntry(
      ExecutorEntry& executor_entry,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      const CompileParams& compile_params,
      const std::vector<at::Tensor>& outputs,
      DataType index_type);

  std::unique_ptr<PrecomputedValues>& evaluatorPrecomputedValues();

  // Creates the initial set of arguments to a kernel, based on the arguments
  // to we have now.
  void computeArgs(ExecutorEntry&, ExpressionEvaluator&, const kir::Kernel*)
      const;
  // Updates an existing set of arguments based on the current arguments. It is
  // is an error to call this before `computeArgs` has been invoked.
  // recomputeArgs will fail if the arity of the function changes, or the rank
  // of any tensor changes (as these are compiled-in to the generated kernel
  // and therefore would require us to do a larger recompilation).
  void recomputeArgs(ExecutorEntry&, ExpressionEvaluator&, const kir::Kernel*)
      const;

  //! Serialize CompiledKernel using flatbuffers
  flatbuffers::Offset<serde::CudaKernel> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const executor_utils::CompiledKernel* kernel) const;

  // ExecutorEntry is an internal POD struct for the KernelExecutor class.
  // We define ExecutorEntry's serialize and deserialize as private methods in
  // KernelExecutor.
  flatbuffers::Offset<serde::ExecutorEntry> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const ExecutorEntry& data) const;

  //! Deserialize ExecutorEntry using flatbuffers
  ExecutorEntry deserialize(const serde::ExecutorEntry* buffer);

  // GlobalBufferInfo is an internal POD struct for the KernelExecutor class.
  // We define GlobalBufferInfo's serialize and deserialize as private methods
  // in KernelExecutor.
  flatbuffers::Offset<serde::GlobalBufferInfo> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const GlobalBufferInfo& data,
      int64_t tv_position,
      bool is_fusion_output) const;

  //! Deserialize GlobalBufferInfo using flatbuffers
  GlobalBufferInfo deserialize(const serde::GlobalBufferInfo* buffer);

  //! Get the current dynamic shared memory size
  int64_t getAvailableDynamicSmemSize();

  //! Get the static shared memory size of the current compiled kernel
  int64_t getStaticSmemSize();

  //! Check if the shared memory size can be expandable to accommodate
  //! the given dynamic size. The total shared memory size consumed
  //! would be the sum of the static and dynamic sizes.
  void validateDynamicSmemSize(int64_t dynamic_smem_size);

  //! Make sure the dynamic shared memory size is at least as large as
  //! the given size
  int64_t ensureAvailableDynamicSmemSize(int64_t dynamic_smem_size);

  //! Clear the cached properties of the compiled kernel
  void resetCompiledKernelProperties();

  std::unique_ptr<CompiledKernel>& compiledKernel_() {
    return compiled_kernel_2_;
  }

  void disableLaunchParamCache() {
    if (compiledKernel()) {
      compiledKernel()->disableLaunchParamCache();
    }
  }

 private:
  std::unique_ptr<CompiledKernel> compiled_kernel_2_;

  //! Absolute limit of all available shared mem space from cudaDeviceProp
  int64_t device_smem_limit_ = 0;

  //! Static shared memory size of the current compiled kernel
  std::optional<int64_t> static_smem_size_ = std::nullopt;

  //! Available shared memory space for dynamic allocation for the current
  //!  compiled kernel at the current shared memory/L1 configuration
  std::optional<int64_t> available_dynamic_smem_size_ = std::nullopt;

  int64_t warp_size_ = 0;

  // Initialized for non-compiled fusions
  std::unique_ptr<Fusion> fusion_;

  // lookup table to take short cut to retrieve recorded information in order to
  // launch kernels without re-inference parameters.
  std::unordered_map<size_t, ExecutorEntry> executor_entry_lookup_;

  // Compile time information caching. This is used for shape inference
  //  support. The cache stores graph information that are available
  //  without shape information so that each shape inference call will
  //  not need to re-compute them.
  ExecutorCompileTimeInfoCache compile_time_info_cache_;

  // Cached expr eval
  std::unique_ptr<PrecomputedValues> evaluator_precomputed_values_ = nullptr;

  // Profiling support: knob to control wheter we actually execute the
  // kernel on the GPU or not
  bool execute_kernel_ = true;

  // Heuristic tuning support: the last kernel occupancy, if
  // DebugDumpOption::Occupancy is true
  float kernel_occupancy_ = -1.0f;

  // Profiling support: the last launch param used
  LaunchParams launch_params_;

  // Lowering hooks that are called after the GpuLower instance is created
  // before running lowering passes.
  // The main use case is for unit tests to modify the lowering process.
  std::vector<std::function<void(GpuLower*)>> lowering_hooks_;

  // Post-lowering hooks that are called to modify the kernel after lowering.
  // The main use case is for unit tests to modify the kernel.
  std::vector<std::function<void(kir::Kernel*)>> post_lowering_hooks_;

  // TODO: Should this be removed?
  SchedulerType scheduler_type_ = SchedulerType::None;
};

} // namespace nvfuser

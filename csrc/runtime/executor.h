// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <functional>

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

  NVF_API KernelArgumentHolder
  run(const KernelArgumentHolder& args, KernelArgumentHolder outputs = {});

  const std::unique_ptr<Fusion>& fusion() {
    return fusion_;
  }

 private:
  // TODO: Set properly
  std::unique_ptr<Fusion> fusion_;
};

// struct used to hold necessary information to launch compiled kernel on a
// given input set.
//
// TODO: strides would also be important when we handle permutations in
//       codegen.
//
struct KernelExecutorEntry {
  bool init = false;
  LaunchParams launch_params;
  std::vector<GlobalBufferInfo> outputs;
  // If an output is aliased to an input, this will hold the index of the
  // input that it is aliased to. If not aliased, it will hold -1.
  std::vector<int> output_aliased_to_input;
  // Temporary work buffers and intemediate global-memory tensors
  std::vector<GlobalBufferInfo> intermediates;
  std::vector<GlobalBufferInfo> inputs;
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

class GpuLower;

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
      const KernelArgumentHolder& args = {},
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      SchedulerType sceduler_type = SchedulerType::None);

  NVF_API KernelArgumentHolder
  run(KernelArgumentHolder args,
      KernelArgumentHolder outputs = {},
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams());

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

  // Returns whether this `KernelExecutor` has a compiled kernel to execute.
  bool isCompiled() const override {
    if (compiledKernel() && compiledKernel()->isCompiled()) {
      return true;
    }
    return false;
  };

  void evictCache(size_t cache_id) {
    executor_entry_lookup_.erase(cache_id);
  }

  using ExecutorCompileTimeInfoCache =
      executor_utils::caching::ExecutorCompileTimeInfoCache;

  //! Internal knob used for debugging/profiling only
  void setExecuteKernelFlag(bool execute_kernel) {
    execute_kernel_ = execute_kernel;
  }

  //! get occupancy of the last kernel execution
  float getKernelOccupancy() const {
    NVF_ERROR(
        kernel_occupancy_ > 0,
        "Occupancy unknown, should run with dump occupancy or "
        "perf_debug_verbose");
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

  int64_t groupId() const {
    return group_id_;
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
    return compiled_kernel_;
  }

  //! Get the static shared memory size of the current compiled kernel
  int64_t getStaticSmemSize();

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

  //! TODO: Consider changing this to a constructor of KernelExecutorEntry
  void initializeExecutorEntry(
      KernelExecutorEntry& executor_entry,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      const CompileParams& compile_params,
      const KernelArgumentHolder& outputs,
      DataType index_type);

  std::unique_ptr<PrecomputedValues>& evaluatorPrecomputedValues();

  // Creates the initial set of arguments to a kernel, based on the arguments
  // to we have now.
  void computeArgs(KernelExecutorEntry& entry, const KernelArgumentHolder& args)
      const;

  KernelArgumentHolder resolveTMA(
      KernelExecutorEntry& entry,
      const KernelArgumentHolder& args) const;

  //! Serialize CompiledKernel using flatbuffers
  flatbuffers::Offset<serde::CudaKernel> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const executor_utils::CudaExecutable* kernel) const;

  // KernelExecutorEntry is an internal POD struct for the KernelExecutor class.
  // We define KernelExecutorEntry's serialize and deserialize as private
  // methods in KernelExecutor.
  flatbuffers::Offset<serde::KernelExecutorEntry> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const KernelExecutorEntry& data) const;

  //! Deserialize KernelExecutorEntry using flatbuffers
  KernelExecutorEntry deserialize(const serde::KernelExecutorEntry* buffer);

  // GlobalBufferInfo is an internal POD struct for the KernelExecutor class.
  // We define GlobalBufferInfo's serialize and deserialize as private methods
  // in KernelExecutor.
  flatbuffers::Offset<serde::GlobalBufferInfo> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const GlobalBufferInfo& data,
      int64_t tv_position,
      bool is_fusion_output,
      bool is_fusion_input) const;

  //! Deserialize GlobalBufferInfo using flatbuffers
  GlobalBufferInfo deserialize(const serde::GlobalBufferInfo* buffer);

  //! Get the current dynamic shared memory size
  int64_t getAvailableDynamicSmemSize();

  //! Check if the shared memory size can be expandable to accommodate
  //! the given dynamic size. The total shared memory size consumed
  //! would be the sum of the static and dynamic sizes.
  void validateDynamicSmemSize(int64_t dynamic_smem_size);

  //! Make sure the dynamic shared memory size is at least as large as
  //! the given size
  int64_t ensureAvailableDynamicSmemSize(int64_t dynamic_smem_size);

  //! Clear the cached properties of the compiled kernel
  void resetCompiledKernelProperties();

 private:
  std::unique_ptr<CompiledKernel> compiled_kernel_;

  //! Absolute limit of all available shared mem space from cudaDeviceProp
  int64_t device_smem_limit_ = 0;

  //! Static shared memory size of the current compiled kernel
  std::optional<int64_t> static_smem_size_ = std::nullopt;

  //! Available shared memory space for dynamic allocation for the current
  //!  compiled kernel at the current shared memory/L1 configuration
  std::optional<int64_t> available_dynamic_smem_size_ = std::nullopt;

  int64_t warp_size_ = 0;

  // Has an RNG kernel and therefore needs to infer RNG state through expression
  // evaluator
  bool has_rng_ = false;

  // Has a TMA kernel and therefore needs to infer TMA inputs through expression
  // evaluator
  bool has_tma_ = false;

  // Has a dynamic alias and therefore needs to infer what they are through
  // expression evaluator
  bool has_dynamic_alias_ = false;

  // lookup table to take short cut to retrieve recorded information in order to
  // launch kernels without re-inference parameters.
  std::unordered_map<size_t, KernelExecutorEntry> executor_entry_lookup_;

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
};

class HostIrExecutor : public ExecutorAbstract {
 public:
  HostIrExecutor(
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

  static bool supported(Fusion* fusion);

  void compile(Fusion* fusion);

  bool isCompiled() const override;

  NVF_API KernelArgumentHolder
  run(const KernelArgumentHolder& args, KernelArgumentHolder outputs = {});

  const std::unique_ptr<hir::HostIrContainer>& hostContainer() const {
    return host_ir_container_;
  }

 private:
  std::unique_ptr<hir::HostIrContainer> host_ir_container_;
  Communicator* communicator_;
};

} // namespace nvfuser

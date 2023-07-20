// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <device_lower/lower2device.h>
#include <executor_params.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <utils.h>

#include <c10/core/DeviceType.h>

namespace nvfuser {

TORCH_CUDA_CU_API bool shouldFillAllocationWithNan();
TORCH_CUDA_CU_API void setFillAllocationWithNan(bool value);

// TODO: Should this actually be in launch params?
struct TORCH_CUDA_CU_API CompileOptions {
  c10::Device device = c10::Device(c10::DeviceType::CUDA, 0);
};

class TORCH_CUDA_CU_API FusionExecutor : public NonCopyable {
 public:
  struct GlobalBufferInfo {
    TensorView* tv = nullptr;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    at::ScalarType type = at::ScalarType::Undefined;
    bool zero_init = false;
    bool is_profile_buffer = false;
  };

  // Unsafe compilation that's useful for debugging kernels, iterating over
  // slight modifications of a generated kernel
  void debugCompileFusionFromStr(
      Fusion* fusion,
      const std::string& code,
      const std::string& name,
      int id,
      CompileOptions options = CompileOptions());

  //! This function is useful for parallel compilation of segmented fusions.
  //! It returns non-allocated KernelArgumentHolder, representing the output
  //! sizes from kernel execution.
  //! Notes: 1. This API should ignore aliased outputs instead of
  //! pushing scalar int 0 as a place-holder.
  //! 2. This API does not allocate output in memory, but only returns the
  //! inferred output sizes.
  KernelArgumentHolder inferOutputSizes(
      Fusion* fusion,
      const KernelArgumentHolder& args);

  //! To compile a fusion with the 32-bit index type, CompileParams
  //! must be passed in. There used to be an index type associated
  //! with KernelArgumentHolder, but it is no longer the case.
  void compileFusion(
      Fusion* fusion,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      CompileParams compile_params);

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

  std::vector<at::Tensor> runFusion(
      KernelArgumentHolder& args,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      std::vector<at::Tensor> outputs = {});

  std::vector<at::Tensor> runFusion(
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
    return runFusion(args, launch_constraints, compile_params, outputs);
  }

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<c10::IValue>& inputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      const std::optional<size_t>& opt_code = std::nullopt) {
    return runFusion(inputs, {}, launch_constraints, compile_params, opt_code);
  }

  // function to query whether a `FusionExecutor` has a compiled kernel to
  // execute
  bool compiled() const {
    return fusion_id_ != -1 && lowered_ && compiled_kernel_.function != nullptr;
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
    // Aliased output and input mappings
    std::vector<std::pair<int, int>> output_to_input_aliases;
    std::vector<GlobalBufferInfo> outputs;
    // Temporary work buffers and intemediate global-memory tensors
    std::vector<GlobalBufferInfo> intermediates;
    uint64_t rand_offset = 0;
  };

  using ExecutorCompileTimeInfoCache =
      executor_utils::caching::ExecutorCompileTimeInfoCache;

  kir::Kernel* kernel() const {
    TORCH_INTERNAL_ASSERT(lowered_);
    return lowered_->kernel();
  }

  const ThreadPredicateMap& threadPredMap() const {
    return lowered_->threadPredMap();
  }

  //! Internal knob used for debugging/profiling only
  void setExecuteKernelFlag(bool execute_kernel) {
    execute_kernel_ = execute_kernel;
  }

  //! Internal knob used for debugging/profiling only
  void setMeasureKernelTimeFlag(bool measure_kernel_time) {
    measure_kernel_time_ = measure_kernel_time;
  }

  //! Internal knob used for debugging/profiling only
  void setSaveCompiledBinaryFlag(bool save_compiled_binary) {
    save_compiled_binary_ = save_compiled_binary;
  }

  //! Returns the last kernel execution time, in milliseconds
  //!
  //! \note The kernel time is only tracked if enabled by calling
  //!    setMeasureKernelTimeFlag(true)
  //!
  float kernelTimeMs() const {
    return measure_kernel_time_ ? kernel_time_ms_ : 0;
  }

  //! Returns the number of bytes processed last kernel execution
  int64_t bytesProcessed() const {
    return bytes_processed_;
  }

  //! Returns the launch parameters from the last kernel execution
  LaunchParams lastLaunchParams() const {
    return launch_params_;
  }

  //! Returns the string of the compiled kernel
  std::string kernelString() const {
    TORCH_INTERNAL_ASSERT(!kernel_code_.empty(), "Kernel code not generated");
    return kernel_code_;
  }

  // Add preamble and wrap in namespace
  std::string getStructuredCode(
      const std::string& kernel,
      PrimDataType index_type) const;

  std::string getStructuredCode() const;

  //! Returns the latest compile log
  std::string compilerLog() const {
    return last_compiler_log_;
  }

  //! Returns the latest compiled binary
  std::vector<char> compiledBinary() const {
    return last_compiled_binary_;
  }

  //! Returns the disassembled latest compiled binary
  std::string disassembledBinary(const std::string& nvdisasm_args = "") const {
    return executor_utils::disassembleBinary(
        last_compiled_binary_, nvdisasm_args);
  }

  //! Returns the disassembled latest compiled binary
  std::string disassembledKernelSASS() const {
    return executor_utils::disassembleBinary(
        last_compiled_binary_, "-fun 1 -c");
  }

  std::string getCanonicalKernelName() const {
    return kernelNamespace() + "::" + kernelName();
  }

  std::string kernelName() const {
    std::stringstream ss;
    ss << "kernel" << fusion_id_;
    return ss.str();
  }

  //! Internal tests only. Compiles CUDA code with NVRTC directly from
  //! string. This util provides a path to test runtime code, i.e. the resource
  //! strings.
  // TODO: Consider split out compileRtc and runRtc to a different
  //! class. Not much code is shared with the normal path.
  void compileRtc(
      const std::string& code,
      const std::string& name,
      bool structured,
      PrimDataType index_type);

  //! Internal tests only. Runs the compiled CUDA kernel from
  //! compileRtc. Return the elapsed milliseconds.
  float runRtc(
      const LaunchParams& launch_params,
      const std::vector<at::Tensor>& args,
      PrimDataType indextype);

  //! Internal knob used for debugging/profiling only
  void disableLaunchParamCache() {
    disable_parameter_cache_ = true;
  }

 private:
  static std::string kernelNamespace() {
    return "CudaCodeGen";
  }

  LaunchParams computeLaunchParams(
      const LaunchParams& launch_constraints,
      ExpressionEvaluator& expr_eval,
      const int64_t warp_size);

  int64_t computeSharedMemory(
      ExpressionEvaluator& expr_eval,
      const std::vector<const kir::Allocate*>& buffers,
      bool align_padding = false,
      int64_t total = 0);

  //! Return information necessay for allocating intermediate tensors,
  //! including temporary work buffers as well as intermediate
  //! global-memory tensors
  std::vector<GlobalBufferInfo> getIntermediateBufferInfo(
      ExpressionEvaluator& expr_eval);

  //! Return information necessay for allocating output tensors. Input
  //! and output tensors are allowed to alias each other, which is
  //! specified by the list of int pairs of input and output indices
  std::vector<GlobalBufferInfo> getOutputBufferInfo(
      const KernelArgumentHolder& args,
      ExpressionEvaluator& expr_eval,
      const std::vector<std::pair<int, int>>& input_to_output_aliases);

  void setUsedTVs();

  const std::vector<TensorView*>& getUsedTVs() const {
    return used_tvs_;
  };

  ExecutorCompileTimeInfoCache* compileTimeDataCache() {
    return &compile_time_info_cache_;
  }

  //! TODO: Consider changing this to a constructor of ExecutorEntry
  void initializeExecutorEntry(
      ExecutorEntry& executor_entry,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      const CompileParams& compile_params,
      const std::vector<at::Tensor>& outputs);

  std::unique_ptr<PrecomputedValues>& evaluatorPrecomputedValues();

  // Recompile the kernel if the number of threads in the block has increased
  // or maxrregcount has changed
  void recompileKernel(
      const LaunchParams& new_launch_params,
      const CompileParams& new_compile_params);

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

  //! Get the corresponding TensorViews for each argument of the kernel.
  //! If the corresponding argument is not a tensor, use nullptr as placeholder.
  //! Right now, kernel arguments are in the following order:
  //! inputs, outputs, intermediates, philox
  std::vector<TensorView*> getTvsForKernelArguments() const;

 private:
  CompileOptions options_;

  //! Absolute limit of all available shared mem space from cudaDeviceProp
  int64_t device_smem_limit_ = 0;

  //! Static shared memory size of the current compiled kernel
  std::optional<int64_t> static_smem_size_ = std::nullopt;

  //! Available shared memory space for dynamic allocation for the current
  //!  compiled kernel at the current shared memory/L1 configuration
  std::optional<int64_t> available_dynamic_smem_size_ = std::nullopt;

  // Assuming sm70 or above:
  //  limit of statically allocated smem is 48 KB:
  // See:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
  const int64_t max_static_smem_ = 48 << 10;

  int64_t warp_size_ = 0;
  executor_utils::NvrtcFunction compiled_kernel_;

  // TensorViews actually used in the kernel.
  std::vector<TensorView*> used_tvs_;

  // Counter to be used for kernel name.
  int64_t fusion_id_ = -1;
  static int64_t fusion_id_counter_;

  std::unique_ptr<GpuLower> lowered_;
  // Copy of lowered_->kernel()
  Fusion* fusion_ = nullptr;

  // Track the block size this kernel was compiled with. If the block size
  // increases, recompile to adjust maxregister count.
  int64_t block_size_high_water_mark_ = 1;
  int64_t maxrregcount_high_water_mark_ = 255;

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

  // Profiling support: knob to enable measuring kernel execution time
  bool measure_kernel_time_ = false;

  // Profiling support: the last kernel execution time, if measure_kernel_time_
  // is true
  float kernel_time_ms_ = 0;

  // Profiling support: the last kernel Bytes processed
  int64_t bytes_processed_ = 0;

  // Profiling support: the last launch param used
  LaunchParams launch_params_;

  // Profiling support: disable caching of launch params and output allocation
  // output allocation is also disable when output sizes are dependent on
  // runtime scalar inputs, such as for the case of tensor factory. see
  // https://github.com/csarofeen/pytorch/issues/2002
  bool disable_parameter_cache_ = false;

  // Profiling support: kept copy of the cuda kernel
  std::string kernel_code_;

  // Profiling support: nvrtc log for debugging
  std::string last_compiler_log_;

  // save compiled binary
  bool save_compiled_binary_ = false;

  // nvrtc compiled binary
  std::vector<char> last_compiled_binary_;
};

} // namespace nvfuser

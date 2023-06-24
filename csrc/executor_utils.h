// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/csrc/jit/ir/ir.h>

#include <cuda_utils.h>
#include <device_lower/lower2device.h>
#include <executor_kernel_arg.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <kernel.h>

#include <string>
#include <vector>

namespace nvfuser {
namespace executor_utils {

// Include all the functions we might need in generated code
std::string kernelPreamble();

void validateKernelInputs(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const c10::Device& device);

void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    const c10::Device& device);

//! Bind input values to runtime values
TORCH_CUDA_CU_API ExpressionEvaluator bindInputs(
    const KernelArgumentHolder& args,
    Fusion* fusion,
    bool check_consistency = true);

std::string disassembleBinary(
    const std::vector<char>& cubin,
    const std::string& nvdisasm_args);

struct NvrtcFunction {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

// Returns executable function and the ptxas log from compilation
std::tuple<NvrtcFunction, std::string, std::vector<char>> getCompiledKernel(
    std::optional<std::reference_wrapper<const std::string>> kernel_code,
    const std::string& code,
    const std::string& func_name,
    int64_t id,
    const CompileParams& compile_params = CompileParams(),
    std::optional<int64_t> opt_block_size = std::nullopt,
    bool return_compiled_binary = false);

namespace caching {
// TODO: Could consider putting some of
//  the logic in the common space and re-use

//! List of all the possible entry types in
//!  `FusionExecutor` compile-time data cache.
enum class CompileTimeEntryType {
  PARALLEL_BINDING_ITERDOMAINS,
  PARALLEL_ITER_EXTENT_MAP,
  SIMPLIFIED_PARALLEL_ITER_EXTENT_MAP,
  WARP_PADDED_PARALLEL_EXTENTS,
  VECTORIZED_TENSOR_VALIDATION,
  INPUT_ALIAS_INDICES,
  OUTPUT_ALIAS_INDICES
};

//! Entry class definitions for each entry type:
//!  each class defines the data type for each entry type

//! Compile-time info to be cached in each FusionExecutor:
//!  ParallelBindingIterDomains:
//!    Stores all the iterdomains that are parallelized
//!    on the scheduled Fusion graph. They will be used
//!    in launch param iteration and their extents may
//!    come from launch constraints.
class ParallelBindingIterDomains {
 public:
  using DataType = std::vector<IterDomain*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::PARALLEL_BINDING_ITERDOMAINS;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  ParallelIterExtentMap
//!    Stores the symbolic extents of all the parallelized
//!    iterdomains corresponding to each used parallel type.
class ParallelIterExtentMap {
 public:
  using DataType = std::unordered_map<ParallelType, std::vector<const Val*>>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::PARALLEL_ITER_EXTENT_MAP;
};

//!  VectorizedTensorInfo:
//!    Auxiliary data type for entry class VectorizedTensorValidation
struct VectorizedTensorInfo {
  //! Aligned vectorized fusion inputs
  std::vector<int> aligned_vectorized_inp_tensor_pos;
  //! Aligned vectorized fusion outputs
  std::vector<int> aligned_vectorized_out_tensor_pos;
  //! Misaligned vectorized input tensors
  std::unordered_set<TensorView*> global_inp_misaligned_tv;
  //! Misaligned vectorized output tensors
  std::unordered_set<TensorView*> global_out_misaligned_tv;
  //! Positions of misaligned input tensors
  std::vector<int> inp_misaligned_tensors_pos;
  //! Positions of misaligned output tensors
  std::vector<int> out_misaligned_tensors_pos;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  VectorizedTensorValidation
//!    Stores position info and vector word sizes of
//!    vectorized input/output tensors, to be used
//!    in misaligned vectorization validation.
class VectorizedTensorValidation {
 public:
  using DataType = VectorizedTensorInfo;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::VECTORIZED_TENSOR_VALIDATION;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  InputAliasIndices
//!    Stores position info of aliased input tensors
class InputAliasIndices {
 public:
  using DataType = std::vector<std::pair<int, int>>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::INPUT_ALIAS_INDICES;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  OutputAliasIndices
//!    Stores position info of aliased output tensors
class OutputAliasIndices {
 public:
  using DataType = std::unordered_set<int>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::OUTPUT_ALIAS_INDICES;
};

//! Base abstract class for unified storage in `ExecutorCompileTimeInfoCache`,
//!  each entry in `ExecutorCompileTimeInfoCache` will be a subclass.
class CompileTimeInfoBase : public PolymorphicBase {
 public:
  CompileTimeInfoBase(CompileTimeEntryType entry_type)
      : entry_type_(entry_type) {}
  CompileTimeEntryType type() {
    return entry_type_;
  }

 private:
  CompileTimeEntryType entry_type_;
};

// Note: Do NOT export this class. MSVC issue with exported class that contains
// std::vector<unique_ptr<xxx>>: https://godbolt.org/z/3E4e8T1P1
//! Compile-time information cache
class ExecutorCompileTimeInfoCache {
  using Entry = CompileTimeInfoBase;
  using EntryOwningPtr = std::unique_ptr<Entry>;
  using EntryPtr = Entry*;
  using EntryType = CompileTimeEntryType;

 public:
  void insert(EntryOwningPtr new_entry);

  EntryPtr at(EntryType entry_type) {
    return entry_type_map_.at(entry_type);
  }

  bool has(EntryType entry_type) {
    return entry_type_map_.count(entry_type);
  }

 private:
  std::vector<EntryOwningPtr> entries_;
  std::unordered_map<EntryType, EntryPtr> entry_type_map_;
};

//! A utility class to facilitate accessing ExecutorCompileTimeInfoCache.
template <typename EntryClass>
class ExecutorCompileTimeEntry {
  using EntryDataType = typename EntryClass::DataType;
  using EntryDataTypeOwnPtr = std::unique_ptr<EntryDataType>;
  using MakerFnType = std::function<EntryDataTypeOwnPtr()>;

 public:
  //! Creates a data entry with type defined in EntryClass,
  //!  eg. EntryClass = VectorizableInputsAndOutputs;
  //!
  //! @param data_cache, a pointer to an instantiated compile-time
  //!  info cache. The info data will be
  //!    1. read from data cache if data cache has the corresponding entry.
  //!    2. written into data cache if data cache doesn't have the entry.
  //!    3. managed by owned_data_ if data cache is nullptr
  //! @param fn:
  //!   The factory function that needs to return a owning pointer
  //!  i.e. std::unique_ptr<EntryClass::DataType>. It will only
  //!  be called either when data cache is missing an entry or when no data
  //!  cache is given.
  ExecutorCompileTimeEntry(
      ExecutorCompileTimeInfoCache* data_cache,
      MakerFnType fn);

  //! Unified interface to get actual data, either from cache
  //!  or from factory function.
  EntryDataType& get() {
    return *data_ptr_;
  }

 private:
  //! Internal data owing pointer that will manage the computed
  //!  data where there is no data cache.
  EntryDataTypeOwnPtr owned_data_ = nullptr;

  //! Pointer to the valid data entry that could be accessed.
  EntryDataType* data_ptr_ = nullptr;
};

} // namespace caching

//! Returns the vector of tensorviews that will be used to bind parallel
//!  dimensions.
std::vector<IterDomain*> getParallelBindingsIterDomains(
    GpuLower* lower,
    const std::vector<TensorView*>& used_tvs);

using ParallelExtentMap =
    std::unordered_map<ParallelType, std::vector<const Val*>>;

//! Returns the extents of all parallel binding iterdomains corresponding
//!  to each parallel type.
std::unique_ptr<ParallelExtentMap> getParallelIterExtents(
    std::vector<IterDomain*>& parallel_binding_ids);

void validateVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval);

//! Kernel timing utility
//!
//! Usage example:
//!
//!   CudaKernelTimer timer(stream);
//!   timer.init();
//!   kernel<<<..., stream>>>(...);
//!   auto elapsed_ms = timer.elapsed();
//!
class CudaKernelTimer {
 public:
  CudaKernelTimer(cudaStream_t s) : stream_(s) {}

  ~CudaKernelTimer() {
    if (initialized_) {
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(finish_event));
    }
  }

  void init() {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));
  }

  void start() {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream_));
  }

  float elapsed() {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(finish_event, stream_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(finish_event));
    NVFUSER_CUDA_RT_SAFE_CALL(
        cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event));
    return kernel_time_ms_;
  }

 private:
  cudaStream_t stream_;
  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};
  bool initialized_ = false;
  float kernel_time_ms_ = 0;
};

} // namespace executor_utils
} // namespace nvfuser

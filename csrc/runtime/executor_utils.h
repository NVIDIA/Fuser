// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <exceptions.h>
#include <visibility.h>

#include <cuda_runtime.h>

#include <torch/csrc/jit/ir/ir.h>

#include <cuda_utils.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <kernel.h>
#include <runtime/executor_kernel_arg.h>

#include <string>
#include <vector>

namespace nvfuser {

class GpuLower;

namespace executor_utils {

// I'm not happy with CudaExecutable being a struct exposing all the fields.
// This could be refactored.
struct CudaExecutable : public NonCopyable {
  NVF_API ~CudaExecutable();

  CUmodule module = nullptr;
  CUfunction function = nullptr;
  std::string compile_log;
  std::vector<char> ptx;
  std::string ptx_filename;
  std::vector<char> cubin;
  std::string cubin_filename;
  std::string kernel_name;
  std::string compile_args;
  std::vector<char> sass;
  std::string sass_filename;
  long block_size = -1;
  int register_spills = -1;
};

//! Bind input values to runtime values
NVF_API ExpressionEvaluator
bindInputs(const KernelArgumentHolder& args, Fusion* fusion);

// Returns a vector where vector[out_idx] == the input index in fusion->inputs()
// that output[out_idx] is aliased to. If output[out_idx] is not aliased to any
// input, then vector[out_idx] is -1.
std::vector<int> getOutputAliasToInputMap(const Fusion* fusion);

// Compile time cache for execution
namespace caching {
// TODO: Could consider putting some of
//  the logic in the common space and re-use

//! List of all the possible entry types in
//!  `KernelExecutor` compile-time data cache.
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

//! Compile-time info to be cached in each KernelExecutor:
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

//! Compile-time info to be cached in each KernelExecutor:
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
  std::vector<int64_t> aligned_vectorized_inp_tensor_pos;
  //! Aligned vectorized fusion outputs
  std::vector<int64_t> aligned_vectorized_out_tensor_pos;
};

//! Compile-time info to be cached in each KernelExecutor:
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
    const KernelArgumentHolder& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval);

// Check that for all circular buffer TensorViews, the extent of the circular
// buffer axis is >= number of stages in circular buffer. If not, throw an
// exception at runtime.
void validateCircularBuffering(
    kir::Kernel* kernel,
    ExpressionEvaluator& expr_eval);

//! Check that any narrowing casts from DataType::Index do not overflow.
//! In particular, if TMA expressions are present in the kernel, compute bounds
//! for integer expressions in order to validate that the 32-bit coordinates
//! passed to the TMA PTX instructions do not overflow.
void validateIndexCasts(
    kir::Kernel* kernel,
    ExpressionEvaluator& expr_eval,
    const LaunchParams& launch_params);

} // namespace executor_utils
} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/pointwise_utils.h>
#include <scheduler/scheduler_types.h>
#include <scheduler/tools/domain_map.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>

namespace nvfuser {

//! namespace for hosting catalog of possible compile time
//!  info that can be cached. Each possible entry type has
//!  a value in `CompileTimeEntryType` and an entry type class
//!  definition like `VectorizableInputsAndOutputs`. The corresponnding
//!  classes contain their entry type, data type and maybe more
//!  later depending on use cases.
namespace HeuristicCompileTime {

//! Each entry type under this category represent some information
//!  that can be inferred compile-time, i.e. without any runtime input
//!  meta data. They will be stored in `HeuristicDataCache` and will
//!  be re-used each time the same fusion is visited.

//! Enum for all possible types of cached entries of compile-time info.
enum class CompileTimeEntryType {
  DOMAIN_MAP,
  TRANSPOSE_DOMAIN_MAP,
  REFERENCE_TENSORS,
  REFERENCE_TENSORS_FOR_GROUPS,
  VECTORIZABLE_INPUTS_AND_OUTPUTS,
  INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS,
  TV_TO_CONTIG_INNER_SIZE_MAPS,
  RESIZE_VECTORIZATION_FACTORS,
  UNROLLABLE_INPUTS_AND_OUTPUTS,
  REDUCTION_TVS,
  PERSISTENT_BUFFER_INFO,
  SCOPE_PERSISTENT_FACTOR_INFO,
  BROADCAST_BYTE_MULTIPLES,
  INNER_MOST_DIMS_INFO,
  CAN_SCHEDULE_TRANSPOSE,
  CAN_SCHEDULE_MUL_SUM_AS_MMA,
  LOGICAL_REORDER_MAP,
  VECTORIZATION_BREAK_POINT_OF_RED_PROD,
  HAS_BLOCK_QUANTIZATION_OPS
};

//! Entry type definition class for `DOMAIN_MAP`,
//!  stores the domain map of a fusion.
class DomainMap {
 public:
  using DataType = scheduler_tools::DomainMap;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::DOMAIN_MAP;
};

//! Entry type definition class for `DOMAIN_MAP`,
//!  stores the domain map of a fusion, used by transpose scheduler.
class TransposeDomainMap {
 public:
  using DataType = scheduler_tools::DomainMap;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::TRANSPOSE_DOMAIN_MAP;
};

//! Entry type definition class for `REFERENCE_TENSORS`,
//!  stores the the reference TensorViews used to schedule a fusion.
class ReferenceTensors {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::REFERENCE_TENSORS;
};

//! Entry type definition class for `REFERENCE_TENSORS`,
//!  stores the the reference TensorViews used to schedule a fusion, used by
//!  transpose scheduler.
class ReferenceTensorsForGroups {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::REFERENCE_TENSORS_FOR_GROUPS;
};

//! Entry type definition class for `VECTORIZABLE_INPUTS_AND_OUTPUTS`,
//!  stores the vectorizable TensorViews on a fusion's inputs and outputs.
class VectorizableInputsAndOutputs {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS;
};

//! Entry type definition class for `TV_TO_CONTIG_INNER_SIZE_MAPS`,
//!  stores the vectorizable TensorViews on a fusion's inputs and outputs.
class TvToContigInnerSizeMaps {
 public:
  using DataType = std::vector<std::unordered_map<TensorView*, Val*>>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::TV_TO_CONTIG_INNER_SIZE_MAPS;
};

//! Stores the scalar vals that a vectorization factor must be able to
//! divide evenly
class ResizeVectorizationFactors {
 public:
  using DataType = std::unordered_set<Val*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::RESIZE_VECTORIZATION_FACTORS;
};

//! Entry type definition class for `INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS`,
//!  stores the fusion's inputs and outputs grouped by inner most dimension.
class InputsOutputsInnerDimGroups {
 public:
  using DataType = std::vector<std::vector<TensorView*>>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS;
};

//! Entry type definition class for `UNROLLABLE_INPUTS_AND_OUTPUTS`,
//!  stores the unrollable TensorViews on a fusion's inputs and outputs.
class UnrollableInputsAndOutputs {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::UNROLLABLE_INPUTS_AND_OUTPUTS;
};

//! Entry type definition class for `REDUCTION_TVS`,
//!  stores the all tvs with reduction axes in a fusion.
class ReductionTVs {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::REDUCTION_TVS;
};

//! Entry type definition class for `HAS_BLOCK_QUANTIZATION_OPS`,
//!  stores a boolean flag indicating whether the fusion contains any
//!  BlockQuantizationOp operations.
class HasBlockQuantizationOps {
 public:
  using DataType = bool;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::HAS_BLOCK_QUANTIZATION_OPS;
};

//! Entry type definition class for `PERSISTENT_BUFFER_INFO`,
//!  stores persistent buffers inferred from topology and scheduling of fusion.
class PersistentBufferInfo {
 public:
  using DataType = scheduler_utils::PersistentBufferInfo;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::PERSISTENT_BUFFER_INFO;
};

//! Entry type definition class for `INNER_MOST_DIMS_INFO`,
//!  Used in the transpose scheduler to store inner most IterDomains and their
//!  position in reference1 of group 1 and group 2
//!  Note, negative value indicates mapping failure
class InnerMostDimInfo {
 public:
  using DataType = std::vector<int64_t>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::INNER_MOST_DIMS_INFO;
};

//! Auxiliary data types for `SCOPE_PERSISTENT_FACTOR_INFO` entry type.
using ScopedPersistenceBufferMap = std::unordered_map<Val*, std::vector<bool>>;

//! Entry type definition class for `SCOPE_PERSISTENT_FACTOR_INFO`,
// Tracks which buffers are active at a given Val*, order of bool vector is
// based on persistence buffer order from persistence buffer info, this is then
// appended by the projectable persistent buffers' inputs. True in the bool
// vector means the persistent buffer is active at the generation of the key.
class ScopePersistentFactorInfo {
 public:
  using DataType = ScopedPersistenceBufferMap;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::SCOPE_PERSISTENT_FACTOR_INFO;
};

//! Entry type definition class for `BROADCAST_BYTE_MULTIPLES`,
//!  stores "byte multiples" information. This information can be used to figure
//!  out if using a 2D scheduler how many bytes have to be transferred with
//!  varying split locations. See BroadcastMultiple definition for more
//!  information.
class BroadcastMultiples {
 public:
  using DataType = scheduler_utils::BroadcastMultipleInformation;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::BROADCAST_BYTE_MULTIPLES;
};

//! Entry type definition class for `CAN_SCHEDULE_TRANSPOSE`,
//!  stores if the transpose scheduler can scheduler this fusion
class CanScheduleTranspose {
 public:
  using DataType = bool;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::CAN_SCHEDULE_TRANSPOSE;
};

//! Entry type definition class for `LOGICAL_REORDER_MAP`,
//!  stores the domain map of a fusion.
class LogicalReorderMap {
 public:
  using DataType = std::unordered_map<int64_t, int64_t>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::LOGICAL_REORDER_MAP;
};

class VectorizationBreakPointOfReductionProducer {
 public:
  using DataType = int64_t;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::VECTORIZATION_BREAK_POINT_OF_RED_PROD;
};

//! Base abstract class for unified storage in `HeuristicDataCache`,
//!  each entry in `HeuristicDataCache` will be a subclass.
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

} // namespace HeuristicCompileTime

//! Note: Do NOT export this class. MSVC issue with exported class that contains
//! std::vector<unique_ptr<xxx>>: https://godbolt.org/z/3E4e8T1P1
//! Compile-time information cache for `canSchedule` and `getHeuristics`
//! interfaces. Each cache instance stores information that could be inferred at
//! compile time in a fusion and therefore corresponds to an instance of
//! KernelExecutor.
class HeuristicDataCache {
  using EntryOwningPtr =
      std::unique_ptr<HeuristicCompileTime::CompileTimeInfoBase>;
  using EntryPtr = HeuristicCompileTime::CompileTimeInfoBase*;
  using EntryType = HeuristicCompileTime::CompileTimeEntryType;

 public:
  bool hasEntry(EntryType entry_type) {
    return entry_type_map_.find(entry_type) != entry_type_map_.end();
  }

  void insert(EntryOwningPtr new_entry);

  EntryPtr at(EntryType entry_type) {
    return entry_type_map_.at(entry_type);
  }

 private:
  std::vector<EntryOwningPtr> entries_;
  std::unordered_map<EntryType, EntryPtr> entry_type_map_;
};

//! A utility class to facilitate accessing HeuristicDataCache.
//!  This utility is needed because the information to be stored
//!    in HeuristicDataCache is used in several different scenarios
//!    and we want to support all these use cases in one interface.
//!  The current use examples are:
//!   1. During fusion segmentation process, all the fusions
//!     given to canSchedule are temporary and therefore the
//!     compile time info do not need to be cached, and in fact
//!     a cache wouldn't be instantiated by that time.
//!
//!   2. When a kernel is created for the first time, entries will be
//!     missing in the cache and all the computed information will be
//!     captured and written into the cache.
//!
//!   3. When we check a compiled fusion for heuristic hit, we want to
//!     use the cached info to save runtime latency.
//!
//! The designed interface is used as:
//!   auto entry = HeuristicDataCacheEntry<EntryClass>(data_cache, maker_fn);
//!   auto& data = entry.get();
//!
//!  `maker_fn` will be called to compute the information when no cached data
//!   exists and `entry` will own the computed data when no data cache is
//!   supplied.
template <typename EntryClass>
class HeuristicDataCacheEntry {
  using EntryDataType = typename EntryClass::DataType;
  using EntryDataTypeOwnPtr = std::unique_ptr<EntryDataType>;
  using MakerFnType = std::function<EntryDataTypeOwnPtr()>;

 public:
  //! Creates a data entry with type defined in EntryClass,
  //!  eg. EntryClass = VectorizableInputsAndOutputs;
  //!
  //! @param data_cache, a pointer to an instantiated compile-time
  //!  info cache. The info data will be
  //!    1. read from data cache if data cache is not recording.
  //!    2. written into  data cache if data cache is recording.
  //!    3. managed by owned_data_ if data cache is nullptr
  //! @param fn:
  //!   The factory function that needs to return a owning pointer
  //!  i.e. std::unique_ptr<EntryClass::DataType>. It will only
  //!  be called either when data cache is recording or when no data
  //!  cache is given.
  HeuristicDataCacheEntry(HeuristicDataCache* data_cache, MakerFnType fn);

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

} // namespace nvfuser

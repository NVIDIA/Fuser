// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <runtime/executor.h>
#include <scheduler/heuristic.h>
#include <serde/fusion_cache_generated.h>
#include <utils.h>

#include <c10/util/ArrayRef.h>

#include <list>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class SegmentedGroup;
class SegmentedFusion;

// Utilities for benchmarking and profiling
struct ExecutorLog {
  std::unique_ptr<HeuristicParams> params = nullptr;
  ExecutorAbstract* fusion_executor = nullptr;
};

struct RuntimeWorkSpace {
  //! Pre-determined order to run the segmented groups
  std::vector<SegmentedGroup*> group_run_order;

  //! Pre-determined order to bind tensor input meta data
  std::vector<Val*> group_extent_binding_order;
};

// Perform a topological sort of different groups composiong the Segmented
// Fusion
void prepareRuntimeOrder(SegmentedFusion*, RuntimeWorkSpace&);

//! Simple hasher for pair<T, const U*>. There is no default hasher for pairs,
//! since there are a lot of options how to combine hashes. In a case where one
//! element of the pair is unlikely to change much, the following hash is fast
//! and effective.
struct PairPointerHash {
  template <typename T, typename U>
  size_t operator()(const std::pair<T, const U*>& p) const {
    auto hT = std::hash<T>{}(p.first);
    // Using pointer as an optional
    auto hU =
        p.second ? std::hash<U>{}(*(p.second)) : std::hash<void*>{}(nullptr);
    return hT ^ hU;
  }
};

struct PairPointerEquals {
  template <typename T, typename U>
  bool operator()(
      const std::pair<T, const U*>& lhs,
      const std::pair<T, const U*>& rhs) const {
    if (lhs.first != rhs.first) {
      return false;
    }
    if (lhs.second == rhs.second) {
      return true;
    }
    // Compare by dereference, but only if both pointers are non-null
    if (!lhs.second || !rhs.second) {
      // We've already compared pointers, so if either is null, they're not both
      return false;
    }
    return *(lhs.second) == *(rhs.second);
  }
};

// ArgumentManager is used to manage the lifetime of arguments. It accepts
// runtime_workspace and fusion_inputs which determine the lifetime of
// arguments. When updateWithSegmentOutputs is called, it will remove arguments
// based on the information in runtime_workspace and the group_id passed into
// updateWithSegmentOutputs.
class ArgumentManager {
 public:
  ArgumentManager(
      const KernelArgumentHolder& args,
      const RuntimeWorkSpace& runtime_workspace,
      const std::vector<Val*>& fusion_inputs);

  // Delete copy constructor and assignment operator since we have unique_ptrs
  ArgumentManager(const ArgumentManager&) = delete;
  ArgumentManager& operator=(const ArgumentManager&) = delete;

  // Allow move operations
  ArgumentManager(ArgumentManager&&) = default;
  ArgumentManager& operator=(ArgumentManager&&) = default;

  // This map is only taken on destruction. It might be good to steal the tensor
  // map instead of make a copy.
  std::unordered_map<Val*, PolymorphicValue> getTensorMap() const {
    return tensor_map_;
  }

  const PolymorphicValue& checkTensorMap(Val* v) const;

  // Translate a vector of Vals to their corresponding entries in tensor_map_
  KernelArgumentHolder translateValsToArgs(const std::vector<Val*>& vals) const;

  // Update argument manager with outputs from a segment
  void updateWithSegmentOutputs(
      const std::vector<Val*>& group_outputs,
      const KernelArgumentHolder& group_runtime_outputs,
      const int64_t group_id);

  std::string toString() const {
    std::stringstream ss;
    ss << "ArgumentManager {";
    for (const auto& [key, value] : tensor_map_) {
      ss << "  " << key->toString() << " : "
         << PolymorphicValue_functions::toString(value) << std::endl;
    }
    ss << "}" << std::endl;
    return ss.str();
  }

 private:
  std::unordered_map<Val*, PolymorphicValue> tensor_map_;
  // map segment_id to vector of fusion vals lastly used at this segment
  std::unordered_map<int64_t, std::vector<Val*>> vals_last_used_at_segment_;

  void mapFusionInputsToArgs(
      const std::vector<Val*>& fusion_inputs,
      const KernelArgumentHolder& args,
      const std::vector<Val*>& group_extent_binding_order);

  void setLastUsedSegmentID(
      const std::vector<SegmentedGroup*>& group_run_order);
};

//! Encoding an input set to unique id, which is used to short-cut cache entry
//! selection in our nested cache implementation to cut off overhead.
//!
//! We have implemented naive LRU cache eviction policy here, since each entry
//! in `InputsIdLookup` is attached to a static input shape/stride, and could
//! grow gigantic when we have input shapes that does not stabalize to a finite
//! set.
//!
//! \note the uniqueness of the ide generated for a given input set is only
//!   local to the instance of `InputsIdLookup`.
//!
class InputsIdLookup : public NonCopyable {
 public:
  //! constructor where maximum cache size is fixed during init
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  explicit InputsIdLookup(size_t max_cache_size = 100)
      : max_cache_size_(max_cache_size) {}

  //! struct to hold return value for lookupId.
  struct IdLookupReturn {
    size_t id = 0;
    size_t evict_id = 0;
    bool eviction = false;
  };
  //! Encode each input sets to with an unique id;
  //! The returned data structure also indicates whether eviction has happened
  //! within the lookup cache. This is needed because lookup shortcut is also
  //! cached in nested `FusionExecutorCache` and `KernelExecutor`.
  //! see [ Note -- Post-definition cache implementation ] and [ Note -- 2 level
  //! cache implementation ].
  //!
  //! In the presence of dynamic operations like reshape and resize, the
  //! structure of the concretized Fusion might depend on not only the extents
  //! of input tensors, but on input scalars. For example,
  //!
  //!    auto s = IrBuilder::create<Val>();
  //!    auto tv1 = reshape(tv0, {IrBuilder::create<Val>(-1), s});
  //!
  //!
  //! This code will accept an integer s and reshape tv0 such that its last
  //! dimension's extent is equal to s. During concretization,
  //! this _dynamic_ reshape is translated to a sequence of Merge and Split
  //! operations, which might differ depending on the value of s and the shape
  //! of tv0. This means that both the extents of tv0 as well as the value of s
  //! must affect the unique id returned by lookupId.
  //!
  //! By default, no scalar inputs affect the return value of this function.
  //! However, if scalar_inputs_to_record is provided, then the values of scalar
  //! inputs at the integer locations specified in that argument will affect the
  //! returned ID.
  NVF_API IdLookupReturn lookupId(
      const KernelArgumentHolder& args,
      const std::unordered_set<size_t>& scalar_inputs_to_record = {});

  //! debugging API that returns the size of lookup table
  size_t size() const {
    return encoding_lookup_.size();
  }

  //! Serialize InputsIdLookup using flatbuffers
  flatbuffers::Offset<serde::InputsIdLookup> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  //! Deserialize InputsIdLookup using flatbuffers
  void deserialize(const serde::InputsIdLookup* buffer);

 private:
  // string to store encoded input meta information. Reuse the buffer instead of
  // stringtream gives few us perf gain.
  std::string encoding_; // Note: shared state, guarded by mutex_

  // mutex_ used to guard reused encoding_
  std::mutex mutex_;

  //! entry stored in `encoding_lookup_` to implement LRU
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct EncodingEntry {
    size_t id = 0;
    std::list<std::string>::iterator lru_iter;
  };

  //! maximum cache size for LRU
  size_t max_cache_size_ = 0;

  //! next available unique id, we monotonically increase `current_id_` avoid
  //! conflicts
  size_t current_id_ = 1;

  //! entry in the cache, This is used to implement LRU cache, where entries in
  //! the list is ordered by their recent usage (freshly used entry is placed at
  //! the beginning)
  std::list<std::string> used_entry_;

  //! map from `std::string` to a unique id `size_t` (packaged in
  //! `EncodingEntry`
  //! ). We store an iterator to `used_entry_` to implement LRU
  std::unordered_map<std::string, EncodingEntry> encoding_lookup_;
};

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dynamic_transform.h>
#include <evaluator_common.h>
#include <executor.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/registry.h>

#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>

#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace nvfuser {

class SegmentedGroup;
class FusionHeuristics;
class SchedulerRuntimeInfo;

// Utilities for benchmarking and profiling
struct ExecutorLog {
  std::shared_ptr<HeuristicParams> params = nullptr;
  FusionExecutor* fusion_executor = nullptr;
};

struct RuntimeWorkSpace {
  //! Pre-determined order to run the segmented groups
  std::vector<SegmentedGroup*> group_run_order;

  //! Pre-determined order to bind tensor input meta data
  std::vector<Val*> group_extent_binding_order;
};
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

//! FusionKernelRuntime is the unified interface from fusion graphs into
//!  caching, compilation into kernels, and kernel launches.
//!
//! Each instance is also a cache entry tracked by FusionKernelRuntimeCache.
//!
//! Two types of instance can be created, one for complete/single-kernel fusion
//!  and one for segmented/multi-kernel fusion.
//! Conceptually this is a generalization of FusionExecutor that supports both
//!  single-kernel and multi-kernel caching/compiling/launching
class TORCH_CUDA_CU_API FusionKernelRuntime {
 public:
  explicit FusionKernelRuntime(
      std::unique_ptr<Fusion> fusion,
      const KernelArgumentHolder& inputs,
      std::optional<PrimDataType> forced_index_type = std::nullopt);

  //! Type notations within FusionKernelRuntime Context
  using HashType = size_t;
  using SchedulerEntryPtr = std::unique_ptr<SchedulerEntry>;

  //! Evicts internally cached parameters based on input sizes.
  //!  An interface used by runtime caches.
  void evictCache(size_t input_id) {
    for (auto& fe : executors_) {
      fe.evictCache(input_id);
    }
  }

  //! query if we already have a compiled kernel for execution
  bool isCompiled() {
    std::lock_guard<std::mutex> guard(mutex_);
    return std::all_of(
        executors_.begin(), executors_.end(), [](const auto& executor) {
          return executor.compiled();
        });
  }

  //! Note that all heuristics use the same index type.
  PrimDataType getIndexType() const {
    // No scheduler means nothing to run. It may still be unsafe to
    // save tensor sizes and strides in Int32
    if (schedulers().empty()) {
      return PrimDataType::Int;
    }
    auto index_type = schedulers().at(0).get()->params()->cparams.index_type;
    TORCH_INTERNAL_ASSERT(index_type.has_value());
    return index_type.value();
  }

  //! Unified interface to run the managed kernels with given input
  std::vector<at::Tensor> runWithInputs(KernelArgumentHolder& args);

  //! Compile a kernel executor for given inputs. Note: The compilation is
  //! multithreaded. The segments in the fusion are compiled independently.
  void compileFusionParallel(KernelArgumentHolder args);

  const std::vector<int64_t>& getArgsNumAfterSegmentRuns() {
    return num_live_args_after_segment_runs_;
  }

  //! Turn On/Off profiling
  void profile(bool to_profile = true) {
    profiling_ = to_profile;
  }

  //! Enable kernel time measurement. Only the device time is
  //! inclued.
  void enableKernelTimeMeasurement() {
    measure_kernel_time_ = true;
  }

  void disableKernelTimeMeasurement() {
    measure_kernel_time_ = false;
  }

  //! Return the total kernel time of all segments
  float kernelTimeMs() const {
    return kernel_time_ms_;
  }

  //! Internal knob for profiling shape inference
  void disableLaunchParamCache() {
    for (auto& executor : executors_) {
      executor.disableLaunchParamCache();
    }
  }

  //! Internal knob for profiling shape inference
  void disableKernelLaunch() {
    for (auto& executor : executors_) {
      executor.setExecuteKernelFlag(false);
    }
  }

  //! Returns if this runtime is segmented
  bool isSegmented() {
    return is_segmented_;
  }

  //! Returns the fusion segments if applicable
  SegmentedFusion* fusionSegments() {
    return segmented_fusion_.get();
  }

  //! Returns the list of heuristics in this runtime
  FusionHeuristics* schedulerHeuristics() {
    return heuristics_.get();
  }

  //! Return the most recently used executor, corresponding to the
  //!  most recent kernel launch.
  //! TODO: have a interface for grabbing all recent logs. Need to put a buffer
  //! space for recent logs
  ExecutorLog getMostRecentExecutorLog() {
    TORCH_INTERNAL_ASSERT(
        profiling_, "Executor log is only produced in profiling mode");
    return most_recent_executor_log_;
  }

  // Try to compute heuristics based on the SegmentedFusion managed
  //  in this kernel runtime, and will return a nullopt if either
  //  any segment cannot be scheduled or the parameters don't match
  //
  // Heuristics must use the index type of forced_index_type if given.
  using HeuristicsPtr = std::unique_ptr<FusionHeuristics>;
  std::optional<HeuristicsPtr> getMaybeHeuristicsFor(
      const KernelArgumentHolder& args,
      std::optional<PrimDataType> forced_index_type = std::nullopt);

  //! Copy the launch params given in the parameter heuristics to prepare
  //!  for kernel launch for a new input dimension but same heuristics
  void updateHeuristicsLaunchParams(FusionHeuristics* update_heuristics);

  const std::vector<FusionExecutor>& executors() const {
    return executors_;
  }

 private:
  //! Runs each fusion segment given arguments. The outputs for a fusion are
  //! added back to the arguments, so they can be used as inputs to successive
  //! segments. Returns a map that links each NvFuser Val to its corresponding
  //! tensor.
  std::unordered_map<Val*, const ArgAbstract*> runSegmentsWithInputs(
      KernelArgumentHolder& args);

  //! Interface to run a single kernel, either one kernel for single-kernel
  //! fusions, or a kernel for a segmentedGrouup in a segmented fusion. Returns
  //! the kernel outputs.
  std::vector<at::Tensor> runKernelWithInput(
      KernelArgumentHolder& args,
      SegmentedGroup* sg);

  //! Interface to compile a single kernel. It is either a single kernel for a
  //! fusion or a kernel for a segmentedGrouup in a segmented fusion. Returns
  //! launch and compile parameters for kernel.
  void compileKernel(const KernelArgumentHolder& args, SegmentedGroup* sg);

  std::pair<LaunchParams, CompileParams> getKernelConfig(
      const KernelArgumentHolder& args,
      SegmentedGroup* sg);

  //! Access the list of schedulers maintained in this runtime instance
  const std::vector<SchedulerEntryPtr>& schedulers() const;

  void prepareRuntimeOrder();

 private:
  //! Entries indexed by groupID:
  //! Executors holding compiled kernels
  std::vector<FusionExecutor> executors_;

  //! Heuristics object holding scheduler entries for all segments
  std::unique_ptr<FusionHeuristics> heuristics_;

  // Checks if this runtime instance is for a single-kernel fusion (false) or a
  //  segmented fusion (true).
  bool is_segmented_ = true;

  //! Multi-Kernel fusion segment when applies
  std::unique_ptr<SegmentedFusion> segmented_fusion_ = nullptr;

  //! Pre-allocated runtime workspace to speed up kernel launch preparation.
  RuntimeWorkSpace runtime_workspace_;

  //! Utility to speed up value evaluation at runtime
  std::unique_ptr<PrecomputedValues> precomputed_values_;

  //! Cache of all tensors in the complete fusion
  std::vector<TensorView*> all_tvs_;

  //! store number of arguments in KernelArgumentHolder after each segment
  //! used to check if arguments are erased if not being used in the following
  //! segments
  std::vector<int64_t> num_live_args_after_segment_runs_;

  // States for profiling support
  bool profiling_ = false;

  //! Flag to indicate kernel timing measurement. Should be disabled
  //! unless benchmarking the kernel timing only as the measurement
  //! itself incurs an overhead.
  bool measure_kernel_time_ = false;

  //! The sum of the last kernel execution times
  float kernel_time_ms_ = 0;

  std::mutex mutex_;

  // The heuristics and executor for most recent kernel launch
  ExecutorLog most_recent_executor_log_;
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
class TORCH_CUDA_CU_API InputsIdLookup : public NonCopyable {
 public:
  //! constructor where maximum cache size is fixed during init
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  explicit InputsIdLookup(size_t max_cache_size = 100)
      : max_cache_size_(max_cache_size){};

  //! struct to hold return value for lookupId.
  struct IdLookupReturn {
    size_t id = 0;
    size_t evict_id = 0;
    bool eviction = false;
  };

  //! Encode each input sets to with an unique id;
  //! The returned data structure also indicates whether eviction has happened
  //! within the lookup cache. This is needed because lookup shortcut is also
  //! cached in nested `GraphCache`, `FusionExecutorCache` and `FusionExecutor`.
  //! see [ Note -- Post-definition cache implementation ] and [ Note -- 2 level
  //! cache implementation ].
  //!
  //! In the presence of dynamic operations like reshape and resize, the
  //! structure of the concretized Fusion might depend on not only the extents
  //! of input tensors, but on input scalars. For example,
  //!
  //!    auto s = IrBuilder::create<int>();
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
  IdLookupReturn lookupId(
      const at::ArrayRef<c10::IValue>& inputs,
      const std::unordered_set<size_t>& scalar_inputs_to_record = {},
      int8_t device = 0);

  //! debugging API that returns the size of lookup table
  size_t size() const {
    return encoding_lookup_.size();
  }

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
  const size_t max_cache_size_;

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

//! [ Note -- Post-definition cache implementation ]
//!
//! First note that depending on how we acquire a computational graph, there may
//! be additional levels of caching above those discussed in this note. For
//! example, our Python frontend contains a cache of defined Fusions in order to
//! speed up acquisition of a Fusion object in its designed use cases. In this
//! note, we will discuss caching that occurs below the definition level,
//! assuming there is another mechanism that builds a Fusion object and passes
//! it to us.
//!
//! The primary interface to post-definition caching is the
//! `FusionExecutorCache`. This class holds an unsegmented, unscheduled Fusion
//! object that might contain symbolic operations. This Fusion is then evaluated
//! using `FusionExecutorCache::runFusionWithInputs` to produce outputs in the
//! form of ATen Tensors.
//!
//! FusionKernelRuntime is responsible for segmentation and execution of a
//! single concretized Fusion object with a given set of inputs. Each
//! FusionKernelRuntime is valid only for a given concrete Fusion and applies
//! only to a range of input properties. If the properties of some inputs
//! change, we can check if a Fusion can be used with the new inputs, but if not
//! then a new FusionKernelRuntime would need to be created, which means a whole
//! new segmentation run. No additional caching is performed beneath the
//! FusionKernelRuntime level, so caching is implemented in
//! FusionExecutorCache::getKernelRuntimeFor to reduce the latency in mapping
//! from a set of inputs to a valid FusionKernelRuntime object.
//!
//! The content of input tensors does not affect the structure of the Fusion
//! graph or the validity of compiled CUDA kernels. However, the following other
//! properties might: rank, DataType, contiguity, stride order, size (whether a
//! dimension has size=1). When all of these properties are repeated, there is
//! an opportunity to reduce the latency of producing a compiled Fusion and
//! launch params (a FusionExecutor). Given inputs, we first compute an ID using
//! InputsIdLookup::lookupId that encodes tensor properties along with values of
//! any integer-valued input scalars that might affect concretization. This ID
//! is guaranteed not to conflict unless the inputs can be executed by the same
//! compiled Fusion. It is mapped to a segmented and compiled
//! FusionKernelRuntime that can be immediately run. This is the most common,
//! lowest-latency path and is followed when the Fusion is repeatedly evaluated
//! with inputs that differ only in their tensor values.
//!
//! When there is no FusionKernelRuntime matching the given input ID, we first
//! map the inputs to a concrete Fusion. For static Fusions this is trivial,
//! but when the Fusion is dynamic it means we must use the inputs to
//! "concretize" the Fusion by performing replacements such that the Fusion no
//! longer contains Symbolic IterDomains. A static Fusion is considered to be
//! already concretized. As discussed above, each concretized Fusion might have
//! multiple FusionKernelRuntimes applying to different ranges of inputs.
//! In the second layer of post-definition caching, we map concretized Fusions
//! (in the form of a DynamicTransformConcretizationInfo object) to a vector of
//! FusionKernelRuntimes, and check whether each one is able to run the present
//! inputs. If not, we create a new FusionKernelRuntime for those inputs and
//! record it in the list (as well as recording a mapping from the input ID to
//! the new runtime).
//!
//! In the case of a dynamic Fusion, input scalars such as integer parameters to
//! a model could potentially affect the structure of a concretized Fusion, so
//! we take care to include those scalars in the input ID along with the extents
//! of tensor arguments.
//!
//! * note on unique computational graph
//! In theory, computational graph should refer to only the computational nodes
//! in a subgraph and should remain agnostic to input meta info, like
//! shape, strides, type e.t.c.. However, the contract right here is fuzzy.
//! Different executor applies their own protocol of what is a unique
//! computational graph. e.g. Legacy Executor embeds tensor type &
//! dimensionality in the graph, while Profiling Executor keeps symbolic shape
//! as well as stride order in the graph as well.
//!
//! Our definition of a "unique" computational graph is aligned with `Fusion`
//! IR, hence the requirement extends to meta information on input tensors.
//! Which means, for each input tensor, following properties are fixed:
//!     a) stride order;
//!     b) contiguity information;
//!     c) broadcasting semantics (size-1 or not);
//!     d) rank;
//!     e) scalar type;
//!
//! [ Note -- Segmented Fusion Tentative Design ]
//! Segmentation adds an extra dimension in caching. Initial implementation,
//! assumed graph partition strategy is independent of input pattern, which we
//! can revisit once we have more advanced graph segmentation logic Each
//! FusionExecutorCache corresponds to one graph and one graph segmentation.
class TORCH_CUDA_CU_API FusionExecutorCache {
 public:
  //! create new fusion executor cache at a given device to handle kernel
  //! generation of dynamic sizes
  //! fusion executor is taking the ownership of `fusion`
  explicit FusionExecutorCache(std::unique_ptr<Fusion> fusion);

  //! Execute fusion graph with given inputs, create `FusionExecutor` as needed
  //! Note this function also handles permutation & input update outside of
  //! codegen.
  //!
  //! If given, the index type of forced_index_type is used no matter
  //! what inputs and the fusion look like. This may be useful in some
  //! cases as our analysis of index type may be overly conservative
  //! for intermediate tensors.
  //! WARING: Correctness is not guaranteed.
  std::vector<at::Tensor> runFusionWithInputs(
      const at::ArrayRef<c10::IValue>& inputs,
      std::optional<PrimDataType> forced_index_type = std::nullopt,
      std::optional<int8_t> selected_device = std::nullopt);

  //! Converts inputs from IValue to KernelArgumentHolder, also handles cache
  //! lookup
  KernelArgumentHolder prepareInputs(
      const at::ArrayRef<c10::IValue>& inputs,
      std::optional<int8_t> selected_device = std::nullopt);

  //! query if there's a kernel ready to go for given inputs
  bool isCompiled(const at::ArrayRef<c10::IValue>& inputs, int8_t device = 0);

  Fusion* fusion() {
    return fusion_.get();
  }

  const Fusion* fusion() const {
    return fusion_.get();
  }

  void printFusion() {
    fusion_->printMath();
  }

  FusionKernelRuntime* getMostRecentKernelRuntime() const {
    return most_recent_runtime_;
  }

  //! Gets the kernel code for the associated runtime
  std::string getCode(
      FusionKernelRuntime* kernel_runtime,
      bool instrinsic_code = false) const;
  //! Get the most recently executed kernel code
  std::string getMostRecentCode(bool instrinsic_code = false) const;
  //! Get the kernel code for the given inputs
  std::string getCodeFor(
      const at::ArrayRef<c10::IValue>& inputs,
      bool intrinsic_code);
  //! Gets the Scheduled IR for the associated runtime
  std::string getScheduledIr(
      FusionKernelRuntime* kernel_runtime,
      bool tensor_transforms = false) const;
  //! Get the most recently executed Scheduled IR
  std::string getMostRecentScheduledIr(bool tensor_transforms = false) const;
  //! Get the Scheduled IR for the given inputs
  std::string getScheduledIrFor(
      const at::ArrayRef<c10::IValue>& inputs,
      bool tensor_transforms = false);

  // TODO: in a follow up we need a global logging structure
  //  to capture runtime profiling info. We also need to define
  //  a suitable profiling window / buffer size.
  ExecutorLog getMostRecentExecutorInfo() {
    TORCH_INTERNAL_ASSERT(most_recent_runtime_ != nullptr);
    return most_recent_runtime_->getMostRecentExecutorLog();
  }

  //! Get all cached runtimes
  const auto& getKernelRuntimes() const {
    return kernel_runtimes_;
  }

  //! Count concretizations. Note that each might have multiple
  //! FusionKernelRuntimes. If device is given, count only concretizations on
  //! the given device; otherwise count concretizations on all devices.
  size_t countConcretizations(int8_t device = -1) const {
    size_t concs = 0;
    for (auto& it : kernel_runtimes_) {
      if (device >= 0 && it.first.first != device) {
        continue;
      }
      concs++;
    }
    return concs;
  }

  //! Count kernel runtimes across all concretizations. If device is given,
  //! count only runtimes on the given device; otherwise count
  //! runtimes on all devices.
  size_t countRuntimes(int8_t device = -1) const {
    size_t runtimes = 0;
    for (auto& it : kernel_runtimes_) {
      if (device >= 0 && it.first.first != device) {
        continue;
      }
      runtimes += it.second.size();
    }
    return runtimes;
  }

  void profile(bool to_profile) {
    profiling_ = to_profile;
    for (auto& it : kernel_runtimes_) {
      for (auto& kernel_runtime : it.second) {
        kernel_runtime->profile(to_profile);
      }
    }
  }

  //! Internal knob for profiling shape inference
  void disableLaunchParamCache() {
    for (auto& it : kernel_runtimes_) {
      for (auto& kernel_runtime : it.second) {
        kernel_runtime->disableLaunchParamCache();
      }
    }
  }

  //! Internal knob for profiling shape inference
  void disableKernelLaunch() {
    for (auto& it : kernel_runtimes_) {
      for (auto& kernel_runtime : it.second) {
        kernel_runtime->disableKernelLaunch();
      }
    }
  }

  //! Enable kernel time measurement through FusionKernelRuntime. See
  //! FusionKernelRuntime::enableKernelTimeMeasurement() as well
  void enableKernelTimeMeasurement() {
    measure_kernel_time_ = true;
  }

  void disableKernelTimeMeasurement() {
    measure_kernel_time_ = false;
  }

  //! Return the kernel time of the most recent fusion execution. Can
  //! be zero if the measurement is not enabled
  float getMostRecentKernelTimeMs() const {
    auto rt = getMostRecentKernelRuntime();
    TORCH_INTERNAL_ASSERT(rt != nullptr);
    return rt->kernelTimeMs();
  }

  //! Allocate the outputs of the Fusion given inputs
  //! TODO: re-implement
  std::vector<at::Tensor> allocOutputSpace(
      const at::ArrayRef<c10::IValue>& inputs) {
    return runFusionWithInputs(inputs);
  }

 private:
  //! evict cached short cut entry in `code_to_fe_lookup_` as well as cached
  //! entry in `FusionExecutor`
  void evictCache(size_t cache_id);

  //! The index type of forced_index_type is used to get a kernel
  //! runtime no matter what sizes inputs have
  FusionKernelRuntime* getKernelRuntimeFor(
      const KernelArgumentHolder& inputs,
      std::optional<PrimDataType> forced_index_type = std::nullopt);

  //! Get initial concretization info (without inputs). This computes the info
  //! if it has not yet been computed, then caches it for later use. This means
  //! this method should not be called until the definition of the Fusion is
  //! finalized.
  DynamicTransformInitialInfo& initialInfo();

 private:
  //! original un-scheduled `Fusion`. This may contain dynamic transforms and
  //! Symbolic IterDomains.
  std::unique_ptr<Fusion> fusion_;

  //! inputs to unique_id lookup table;
  InputsIdLookup inputs_id_lookup_;

  //! Holds FusionKernelRuntime for scheduled, static Fusions. The key in this
  //! map is a (device, concretization info) pair. In case fusion_ contains
  //! no dynamic transforms, the second part of the key is null. When a new set
  //! of inputs is received, we extract the corresponding value from this map,
  //! which is a vector of FusionKernelRuntime objects representing scheduled
  //! Fusions. We then check each of these to see if we can re-use any of those
  //! kernels and if not, we create a new one.
  std::unordered_map<
      std::pair<int8_t, const DynamicTransformConcretizationInfo*>,
      std::vector<std::unique_ptr<FusionKernelRuntime>>,
      PairPointerHash,
      PairPointerEquals>
      kernel_runtimes_;

  //! This class owns the initial info and concretization info associated to
  //! each vector of kernel runtimes
  std::vector<std::unique_ptr<DynamicTransformInitialInfo>>
      cached_initial_info_;
  std::vector<std::unique_ptr<DynamicTransformConcretizationInfo>>
      cached_conc_info_;

  //! Logging state for most recent compilation
  bool profiling_ = false;

  //! Flag to indicate kernel time measurement
  bool measure_kernel_time_ = false;

  //! Logging state for most recent compilation
  ExecutorLog most_recent_executor_log_;

  //! short-cut for cache hit
  std::unordered_map<size_t, FusionKernelRuntime*> id_to_kernel_runtime_;

  //! Profiling info:
  //! TODO: this can be largely expanded to look at complete
  //!   caching profiles. Currently it just makes it easier to test
  FusionKernelRuntime* most_recent_runtime_ = nullptr;

  //! Initial concretization info
  std::optional<DynamicTransformInitialInfo> initial_info_ = std::nullopt;
};

//! [ Note -- 2 level cache implementation ]
//!
//! Compiling PyTorch IR requires an addition translation to Fusion IR, which is
//! cached using `GraphCache`.
//!
//! 2 level hierarchically nested cache is to handle the code generation and
//! execution of a given PyTorch IR graph that is unique in its computational
//! graph (see note on unique computational graph down).
//!
//! The nested cache structures are:
//!     a. GraphCache
//!        - GraphCache translates PyTorch IR into Fusion IR and pass it to a
//!          `FusionExecutorCache`;
//!        - GraphCache assumes all inputs to comply with profiling information,
//!          mostly tensor size & contiguity (see note on unique computational
//!          graph). The assumption is assured at runtime by
//!          `prim::CudaFusionGuard`;
//!     b. FusionExecutorCache
//!        - has a single `Fusion`, FusionExecutorCache handles kernel schedule
//!          and passed scheduled tensor to `FusionExecutor` to generate code;
//!        - create `FusionExecutor` instances to handle heuristics from dynamic
//!          shape (varying tensor sizes);
//!        - create `FusionExecutor` instances to handle different devices;
//!        - holds input cache `InputsIdLookup`, which allow cache on heuristics
//!          and launch parameters to reduce latency.
//!
class GraphCache {
 public:
  //! TODO: we should probably change shared_ptr to unique_ptr, as we want to
  //!       claim the ownership of the computational graph.
  //! create GraphCache on a given graph;
  //! We extract global stride index order and translate PyTorch JIT IR to
  //! Fusion IR.
  explicit GraphCache(const std::shared_ptr<torch::jit::Graph>& graph);

  //! execute graph with given inputs
  std::vector<at::Tensor> runGraphWithInputs(
      const at::ArrayRef<c10::IValue>& inputs);

 private:
  //! construct FusionExecutorCache
  void createFusion(const std::shared_ptr<torch::jit::Graph>& graph);

 private:
  //! FusionExecutorCache that performs schedule and kernel execution;
  std::unique_ptr<FusionExecutorCache> fusion_executor_cache_;

  //! num of outputs
  size_t num_of_outputs_ = 0;
};

} // namespace nvfuser

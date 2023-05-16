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
//! Simple hasher for pair<T, U>. There is no default hasher for pairs, since
//! there are a lot of options how to combine hashes. In a case where one
//! element of the pair is unlikely to change much, the following hash is fast
//! and effective.
struct SimplePairHash {
  template <typename T, typename U>
  size_t operator()(const std::pair<T, U>& p) const {
    auto hT = std::hash<T>{}(p.first);
    auto hU = std::hash<U>{}(p.second);
    return hT ^ hU;
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
    std::unique_lock<std::mutex> lock0(mutex_, std::try_to_lock);
    std::unique_lock<std::mutex> lock1(compiling_, std::try_to_lock);
    if (!lock0.owns_lock() || !lock1.owns_lock()) {
      // compilation in progress
      return false;
    }

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

  const std::vector<int64_t>& getArgsNumAfterSegmentRuns() {
    return num_live_args_after_segment_runs_;
  }

  //! starts compilation async
  void startAsyncCompile(const KernelArgumentHolder& input_args);

  //! Turn On/Off profiling
  void profile(bool to_profile = true) {
    profiling_ = to_profile;
  }

  void setMeasureKernelTime(bool val = true) {
    measure_kernel_time_ = val;
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
  c10::optional<HeuristicsPtr> getMaybeHeuristicsFor(
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
  //! tensor. The is_dry_run flag determines if the ArgAbstract value maps to a
  //! real PyTorch tensor or a fake MetaData tensor.
  std::unordered_map<Val*, const ArgAbstract*> runSegmentsWithInputs(
      KernelArgumentHolder& args,
      bool is_dry_run);

  //! Interface to run a single kernel, either one kernel for single-kernel
  //! fusions, or a kernel for a segmentedGrouup in a segmented fusion. Returns
  //! the kernel outputs.
  std::vector<at::Tensor> runKernelWithInput(
      KernelArgumentHolder& args,
      SegmentedGroup* sg);

  //! Interface to compile a single kernel and returns the kernel outputs
  //! but the tensor does not own memory.
  KernelArgumentHolder dryRunKernelWithInput(
      const KernelArgumentHolder& args,
      SegmentedGroup* sg);

  //! Maps entries in `args` to fusion inputs.
  //! Note that this function also pushes extra bits like dimension extent into
  //! `args` for expression evaluator binding. So consider your `args` polluted
  //! after this function and use it with caution.
  std::unordered_map<Val*, const ArgAbstract*> mapFusionInputsToArgs(
      KernelArgumentHolder& args);

  //! Interface to compile a single kernel. It is either a single kernel for a
  //! fusion or a kernel for a segmentedGrouup in a segmented fusion. Returns
  //! launch and compile parameters for kernel.
  std::pair<LaunchParams, CompileParams> compileKernel(
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
  bool measure_kernel_time_ = false;

  std::mutex mutex_;

  //! A second mutex used in startAsyncCompile
  std::mutex compiling_;

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

  //! encode each input sets to with an unique id;
  //! Returned data structure also indicates whether eviction has happened
  //! within the lookup cache. This is needed because lookup shortcut is also
  //! cached in nested `GraphCache`, `FusionExecutorCache` and `FusionExecutor`.
  //! see [ Note -- 2 level cache implementation ]
  //! If hash_scalars is true, this unique id also contains the values of input
  //! integer scalars. This is used for dynamic reshapes, since they might
  //! depend on those inputs and omitting them would lead to a collision.
  IdLookupReturn lookupId(
      const at::ArrayRef<c10::IValue>& inputs,
      bool hash_scalars = false);

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

//! [ Note -- 2 level cache implementation ]
//!
//! We have 2 level cache for a separation in function to keep them simpler.
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
//!
//! [ Note -- Segmented Fusion Tentative Design ]
//! Segmentation adds an extra dimension in caching. Initial implementation,
//! assumed graph partition strategy is independent of input pattern, which we
//! can revisit once we have more advanced graph segmentation logic Each
//! FusionExecutorCache corresponds to one graph and one graph segmentation.
//!
//!
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
      std::optional<PrimDataType> forced_index_type = std::nullopt);

  //! Compile a kernel executor for given inputs. Note: The compilation is
  //! async, there's some restriction on the user side. e.g. Do not overlap
  //! compilation and execution for the same FusionExecutor entry. This is
  //! experimental at this moment, please use with extra caution.
  void compileFusionAsync(const at::ArrayRef<c10::IValue>& inputs);

  //! Converts inputs from IValue to KernelArgumentHolder, also handles cache
  //! lookup
  KernelArgumentHolder prepareInputs(const at::ArrayRef<c10::IValue>& inputs);

  //! query if there's a kernel ready to go for given inputs
  bool isCompiled(const at::ArrayRef<c10::IValue>& inputs);

  Fusion* fusion() {
    return fusion_.get();
  }

  const Fusion* fusion() const {
    return fusion_.get();
  }

  void printFusion() {
    fusion_->printMath();
  }

  FusionKernelRuntime* getMostRecentKernelRuntime() {
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
  auto& getKernelRuntimes() {
    return kernel_runtimes_;
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

 private:
  //! evict cached short cut entry in `code_to_fe_lookup_` as well as cached
  //! entry in `FusionExecutor`
  void evictCache(size_t cache_id);

  //! The index type of forced_index_type is used to get a kernel
  //! runtime no matter what sizes inputs have
  FusionKernelRuntime* getKernelRuntimeFor(
      const KernelArgumentHolder& inputs,
      std::optional<PrimDataType> forced_index_type = std::nullopt);

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
      std::pair<size_t, std::optional<DynamicTransformConcretizationInfo>>,
      std::vector<std::unique_ptr<FusionKernelRuntime>>,
      SimplePairHash>
      kernel_runtimes_;

  //! Logging state for most recent compilation
  bool profiling_ = false;

  //! Logging state for most recent compilation
  ExecutorLog most_recent_executor_log_;

  //! short-cut for cache hit
  std::unordered_map<size_t, FusionKernelRuntime*> id_to_kernel_runtime_;

  //! Profiling info:
  //! TODO: this can be largely expanded to look at complete
  //!   caching profiles. Currently it just makes it easier to test
  FusionKernelRuntime* most_recent_runtime_ = nullptr;

  //! Whether fusion_ contains dynamic reshapes
  bool has_dynamic_reshape_ = false;
};

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

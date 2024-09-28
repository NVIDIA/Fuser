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
#include <exceptions.h>
#include <fusion.h>
#include <fusion_executor/executor.h>
#include <fusion_segmenter.h>
#include <kernel_cache_utils.h>
#include <logical_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/heuristic.h>
#include <scheduler/registry.h>
#include <serde/fusion_cache_generated.h>

#include <c10/util/ArrayRef.h>

#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace nvfuser {

class HeuristicParamsList;
class SchedulerRuntimeInfo;
class SegmentedGroup;

//! FusionKernelRuntime is the unified interface from fusion graphs into
//!  caching, compilation into kernels, and kernel launches.
//!
//! Each instance is also a cache entry tracked by FusionKernelRuntimeCache.
//!
//! Two types of instance can be created, one for complete/single-kernel fusion
//!  and one for segmented/multi-kernel fusion.
//! Conceptually this is a generalization of FusionExecutor that supports both
//!  single-kernel and multi-kernel caching/compiling/launching
//!
//! When serde_buffer argument is a nullptr, we run the
//! SegmentCandidateFinder::segment pass in the constructor and compile the
//! fusions. When serde_buffer exists, we deserialize the segmented_fusion_ and
//! executors_ objects from the flatbuffer binary.
class FusionKernelRuntime {
 public:
  explicit FusionKernelRuntime(
      std::unique_ptr<Fusion> fusion,
      const KernelArgumentHolder& inputs,
      const serde::FusionKernelRuntime* serde_buffer = nullptr,
      std::optional<PrimDataType> forced_index_type = std::nullopt,
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      bool auto_schedule = true);

  //! Type notations within FusionKernelRuntime Context

  //! Evicts internally cached parameters based on input sizes.
  //!  An interface used by runtime caches.
  void evictCache(size_t input_id) {
    for (auto& fe : executors_) {
      fe.evictCache(input_id);
    }
  }

  //! query if we have already attempted compilation
  bool isCompiled() {
    std::lock_guard<std::mutex> guard(mutex_);
    return std::all_of(
        executors_.begin(), executors_.end(), [](const auto& executor) {
          return executor.isCompiled();
        });
  }

  //! Serialize Fusion Kernel Runtime using flatbuffers
  flatbuffers::Offset<serde::FusionKernelRuntime> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  //! Deserialize Fusion Kernel Runtime using flatbuffers
  void deserialize(
      const serde::FusionKernelRuntime* buffer,
      int8_t device_index);

  //! Note that all heuristics use the same index type.
  PrimDataType getIndexType() const {
    // No scheduler means nothing to run. It may still be unsafe to
    // save tensor sizes and strides in Int32
    if (schedulers().empty()) {
      return PrimDataType::Int;
    }
    auto index_type = schedulers().at(0).get()->cparams.index_type;
    NVF_ERROR(index_type.has_value());
    return index_type.value();
  }

  //! Unified interface to run the managed kernels with given input
  NVF_API std::vector<at::Tensor> runWithInputs(KernelArgumentHolder& args);

  //! Compile a kernel executor for given inputs. Note: The compilation is
  //! multithreaded. The segments in the fusion are compiled independently.
  NVF_API void compileFusionParallel(KernelArgumentHolder args);

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
  bool isSegmented() const {
    return is_segmented_;
  }

  //! Returns the fusion segments if applicable
  SegmentedFusion* fusionSegments() const {
    return segmented_fusion_.get();
  }

  //! Returns the list of heuristics in this runtime
  HeuristicParamsList* schedulerHeuristics() const {
    return heuristics_.get();
  }

  //! Return the most recently used executor, corresponding to the
  //!  most recent kernel launch.
  //! TODO: have a interface for grabbing all recent logs. Need to put a buffer
  //! space for recent logs
  const ExecutorLog& getMostRecentExecutorLog() const {
    NVF_ERROR(profiling_, "Executor log is only produced in profiling mode");
    return most_recent_executor_log_;
  }

  // Try to compute heuristics based on the SegmentedFusion managed
  //  in this kernel runtime, and will return a nullopt if either
  //  any segment cannot be scheduled or the parameters don't match
  //
  // Heuristics must use the index type of forced_index_type if given.
  NVF_API std::optional<std::unique_ptr<HeuristicParamsList>>
  getMaybeHeuristicsFor(
      const KernelArgumentHolder& args,
      std::optional<PrimDataType> forced_index_type = std::nullopt);

  //! Copy the launch params given in the parameter heuristics to prepare
  //!  for kernel launch for a new input dimension but same heuristics
  void updateHeuristicsLaunchParams(HeuristicParamsList* update_heuristics);

  const std::vector<FusionExecutor>& executors() const {
    return executors_;
  }

 private:
  //! Runs each fusion segment given arguments. The outputs for a fusion are
  //! added back to the arguments, so they can be used as inputs to successive
  //! segments. Returns a map that links each NvFuser Val to its corresponding
  //! tensor.
  std::unordered_map<Val*, const PolymorphicValue*> runSegmentsWithInputs(
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
  NVF_API const std::vector<std::unique_ptr<HeuristicParams>>& schedulers()
      const;

 private:
  //! Entries indexed by groupID:
  //! Executors holding compiled kernels
  std::vector<FusionExecutor> executors_;

  // A metadata copy of initial arguments used to contruct this
  // FusionKernelRuntime. Used during deserialization to schedule the fusion
  // rather than storing the scheduled fusion directly.
  KernelArgumentHolder args_metadata_;

  //! Heuristics object holding scheduler entries for all segments
  std::unique_ptr<HeuristicParamsList> heuristics_;

  // Checks if this runtime instance is for a single-kernel fusion (false) or a
  //  segmented fusion (true).
  bool is_segmented_ = true;

  //! Multi-Kernel fusion segment when applies
  std::unique_ptr<SegmentedFusion> segmented_fusion_ = nullptr;

  //! Pre-allocated runtime workspace to speed up kernel launch preparation.
  RuntimeWorkSpace runtime_workspace_;

  //! store number of arguments in KernelArgumentHolder after each segment
  //! used to check if arguments are erased if not being used in the following
  //! segments
  //! Only used in a single test: test_gpu3::FusionClearGmemBetweenSegments_CUDA
  std::vector<int64_t> num_live_args_after_segment_runs_;

  // States for profiling support
  bool profiling_ = false;

  //! Flag to indicate kernel timing measurement. Should be disabled
  //! unless benchmarking the kernel timing only as the measurement
  //! itself incurs an overhead.
  bool measure_kernel_time_ = false;

  //! The sum of the last kernel execution times
  float kernel_time_ms_ = 0;

  //! something to do with parallel compilation, not sure what it's actually
  //! being used to protect.
  std::mutex mutex_;

  // ID of fusion in python frontend fusion cache, which maps to a single
  // FusionExecutorCache.
  int64_t fusion_id_ = -1;

  // ID of concretized fusion in FusionExecutorCache
  int64_t concrete_id_ = -1;

  // ID of FusionKernelRuntime given (device, concrete_info) key
  int64_t runtime_id_ = -1;

  // The heuristics and executor for most recent kernel launch
  ExecutorLog most_recent_executor_log_;

  // Whether to auto schedule the Fusion. If set to false, scheduling is skipped
  const bool auto_schedule_;
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
class FusionExecutorCache {
 public:
  //! create new fusion executor cache at a given device to handle kernel
  //! generation of dynamic sizes
  //! fusion executor is taking the ownership of `fusion`
  NVF_API explicit FusionExecutorCache(
      std::unique_ptr<Fusion> fusion,
      int64_t fusion_id = 0,
      bool auto_schedule = true);

  //! Execute fusion graph with given inputs, create `FusionExecutor` as needed
  //! Note this function also handles permutation & input update outside of
  //! codegen.
  //!
  //! If given, the index type of forced_index_type is used no matter
  //! what inputs and the fusion look like. This may be useful in some
  //! cases as our analysis of index type may be overly conservative
  //! for intermediate tensors.
  //! WARING: Correctness is not guaranteed.
  //! TODO: Check usage of forced_index_type. It's a lot of plumbing, what's the
  //! value.
  NVF_API std::vector<at::Tensor> runFusionWithInputs(
      const at::ArrayRef<c10::IValue>& inputs,
      std::optional<PrimDataType> forced_index_type = std::nullopt,
      std::optional<int8_t> selected_device = std::nullopt);

  //! Converts inputs from IValue to KernelArgumentHolder, also handles cache
  //! lookup
  KernelArgumentHolder prepareInputs(
      const at::ArrayRef<c10::IValue>& inputs,
      std::optional<int8_t> selected_device = std::nullopt);

  //! query if there's a kernel ready to go for given inputs
  NVF_API bool isCompiled(
      const at::ArrayRef<c10::IValue>& inputs,
      int8_t device = 0);

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
  const ExecutorLog& getMostRecentExecutorInfo() {
    NVF_ERROR(most_recent_runtime_ != nullptr);
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
    NVF_ERROR(rt != nullptr);
    return rt->kernelTimeMs();
  }

  //! Serialize Fusion Executor Cache using flatbuffers
  flatbuffers::Offset<serde::FusionExecutorCache> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  //! Deserialize Fusion Executor Cache using flatbuffers
  void deserialize(const serde::FusionExecutorCache* buffer, int64_t fusion_id);

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

  using ConcreteInfo =
      std::pair<int8_t, const DynamicTransformConcretizationInfo*>;

  //! Holds FusionKernelRuntime for concretized and scheduled Fusions. The key
  //! in this map is a (device, concretization info) pair. In case fusion_
  //! contains no dynamic transforms, the second part of the key is null. When a
  //! new set of inputs is received, we extract the corresponding value from
  //! this map, which is a vector of FusionKernelRuntime objects representing
  //! scheduled Fusions. We then check each of these to see if we can re-use all
  //! of those kernels and if not, we create a new one.
  std::unordered_map<
      ConcreteInfo,
      std::vector<std::unique_ptr<FusionKernelRuntime>>,
      PairPointerHash,
      PairPointerEquals>
      kernel_runtimes_;

  //! This seems to just own the unique pointer of
  //! DynamicTransformConcretizationInfo which is implicitly being used for
  //! lifetime of entries in kernel_runtimes_. We should push the lifetime to
  //! the unordered_map if possible.
  std::vector<std::unique_ptr<DynamicTransformConcretizationInfo>>
      cached_conc_info_;

  //! Map each pair of device_id and concretization info to an integer id
  std::unordered_map<ConcreteInfo, int64_t, PairPointerHash, PairPointerEquals>
      conc_info_id_map_;
  //! For serialization, track a deterministic order for (device_id and
  //! concretization info) pair
  std::vector<ConcreteInfo> deterministic_conc_info_;

  //! This is cached to speed up finding concretization info
  std::unique_ptr<ExactLogicalDomainMap> exact_map_;

  //! Logging state for most recent compilation
  bool profiling_ = false;

  //! Flag to indicate kernel time measurement
  bool measure_kernel_time_ = false;

  //! Logging state for most recent compilation
  ExecutorLog most_recent_executor_log_;

  //! Short-cut for exact size cache hit
  std::unordered_map<size_t, FusionKernelRuntime*> id_to_kernel_runtime_;

  //! Profiling info:
  //! TODO: this can be largely expanded to look at complete
  //!   caching profiles. Currently it just makes it easier to test
  FusionKernelRuntime* most_recent_runtime_ = nullptr;

  //! Initial concretization info
  std::optional<DynamicTransformInitialInfo> initial_info_ = std::nullopt;

  // ID of fusion in python frontend fusion cache, which maps to a single
  // FusionExecutorCache.
  int64_t fusion_id_ = -1;

  // Whether to auto schedule the Fusion. If set to false, scheduling is skipped
  const bool auto_schedule_;
};

} // namespace nvfuser

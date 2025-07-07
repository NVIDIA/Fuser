// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/ArrayRef.h>

#include <fusion_segmenter.h>
#include <host_ir/executor.h>
#include <polymorphic_value.h>
#include <runtime/executor.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/fusion_cache_utils.h>

#include <mutex>
#include <vector>

namespace nvfuser {

class HeuristicParamsList;
enum class PrimDataType;
class Fusion;
class Val;
namespace serde {
struct FusionKernelRuntime;
}

//! FusionKernelRuntime is the unified interface from fusion graphs into
//!  caching, compilation into kernels, and kernel launches.
//!
//! Each instance is also a cache entry tracked by FusionKernelRuntimeCache.
//!
//! Two types of instance can be created, one for complete/single-kernel fusion
//!  and one for segmented/multi-kernel fusion.
//! Conceptually this is a generalization of KernelExecutor that supports both
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
  void evictCache(size_t input_id);

  //! query if we have already attempted compilation
  bool isCompiled() const;

  //! Serialize Fusion Kernel Runtime using flatbuffers
  flatbuffers::Offset<serde::FusionKernelRuntime> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  //! Deserialize Fusion Kernel Runtime using flatbuffers
  void deserialize(
      const serde::FusionKernelRuntime* buffer,
      int8_t device_index);

  //! Note that all heuristics use the same index type.
  PrimDataType getIndexType() const;

  //! Unified interface to run the managed kernels with given input
  NVF_API KernelArgumentHolder runWithInputs(const KernelArgumentHolder& args);

  //! Compile a kernel executor for given inputs. Note: The compilation is
  //! multithreaded. The segments in the fusion are compiled independently.
  NVF_API void compileFusionParallel(KernelArgumentHolder args);

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
  void disableKernelLaunch();

  //! Returns if this runtime is segmented
  bool isSegmented() const {
    return is_segmented_;
  }

  //! Returns the fusion segments if applicable
  SegmentedFusion* fusionSegments() const;

  //! Returns the list of heuristics in this runtime
  HeuristicParamsList* schedulerHeuristics() const;

  //! Return the most recently used executor, corresponding to the
  //!  most recent kernel launch.
  //! TODO: have a interface for grabbing all recent logs. Need to put a buffer
  //! space for recent logs
  const ExecutorLog& getMostRecentExecutorLog() const;

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

  const std::vector<std::unique_ptr<ExecutorAbstract>>& executors() const;

  #ifdef NVFUSER_ENABLE_HOST_IR_JIT
  const hir::HostIrJit& getHostIrJit() const {
    return *hie_.get();
  }
  #else
  const hir::HostIrEvaluator& getHostIrEvaluator() const {
    return *hie_.get();
  }
  #endif

 private:
  //! Runs each fusion segment given arguments. The outputs for a fusion are
  //! added back to the arguments, so they can be used as inputs to successive
  //! segments. Returns a map that links each NvFuser Val to its corresponding
  //! tensor.
  std::unordered_map<Val*, PolymorphicValue> runSegmentsWithInputs(
      const KernelArgumentHolder& args);

  //! Interface to run a single kernel, either one kernel for single-kernel
  //! fusions, or a kernel for a segmentedGrouup in a segmented fusion. Returns
  //! the kernel outputs.
  KernelArgumentHolder runKernelWithInput(
      const KernelArgumentHolder& args,
      SegmentedGroup* sg);

  //! Interface to compile a single kernel. It is either a single kernel for a
  //! fusion or a kernel for a segmentedGrouup in a segmented fusion. Returns
  //! launch and compile parameters for kernel.
  void compileKernel(
      const KernelArgumentHolder& args,
      SegmentedGroup* sg,
      hir::HostIrContainer* hic);

  std::pair<LaunchParams, CompileParams> getKernelConfig(
      const KernelArgumentHolder& args,
      SegmentedGroup* sg);

  //! Access the list of schedulers maintained in this runtime instance
  NVF_API const std::vector<std::unique_ptr<HeuristicParams>>& schedulers()
      const;

  // Create KernelArgumentHolders for all of the segments. Sorted in
  // the run order.
  std::vector<KernelArgumentHolder> prepareInputs(
      const KernelArgumentHolder& args) const;

  int64_t numGroups() const {
    int64_t n_groups = std::ssize(runtime_workspace_.group_run_order);
    NVF_ERROR_EQ(n_groups, std::ssize(segmented_fusion_->groups()));
    return n_groups;
  }

 private:
  //! Entries indexed by groupID:
  //! Executors holding compiled kernels
  std::vector<std::unique_ptr<ExecutorAbstract>> executors_;

  //! Host IR Evaluator
  #ifdef NVFUSER_ENABLE_HOST_IR_JIT
  std::unique_ptr<hir::HostIrJit> hie_;
  #else
  std::unique_ptr<hir::HostIrEvaluator> hie_;
  #endif




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
  mutable std::mutex mutex_;

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

} // namespace nvfuser

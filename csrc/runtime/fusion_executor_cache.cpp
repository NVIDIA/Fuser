// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <runtime/fusion_executor_cache.h>

#include <dynamic_transform.h>
#include <fusion.h>
#include <logical_domain_map.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/fusion_kernel_runtime.h>
#include <type.h>

#include <debug.h>
#include <dynamic_transform.h>
#include <fusion_profiler.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <multidevice/communicator.h>
#include <options.h>
#include <preseg_passes/pre_segmenter.h>
#include <runtime/allocations.h>
#include <runtime/executor_params.h>
#include <runtime/executor_utils.h>
#include <runtime/fusion_cache_utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/registry.h>
#include <utils.h>

namespace nvfuser {

FusionExecutorCache::FusionExecutorCache(
    std::unique_ptr<Fusion> fusion,
    int64_t fusion_id,
    bool auto_schedule)
    : fusion_(std::move(fusion)),
      exact_map_(fusion_.get()),
      fusion_id_{fusion_id},
      auto_schedule_(auto_schedule) {}

KernelArgumentHolder FusionExecutorCache::runFusionWithInputs(
    KernelArgumentHolder args,
    std::optional<PrimDataType> forced_index_type,
    std::optional<int8_t> selected_device) {
  FUSER_PERF_SCOPE("FusionExecutorCache::runFusionWithInputs");

  if (isProfilerEnabled()) {
    FusionProfiler::start(!isProfilerEnabledWithCupti());
  }

  args.setDeviceIndex(selected_device);
  setCacheId(args);
  auto kernel_runtime = getKernelRuntimeFor(args, forced_index_type);

  if (isProfilerEnabled()) {
    FusionProfiler::createSegments(kernel_runtime->executors().size());
  }

  if (!kernel_runtime->isCompiled()) {
    kernel_runtime->compileFusionParallel(args);
  }

  most_recent_runtime_ = kernel_runtime;

  auto fusion = kernel_runtime->fusionSegments()->completeFusion();

  // Make sure the forced index type is indeed used
  if (forced_index_type.has_value()) {
    NVF_ERROR(
        kernel_runtime->getIndexType() == forced_index_type.value(),
        "Enforcing index type of ",
        forced_index_type.value(),
        " failed");
  }

  auto outputs = kernel_runtime->runWithInputs(args);

  // Kernel time measurement is off by default
  kernel_runtime->disableKernelTimeMeasurement();

  // Removing aliased outputs, since those are updated by the Fusion. It is not
  // semantically correct to actually return them as outputs from
  // fusion.
  NVF_ERROR_EQ(std::ssize(fusion->outputs()), outputs.size());
  KernelArgumentHolder unaliased_outputs;
  for (auto out_index : arange(outputs.size())) {
    Val* out = fusion->outputs()[out_index];
    if (!fusion->getOutputAlias(out).hide_output) {
      unaliased_outputs.push(outputs[out_index]);
    }
  }

  // NOTE: This should be the last code in the method to capture all host time
  if (isProfilerEnabled()) {
    FusionProfiler::stop();
  }
  if (isProfilerPrintingEnabled()) {
    debug() << FusionProfiler::profile();
  }

  return unaliased_outputs;
}

void FusionExecutorCache::setCacheId(KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionExecutorCache::setCacheId");
  // TODO: move InputsIdLookup inside KernelArgumentHolder;
  // NOTE: We must ensure that the cache id is in fact unique. Dynamic fusions
  // may contain transformations that depend on input scalars, not just on the
  // extents of tensor inputs, so we must at times include those scalars in the
  // unique id. Currently, we include all integer scalar inputs for dynamic
  // fusions. This may not be ideal in all cases, since it will prevent
  // short-circuiting here, resulting in avoidable rebuilds of concretization
  // info.
  auto id_lookup_ret = inputs_id_lookup_.lookupId(
      args, initialInfo().scalarInputsAffectingConcretization());
  if (id_lookup_ret.eviction) {
    evictCache(id_lookup_ret.evict_id);
  }

  args.setCacheId(id_lookup_ret.id);
}

bool FusionExecutorCache::isCompiled(
    const KernelArgumentHolder& inputs,
    int8_t device) {
  FUSER_PERF_SCOPE("FusionExecutorCache::isCompiled");

  // Access kernels associated with the common device id
  KernelArgumentHolder args(inputs);
  setCacheId(args);
  return getKernelRuntimeFor(args)->isCompiled();
}

Fusion* FusionExecutorCache::fusion() {
  return fusion_.get();
}

const Fusion* FusionExecutorCache::fusion() const {
  return fusion_.get();
}

void FusionExecutorCache::printFusion() {
  fusion_->printMath();
}

FusionKernelRuntime* FusionExecutorCache::getMostRecentKernelRuntime() const {
  return most_recent_runtime_;
}

std::string FusionExecutorCache::getCode(
    FusionKernelRuntime* kernel_runtime,
    bool intrinsic_code) const {
  std::string kernel_code;
  NVF_CHECK(kernel_runtime != nullptr, "Invalid fusion definition!");
  NVF_CHECK(kernel_runtime->isCompiled(), "Fusion is not compiled!");

  bool first_kernel = true;
  for (const auto& ea : kernel_runtime->executors()) {
    if (auto ke = dynamic_cast<KernelExecutor*>(ea.get())) {
      if (first_kernel) {
        first_kernel = false;
      } else {
        kernel_code += "\n";
      }
      kernel_code += ke->compiledKernel()->kernelString();
    }
  }

  if (intrinsic_code) {
    const auto& execs = kernel_runtime->executors();
    const KernelExecutor* first_ke = nullptr;
    auto first_index_type = PrimDataType::Null;
    // Make sure all the segment index types match. All segments currently
    // use the same index type but this could change in the future.
    for (const auto& ea : execs) {
      if (auto ke = dynamic_cast<KernelExecutor*>(ea.get())) {
        if (first_ke == nullptr) {
          first_ke = ke;
        }
        auto cur_index_type = ke->compiledKernel()->kernel()->indexType();
        if (first_index_type == PrimDataType::Null) {
          first_index_type = cur_index_type;
        }
        NVF_CHECK(
            first_index_type == cur_index_type,
            "Index Type mismatch between Segment Executors: ",
            first_index_type,
            " ",
            cur_index_type);
      }
    }
    if (first_ke != nullptr) {
      return first_ke->compiledKernel()->getStructuredCode();
    }
    return "";
  } else {
    return kernel_code;
  }
}

std::string FusionExecutorCache::getMostRecentCode(bool intrinsic_code) const {
  return getCode(most_recent_runtime_, intrinsic_code);
}

std::string FusionExecutorCache::getCodeFor(
    KernelArgumentHolder args,
    bool intrinsic_code) {
  setCacheId(args);
  auto kernel_runtime = getKernelRuntimeFor(args);
  return getCode(kernel_runtime, intrinsic_code);
}

std::string FusionExecutorCache::getScheduledIr(
    FusionKernelRuntime* kernel_runtime,
    bool tensor_transforms) const {
  NVF_CHECK(kernel_runtime != nullptr, "Invalid fusion definition!");
  NVF_CHECK(kernel_runtime->isCompiled(), "Fusion is not compiled!");
  std::stringstream ss;
  if (kernel_runtime->isSegmented()) {
    auto fs = kernel_runtime->fusionSegments();
    ss << "Segmented_Fusion Dump: -- Re-written complete fusion:{\n";
    fs->completeFusion()->print(ss, false);
    ss << "} // {Re-written complete fusion}\n";
    ss << fs << "\n";
  }
  for (auto& ea : kernel_runtime->executors()) {
    if (auto ke = dynamic_cast<KernelExecutor*>(ea.get())) {
      auto sched_ir = ke->compiledKernel()->kernel()->as<Fusion>();
      sched_ir->print(ss, tensor_transforms);
    }
  }
  return ss.str();
}

std::string FusionExecutorCache::getMostRecentScheduledIr(
    bool tensor_transforms) const {
  return getScheduledIr(most_recent_runtime_, tensor_transforms);
}

std::string FusionExecutorCache::getScheduledIrFor(
    KernelArgumentHolder args,
    bool tensor_transforms) {
  setCacheId(args);
  auto kernel_runtime = getKernelRuntimeFor(args);
  return getScheduledIr(kernel_runtime, tensor_transforms);
}

// TODO: in a follow up we need a global logging structure
//  to capture runtime profiling info. We also need to define
//  a suitable profiling window / buffer size.
const ExecutorLog& FusionExecutorCache::getMostRecentExecutorInfo() {
  NVF_ERROR(most_recent_runtime_ != nullptr);
  return most_recent_runtime_->getMostRecentExecutorLog();
}

//! Get all cached runtimes
const std::unordered_map<
    FusionExecutorCache::ConcreteInfo,
    std::vector<std::unique_ptr<FusionKernelRuntime>>,
    PairPointerHash,
    PairPointerEquals>&
FusionExecutorCache::getKernelRuntimes() const {
  return kernel_runtimes_;
}

//! Count concretizations. Note that each might have multiple
//! FusionKernelRuntimes. If device is given, count only concretizations on
//! the given device; otherwise count concretizations on all devices.
size_t FusionExecutorCache::countConcretizations(int8_t device) const {
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
size_t FusionExecutorCache::countRuntimes(int8_t device) const {
  size_t runtimes = 0;
  for (auto& it : kernel_runtimes_) {
    if (device >= 0 && it.first.first != device) {
      continue;
    }
    runtimes += it.second.size();
  }
  return runtimes;
}

void FusionExecutorCache::profile(bool to_profile) {
  profiling_ = to_profile;
  for (auto& it : kernel_runtimes_) {
    for (auto& kernel_runtime : it.second) {
      kernel_runtime->profile(to_profile);
    }
  }
}

//! Internal knob for profiling shape inference
void FusionExecutorCache::disableLaunchParamCache() {
  for (auto& it : kernel_runtimes_) {
    for (auto& kernel_runtime : it.second) {
      NVF_CHECK(
          kernel_runtime->isCompiled(),
          "Tried to set parameters of executors before they were initialized.");
      for (auto& executor : kernel_runtime->executors()) {
        if (auto ke = dynamic_cast<KernelExecutor*>(executor.get())) {
          NVF_CHECK(
              ke->compiledKernel(),
              "Tried to disable parameter cache of uninitialized "
              "CompiledKernel.");
          ke->compiledKernel()->disableLaunchParamCache();
        }
      }
    }
  }
}

//! Internal knob for profiling shape inference
void FusionExecutorCache::disableKernelLaunch() {
  for (auto& it : kernel_runtimes_) {
    for (auto& kernel_runtime : it.second) {
      kernel_runtime->disableKernelLaunch();
    }
  }
}

//! Return the kernel time of the most recent fusion execution. Can
//! be zero if the measurement is not enabled
float FusionExecutorCache::getMostRecentKernelTimeMs() const {
  auto rt = getMostRecentKernelRuntime();
  NVF_ERROR(rt != nullptr);
  return rt->kernelTimeMs();
}

flatbuffers::Offset<serde::FusionExecutorCache> FusionExecutorCache::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See definitions in serde/fusion_cache.fbs for tables
  // FusionExecutorCache and KernelRuntimes

  // For serialization, we require a consistent ordering for the
  // kernel_runtimes_ map.
  std::unordered_map<FusionKernelRuntime*, size_t> kernel_cache_ordering;

  // 1. For each [device, concretization_info] key, serialize its vector of
  // FusionKernelRuntime objects
  std::vector<flatbuffers::Offset<serde::KernelRuntimeState>>
      fb_kernel_runtimes;
  fb_kernel_runtimes.reserve(kernel_runtimes_.size());

  for (const auto& device_concrete_key : deterministic_conc_info_) {
    const auto& device_runtimes = kernel_runtimes_.at(device_concrete_key);
    std::vector<flatbuffers::Offset<serde::FusionKernelRuntime>>
        fb_device_runtimes;
    fb_device_runtimes.reserve(device_runtimes.size());

    for (auto kernel_idx : arange(device_runtimes.size())) {
      auto kernel_runtime_ptr = device_runtimes.at(kernel_idx).get();
      fb_device_runtimes.push_back(kernel_runtime_ptr->serialize(builder));

      // Assign each runtime pointer an integer index.
      kernel_cache_ordering.emplace(
          kernel_runtime_ptr, kernel_cache_ordering.size());
    }

    // We recompute the DynamicTransformConcretizationInfo during
    // deserialization using a metadata copy of kernel inputs.
    auto&& [device_id, conc_info] = device_concrete_key;
    fb_kernel_runtimes.push_back(CreateKernelRuntimeStateDirect(
        builder,
        device_id,
        conc_info_id_map_.at(device_concrete_key),
        (conc_info != nullptr),
        &fb_device_runtimes));
  }

  // 2. Serialize input id to kernel cache
  std::vector<size_t> kernel_cache_keys;
  std::vector<size_t> kernel_cache_values;
  kernel_cache_keys.reserve(id_to_kernel_runtime_.size());
  kernel_cache_values.reserve(id_to_kernel_runtime_.size());

  for (auto&& [cache_id, kernel_runtime_ptr] : id_to_kernel_runtime_) {
    kernel_cache_keys.push_back(cache_id);
    kernel_cache_values.push_back(kernel_cache_ordering.at(kernel_runtime_ptr));
  }

  return serde::CreateFusionExecutorCacheDirect(
      builder,
      fusion_id_,
      inputs_id_lookup_.serialize(builder),
      &fb_kernel_runtimes,
      &kernel_cache_keys,
      &kernel_cache_values);
}

void FusionExecutorCache::deserialize(
    const serde::FusionExecutorCache* buffer,
    int64_t fusion_id) {
  // See definitions in serde/fusion_cache.fbs for tables
  // FusionExecutorCache and KernelRuntimes

  NVF_ERROR(buffer != nullptr, "serde::FusionExecutorCache is nullptr.");
  NVF_ERROR(
      fusion_id == buffer->fusion_id(),
      "Expected serde fusion_id to match given fusion_id.");

  fusion_id_ = buffer->fusion_id();

  inputs_id_lookup_.deserialize(buffer->inputs_cache());

  // For the id_to_kernel_runtime_ cache, we need a flat collection of all
  // FusionKernelRuntime objects.
  std::vector<FusionKernelRuntime*> all_runtimes;

  // 1. Deserialize kernel_runtimes_ unordered_map
  for (auto fb_device_runtimes : *buffer->kernel_runtimes_map()) {
    const auto& initial_info = initialInfo();
    NVF_ERROR(
        initial_info.isDynamic() ==
        fb_device_runtimes->has_dynamic_transform_info());
    NVF_ERROR(fb_device_runtimes->runtimes()->size() > 0);

    DynamicTransformConcretizationInfo* conc_info = nullptr;
    if (initial_info.isDynamic()) {
      // Each FusionKernelRuntime stores a metadata copy of its initial
      // inputs. We deserialize the arguments of the first FusionKernelRuntime
      // to recompute the concretization info.
      KernelArgumentHolder args;
      args.deserialize(fb_device_runtimes->runtimes()->begin()->args());
      auto expr_eval = executor_utils::bindInputs(args, fusion_.get());
      cached_conc_info_.emplace_back(
          std::make_unique<DynamicTransformConcretizationInfo>(
              &initial_info, &expr_eval, &exact_map_));
      conc_info = cached_conc_info_.back().get();
    }

    auto config =
        std::make_pair((int8_t)fb_device_runtimes->device_id(), conc_info);
    auto& device_runtimes = kernel_runtimes_.try_emplace(config).first->second;
    auto result =
        conc_info_id_map_.try_emplace(config, conc_info_id_map_.size() + 1);
    if (result.second) {
      deterministic_conc_info_.emplace_back(config);
    }

    for (auto fb_fusion_kernel_runtime : *fb_device_runtimes->runtimes()) {
      auto conc_fusion = std::make_unique<Fusion>(*fusion_);
      FusionGuard fg(conc_fusion.get());

      // Concretize original unscheduled fusion_ for this kernel runtime
      if (initial_info.isDynamic()) {
        const auto& conc_initial_info =
            conc_fusion->getManaged<DynamicTransformInitialInfo>(
                "initial_info");
        NVF_ERROR(conc_info != nullptr);
        conc_info->setInitialInfo(&conc_initial_info);

        DynamicTransform::concretizeFusion(conc_fusion.get(), conc_info);
        // Initial info is used during concretization and is owned by
        // conc_fusion. After concretization, we stop managing it so that we
        // won't keep cloning it for every subsequent Fusion copy.
        conc_fusion->stopManaging("initial_info");
      }

      // 1. Deserialize arguments for this FusionKernelRuntime
      KernelArgumentHolder args;
      args.deserialize(fb_fusion_kernel_runtime->args());

      NVF_ERROR(
          (int8_t)fb_device_runtimes->device_id() == args.getDeviceIndex(),
          "Expected serde FusionKernelRuntime device_id ",
          ((int64_t)fb_device_runtimes->device_id()),
          " to match KernelArgumentHolder metadata device id ",
          ((int64_t)args.getDeviceIndex()),
          ".");

      // 2. Construct new FusionKernelRuntime
      device_runtimes.emplace_back(std::make_unique<FusionKernelRuntime>(
          std::move(conc_fusion),
          args,
          fb_fusion_kernel_runtime,
          std::nullopt,
          fusion_id_,
          fb_device_runtimes->concrete_id(),
          device_runtimes.size()));

      // 3. For FusionKernelRuntime, we have a separate deserialize function
      // to create the KernelExecutor objects.
      device_runtimes.back()->deserialize(
          fb_fusion_kernel_runtime, args.getDeviceIndex());

      all_runtimes.emplace_back(device_runtimes.back().get());
    }
  }

  // 2. Rebuild input id to kernel cache
  for (auto idx : arange(buffer->kernel_cache_keys()->size())) {
    size_t key = buffer->kernel_cache_keys()->Get(idx);
    size_t value_id = buffer->kernel_cache_values()->Get(idx);
    id_to_kernel_runtime_.emplace(key, all_runtimes.at(value_id));
  }
}

void FusionExecutorCache::evictCache(size_t cache_id) {
  auto it = id_to_kernel_runtime_.find(cache_id);
  NVF_ERROR(it != id_to_kernel_runtime_.end());
  it->second->evictCache(cache_id);
  id_to_kernel_runtime_.erase(it);
}

// getKernelRuntimeFor inspects the inputs to find a usable
// FusionKernelRuntime as quickly as possible. To do so we cache at multiple
// levels:
//   A. If we have seen these inputs before, we re-use the FusionKernelRuntime
//   we used last time. Here, we mean the same input tensor sizes, as well as
//   same input scalars if they are used to compute an intermediate or output
//   tensor size.
//   B. We check how we should concretize the dynamic fusion using these
//   inputs. If we have not concretized the fusion this way previously, then
//   we concretize it and create a new FusionKernelRuntime, which means
//   segmenting and compiling new kernels. Otherwise, we check whether we can
//   re-use any of the previously-segmented runtimes.
//      i. We look at all FusionKernelRuntimes that have been used with
//      this concretized fusion.
//      ii. For each of those runtimes, we compare the heuristic parameters
//      for each segment to those that we compute using the current inputs.
//   If we do not find any runtimes whose heuristic parameters match, then we
//   create a new FusionKernelRuntime, which means segmenting and compiling
//   all new kernels.
//
// In summary, we have the following paths, in order of hottest to coldest:
//   1. Input ID cache hit: re-use runtime used last time these inputs were
//   seen
//   2. Concretization match, runtime heuristic params match: re-use runtime
//   after checking concretization/heuristics.
//   3. Concretization match but no runtime heuristic params match. Segment
//   to create new FusionKernelRuntime
//   4. Concretization is unseen: Segment to create a new FusionKernelRuntime
// For re-used shapes, path 1 is most relevant. For dynamic shape problems
// with a large number of unique shapes, path 2 is important. Paths 3 and 4
// are slow since they both involve re-segmentation and re-compilation of the
// Fusion.
FusionKernelRuntime* FusionExecutorCache::getKernelRuntimeFor(
    const KernelArgumentHolder& args,
    std::optional<PrimDataType> forced_index_type) {
  // Check for id hit case (Path 1)
  FUSER_PERF_SCOPE("FusionExecutorCache::getKernelRuntimeFor");
  auto unique_id_opt = args.getCacheId();
  NVF_CHECK(
      unique_id_opt.has_value(),
      "KernelArgumentHolder has no cache ID in getKernelRuntimeFor");
  auto unique_id = *unique_id_opt;
  auto id_it = id_to_kernel_runtime_.find(unique_id);
  if (id_it != id_to_kernel_runtime_.end()) {
    // If the forced index type is given, don't use the cached runtime
    // if its index type does not match with the forced type
    if (!forced_index_type.has_value() ||
        forced_index_type.value() == id_it->second->getIndexType()) {
      return id_it->second;
    }
  }

  // Compute or get cached initial concretization info
  const auto& initial_info = initialInfo();

  // Compute concretization info to use as cache key
  DynamicTransformConcretizationInfo* conc_info = nullptr;
  if (initial_info.isDynamic()) {
    // This class needs to own conc_info so it can be compared in subsequent
    // invocations.
    auto expr_eval = executor_utils::bindInputs(args, fusion_.get());
    cached_conc_info_.emplace_back(
        std::make_unique<DynamicTransformConcretizationInfo>(
            &initial_info, &expr_eval, &exact_map_));
    conc_info = cached_conc_info_.back().get();
  }

  // Initialize or fetch vector of FusionKernelRuntime objects associated with
  // each pair of device ID and concretization info.
  auto device_concrete_key = std::make_pair(args.getDeviceIndex(), conc_info);
  auto& kernel_runtimes =
      kernel_runtimes_.try_emplace(device_concrete_key).first->second;
  auto result = conc_info_id_map_.try_emplace(
      device_concrete_key, conc_info_id_map_.size() + 1);
  if (result.second) {
    deterministic_conc_info_.emplace_back(device_concrete_key);
  }

  // Check for re-use hit case
  //  a kernel runtime is re-usable if all the compiled
  //  kernels have the same heuristic parameters
  std::unique_ptr<HeuristicParamsList> new_heuristics;

  FusionKernelRuntime* kernel_runtime = nullptr;

  // Check if we missed the KernelRuntime cache (Path 2) and need to generate
  // a new kernel runtime (Path 3/4). By default, we try to avoid recompiling
  // whenever possible. However, this can lead to suboptimal code if we only
  // check that a compiled kernel is able to run with some inputs, instead of
  // whether it is optimal to do so. The NVFUSER_DISABLE=kernel_reuse option
  // is a coarse tool that just enforces that whenever we encounter a new set
  // of input shapes we segment and compile a new FusionKernelRuntime.
  // Effectively, this option disables Paths 2 and 3 above so that we only
  // have Path 1 (hottest re-use path) and Path 4 (full recompile).
  if (!isOptionDisabled(DisableOption::KernelReuse)) {
    FUSER_PERF_SCOPE("FusionExecutorCache::getKernelRuntimeFor::reuseKRT");
    auto runtime_it = std::find_if(
        kernel_runtimes.begin(),
        kernel_runtimes.end(),
        [&args, &new_heuristics, &forced_index_type](auto& kernel_runtime) {
          auto maybe_heuristics =
              kernel_runtime->getMaybeHeuristicsFor(args, forced_index_type);
          if (!maybe_heuristics.has_value()) {
            return false;
          }
          new_heuristics = std::move(maybe_heuristics.value());
          return true;
        });
    if (runtime_it != kernel_runtimes.end()) {
      kernel_runtime = runtime_it->get();
      kernel_runtime->updateHeuristicsLaunchParams(new_heuristics.get());
      id_to_kernel_runtime_[unique_id] = kernel_runtime;
      return kernel_runtime;
    }
  }

  {
    FUSER_PERF_SCOPE("FusionExecutorCache::getKernelRuntimeFor::compileNewKRT");
    // Paths 3 or 4
    // cache miss, need to re-build an optimized graph for this case

    // Clone fusion_ so that we can safely concretize it
    auto conc_fusion = std::make_unique<Fusion>(*fusion_);
    if (initial_info.isDynamic()) {
      const auto& conc_initial_info =
          conc_fusion->getManaged<DynamicTransformInitialInfo>("initial_info");
      NVF_ERROR(conc_info);
      conc_info->setInitialInfo(&conc_initial_info);

      if (isDebugDumpEnabled(DebugDumpOption::FusionIrConcretized)) {
        debug() << "Fusion before concretization:" << std::endl;
        conc_fusion->printMath();
        debug() << conc_initial_info.toString() << std::endl;
        debug() << conc_info->toString() << std::endl;
      }

      DynamicTransform::concretizeFusion(conc_fusion.get(), conc_info);
      // Initial info is used during concretization and is owned by
      // conc_fusion. After concretization, we stop managing it so that we
      // won't keep cloning it for every subsequent Fusion copy.
      conc_fusion->stopManaging("initial_info");

      if (isDebugDumpEnabled(DebugDumpOption::FusionIrConcretized)) {
        debug() << "Concretized Fusion:" << std::endl;
        conc_fusion->print();
      }
    }
    FusionGuard fg(conc_fusion.get());
    kernel_runtimes.emplace_back(std::make_unique<FusionKernelRuntime>(
        std::move(conc_fusion),
        args,
        /*serde_buffer=*/nullptr,
        forced_index_type,
        fusion_id_,
        conc_info_id_map_.at(device_concrete_key),
        kernel_runtimes.size(),
        auto_schedule_));
    kernel_runtime = kernel_runtimes.back().get();

    if (profiling_) {
      kernel_runtime->profile(true);
    }
    id_to_kernel_runtime_[unique_id] = kernel_runtime;
    return kernel_runtime;
  }
}

DynamicTransformInitialInfo& FusionExecutorCache::initialInfo() {
  if (!initial_info_.has_value()) {
    initial_info_ = DynamicTransform::getInitialInfo(fusion());
    fusion()->manage(
        "initial_info",
        initial_info_.value(),
        [](IrCloner& ir_cloner, std::any data) -> std::any {
          return std::any_cast<DynamicTransformInitialInfo>(data).clone(
              ir_cloner);
        });
  }
  return initial_info_.value();
}

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <runtime/fusion_kernel_runtime.h>

#include <fusion.h>
#include <fusion_profiler.h>
#include <fusion_segmenter.h>
#include <instrumentation.h>
#include <ir/base_nodes.h>
#include <preseg_passes/pre_segmenter.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/translation.h>
#include <runtime/executor.h>
#include <runtime/executor_dispatch.h>
#include <runtime/fusion_cache_utils.h>
#include <scheduler/heuristic.h>
#include <serde/fusion_cache_generated.h>
#include <type.h>

#include <c10/cuda/CUDAGuard.h>

namespace nvfuser {

namespace {
// Replace CUDA tensor with Meta tensor because storing tensors can cause
// out-of-memory issues. Other arguments are returned as-is.
PolymorphicValue convertMetadataArg(PolymorphicValue arg) {
  if (arg.is<at::Tensor>()) {
    if (const auto& tensor = arg.as<at::Tensor>(); tensor.is_cuda()) {
      auto meta_tensor = at::Tensor(at::detail::empty_strided_meta(
          tensor.sizes(),
          tensor.strides(),
          tensor.scalar_type(),
          c10::nullopt,
          c10::Device(c10::DeviceType::Meta, 0),
          c10::nullopt));
      return std::move(meta_tensor);
    }
  }
  return arg;
}

KernelArgumentHolder copyMetadataArg(const KernelArgumentHolder& src) {
  KernelArgumentHolder dst;
  std::transform(
      src.cbegin(), src.cend(), dst.getBackInserter(), convertMetadataArg);
  dst.setDeviceIndex(src.getDeviceIndex());
  return dst;
}
} // namespace

FusionKernelRuntime::FusionKernelRuntime(
    std::unique_ptr<Fusion> fusion,
    const KernelArgumentHolder& args,
    const serde::FusionKernelRuntime* serde_buffer,
    std::optional<PrimDataType> forced_index_type,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    bool auto_schedule)
    : args_metadata_{copyMetadataArg(args)},
      fusion_id_{fusion_id},
      concrete_id_{concrete_id},
      runtime_id_{runtime_id},
      auto_schedule_{auto_schedule} {
  FUSER_PERF_SCOPE("FusionKernelRuntime::FusionKernelRuntime");

  NVF_ERROR(
      !fusion->hasDynamicTransform(),
      "Fusion must be concretized before constructing FusionKernelRuntime");

  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());

  if (isDebugDumpEnabled(DebugDumpOption::FusionIrPreseg)) {
    const auto& communicator = Communicator::getInstance();
    // Only the first local rank will print. Pre-segmenter fusion IR is device
    // agnostic, so letting all ranks print isn't any more useful.
    if (!communicator.is_available() || communicator.local_rank() == 0) {
      debug() << "Fusion IR after pre-segmenter optimization passes:"
              << std::endl;
      fusion->print();
    }
  }

  // SchedulerRuntimeInfo modifies the fusion, so it is required for both
  // compile paths.
  std::vector<TensorView*> all_tvs = fusion->allTvs();
  SchedulerRuntimeInfo runtime_info(
      fusion.get(), args, nullptr, all_tvs, forced_index_type);

  if (serde_buffer == nullptr || !serde_buffer->segmented_fusion()->valid()) {
    // Default compilation path applies segmentation before scheduling and
    // compiling the fusion.
    segmented_fusion_ =
        SegmentCandidateFinder::segment(std::move(fusion), args, runtime_info);
  } else {
    // Serialization path that generates segmented fusion from flatbuffers.
    // Convert Welford to two-pass if option is enabled and the original
    // heuristic is persistent
    const flatbuffers::Vector<flatbuffers::Offset<serde::SegmentedGroup>>*
        segmented_groups = serde_buffer->segmented_fusion()->groups();
    bool has_persistent_heuristic = std::any_of(
        segmented_groups->begin(),
        segmented_groups->end(),
        [](const serde::SegmentedGroup* sg) {
          auto heuristic = static_cast<SchedulerType>(sg->heuristic());
          return heuristic == SchedulerType::InnerPersistent ||
              heuristic == SchedulerType::OuterPersistent ||
              heuristic == SchedulerType::InnerOuterPersistent;
        });

    bool has_welford_ops = ir_utils::hasOpsOfType<WelfordOp>(fusion.get());
    if (has_welford_ops && has_persistent_heuristic) {
      SegmentCandidateFinder::translateWelfordInFusion(fusion.get(), args);
    }
    segmented_fusion_ = std::make_unique<SegmentedFusion>(std::move(fusion));
    segmented_fusion_->deserialize(serde_buffer->segmented_fusion());
  }

  // Pre-compute the executor order so that the run time path
  //  would go directly to kernel launch.
  prepareRuntimeOrder(segmented_fusion_.get(), runtime_workspace_);

  executors_.resize(segmented_fusion_->groups().size());

  if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
    segmented_fusion_->print();
  }

  // Even if we go through the segmented path we may still end up
  //  with a segmented fusion with one group. This case still
  //  counts as un-segmented.
  is_segmented_ = segmented_fusion_->groups().size() > 1;

  // Create Initial Heuristics for Segmented Fusion
  auto maybe_heuristics = getMaybeHeuristicsFor(args, forced_index_type);
  NVF_CHECK(maybe_heuristics.has_value());
  heuristics_ = std::move(maybe_heuristics.value());
}

void FusionKernelRuntime::evictCache(size_t input_id) {
  for (auto& ea : executors_) {
    if (auto ke = dynamic_cast<KernelExecutor*>(ea.get())) {
      ke->evictCache(input_id);
    }
  }
}

bool FusionKernelRuntime::isCompiled() const {
  if (isOptionEnabled(EnableOption::HostIrLowering)) {
    return hie_ != nullptr;
  } else {
    std::lock_guard<std::mutex> guard(mutex_);
    return std::all_of(
        executors_.begin(), executors_.end(), [](const auto& executor) {
          return ExecutorDispatch::isCompiled(executor.get());
        });
  }
}

flatbuffers::Offset<serde::FusionKernelRuntime> FusionKernelRuntime::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See table definition for FusionKernelRuntime in serde/fusion_cache.fbs

  NVF_CHECK(
      isCompiled(),
      "Tried to serialize entries of executors before they were initialized.");

  // 1. Serialize KernelExecutor objects
  std::vector<flatbuffers::Offset<serde::KernelExecutor>> executors_fb;
  executors_fb.reserve(executors_.size());
  for (auto& ea : executors_) {
    if (auto ke = dynamic_cast<KernelExecutor*>(ea.get())) {
      executors_fb.emplace_back(ke->serialize(builder));
    }
  }

  flatbuffers::Offset<serde::SegmentedFusion> segmented_fusion_fb = 0;
  if (segmented_fusion_) {
    segmented_fusion_fb = segmented_fusion_->serialize(builder);
  }

  return serde::CreateFusionKernelRuntimeDirect(
      builder,
      fusion_id_,
      concrete_id_,
      runtime_id_,
      args_metadata_.serialize(builder),
      &executors_fb,
      segmented_fusion_fb);
}

void FusionKernelRuntime::deserialize(
    const serde::FusionKernelRuntime* buffer,
    int8_t device_index) {
  // See table definition in FusionKernelRuntime in serde/fusion_cache.fbs

  NVF_ERROR(buffer != nullptr, "serde::FusionKernelRuntime is nullptr.");
  NVF_ERROR(runtime_workspace_.group_run_order.size() == executors_.size());
  NVF_ERROR(
      fusion_id_ == buffer->fusion_id(),
      "Expected FusionKernelRuntime fusion_id to match serde fusion_id.");
  NVF_ERROR(
      concrete_id_ == buffer->concrete_id(),
      "Expected FusionKernelRuntime concrete_id to match serde concrete_id.");
  NVF_ERROR(
      runtime_id_ == buffer->runtime_id(),
      "Expected FusionKernelRuntime runtime_id to match serde runtime_id.");

  // find the flatbuffer with the same group_id for SegmentedGroup
  auto get_buffer = [&](int64_t group_id) {
    for (auto buffer : *buffer->executors()) {
      if (buffer->group_id() == group_id) {
        return buffer;
      }
    }
    NVF_THROW(
        "Could not find the serialized group associated with id: ", group_id);
  };

  // 1. Deserialize KernelExecutor objects
  for (auto idx : c10::irange(executors_.size())) {
    auto sg = runtime_workspace_.group_run_order.at(idx);

    // Create and schedule Fusion for this SegmentedGroup
    auto group_id = sg->groupId();
    auto heuristic_params = schedulers().at(group_id).get();
    NVF_ERROR(
        !sg || heuristic_params->scheduler_type == sg->schedulerType(),
        "Heuristics do not match.");
    auto fusion_to_run = segmented_fusion_->makeFusion(sg).second;
    FusionGuard fg(fusion_to_run.get());
    SchedulerEntry::makeSchedulerInstance(heuristic_params->scheduler_type)
        ->schedule(fusion_to_run.get(), heuristic_params);

    // Initialize associated executors
    executors_[group_id] = ExecutorDispatch::makeExecutor(
        fusion_to_run.get(), fusion_id_, concrete_id_, runtime_id_, group_id);

    // Deserialize KernelExecutor; Otherwise use ExecutorDispatch
    if (auto ke =
            dynamic_cast<KernelExecutor*>(executors_.at(group_id).get())) {
      ke->deserialize(
          get_buffer(group_id),
          fusion_to_run.get(),
          device_index,
          heuristic_params->cparams,
          heuristic_params->scheduler_type,
          fusion_id_,
          concrete_id_,
          runtime_id_,
          group_id);
    } else {
      ExecutorDispatch::compile(
          executors_.at(group_id).get(),
          fusion_to_run.get(),
          args_metadata_,
          heuristic_params->lparams,
          heuristic_params->cparams,
          heuristic_params->scheduler_type);
    }
  }
}

PrimDataType FusionKernelRuntime::getIndexType() const {
  // No scheduler means nothing to run. It may still be unsafe to
  // save tensor sizes and strides in Int32
  if (schedulers().empty()) {
    return PrimDataType::Int;
  }
  auto index_type = schedulers().at(0).get()->cparams.index_type;
  NVF_ERROR(index_type.has_value());
  return index_type.value();
}

KernelArgumentHolder FusionKernelRuntime::runWithInputs(
    KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::runWithInputs");

  if (isOptionEnabled(EnableOption::HostIrLowering)) {
    if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
      debug() << "=================RUNNING HOSTIR EVALUATOR================="
              << std::endl;
    }

    std::unordered_map<Val*, PolymorphicValue> tensor_map;
    for (const auto i : c10::irange(args.size())) {
      tensor_map.emplace(hie_->inputs()[i], args[i]);
    }
    auto outputs = hie_->runWithInput(tensor_map);
    if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
      debug() << "============= FINISHED RUNNING HOSTIR EVALUATOR ============"
              << std::endl;
    }
    return outputs;
  }

  if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    debug() << "=================RUNNING FUSION SEGMENTS================="
            << std::endl;
  }

  c10::Device device(c10::DeviceType::CUDA, (int8_t)args.getDeviceIndex());
  const auto& tensor_map = runSegmentsWithInputs(args);

  if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    debug() << "============= FINISHED RUNNING FUSION SEGMENTS ============"
            << std::endl;
  }

  // Produce final global output
  KernelArgumentHolder fusion_outputs;
  for (Val* output : segmented_fusion_->outputs()) {
    NVF_ERROR(
        tensor_map.count(output),
        "Segmented fusion output ",
        output->toString(),
        " does not exist in `tensor_map`.");
    fusion_outputs.push(tensor_map.at(output));
  }
  return fusion_outputs;
}

// passing args by value because we will be modify this
void FusionKernelRuntime::compileFusionParallel(KernelArgumentHolder args) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::compileFusionParallel");

  std::lock_guard<std::mutex> guard(mutex_);

  NVF_ERROR(
      args.size() == segmented_fusion_->inputs().size(),
      "Inputs were not set up correctly, received ",
      args.size(),
      " inputs but expecting ",
      segmented_fusion_->inputs().size());

  ArgumentManager args_manager(
      args, runtime_workspace_, segmented_fusion_->inputs());

  // group should share cache id.
  auto group_cache_id = args.getCacheId();

  const int64_t num_groups = (int64_t)runtime_workspace_.group_run_order.size();
  num_live_args_after_segment_runs_.reserve(num_groups);
  if (isProfilerEnabled()) {
    FusionProfiler::startCompile();
  }

  // host ir
  std::unique_ptr<hir::HostIrContainer> hic;
  if (isOptionEnabled(EnableOption::HostIrLowering)) {
    hic = std::make_unique<hir::HostIrContainer>(
        num_groups); // Some indices will be empty
  }

  std::atomic<bool> detect_exception_in_thread_pool{false};
  std::string thread_pool_error_message;
  std::mutex thread_pool_error_message_mutex;
  for (int64_t run_order_id = 0; run_order_id < num_groups; ++run_order_id) {
    auto group_to_run = runtime_workspace_.group_run_order.at(run_order_id);

    if (isDebugDumpEnabled(DebugDumpOption::PythonDefinitionSegments)) {
      debug() << "Python definition for segmented group "
              << group_to_run->groupId() << ":" << std::endl;
      python_frontend::FusionDefinition fd(/*id=*/std::nullopt);
      python_frontend::translate(group_to_run->getFusion(), &fd);
      fd.print(debug());
    }

    // TODO: index mode should be updated per segmented kernel
    // Prepare input vector
    auto group_runtime_inputs =
        args_manager.translateValsToArgs(group_to_run->inputs());
    group_runtime_inputs.setDeviceIndex(args.getDeviceIndex());
    if (group_cache_id.has_value()) {
      group_runtime_inputs.setCacheId(group_cache_id.value());
    }

    if (num_groups == 1 || isOptionDisabled(DisableOption::ParallelCompile)) {
      FUSER_PERF_SCOPE("FusionKernelRuntime::compileFusionParallel");
      c10::cuda::CUDAGuard dg(args.getDeviceIndex());
      c10::Device device(c10::DeviceType::CUDA, args.getDeviceIndex());
      compileKernel(group_runtime_inputs, group_to_run, hic.get());
    } else {
      hir::HostIrContainer* hic_ptr = hic.get();
      // launch compileKernel thread here
      getThreadPool()->run([this,
                            args,
                            group_runtime_inputs,
                            group_to_run,
                            &detect_exception_in_thread_pool,
                            &thread_pool_error_message,
                            &thread_pool_error_message_mutex,
                            hic_ptr]() {
        FUSER_PERF_SCOPE("FusionKernelRuntime::compileFusionParallel");
        try {
          c10::cuda::CUDAGuard dg(args.getDeviceIndex());
          c10::Device device(c10::DeviceType::CUDA, args.getDeviceIndex());
          compileKernel(group_runtime_inputs, group_to_run, hic_ptr);
        } catch (const std::exception& e) {
          // Set flag inside lambda so we can throw an exception after thread
          // pool completes its work.
          detect_exception_in_thread_pool.store(true);
          const std::lock_guard<std::mutex> lock(
              thread_pool_error_message_mutex);
          std::stringstream ss;
          ss << thread_pool_error_message << "\nError from segmentation group "
             << group_to_run->groupId() << ": " << e.what() << "\n";
          thread_pool_error_message = ss.str();
        }
      });
    }

    auto fusion_to_run = segmented_fusion_->makeFusion(group_to_run).second;
    auto group_runtime_outputs =
        inferOutputSizes(fusion_to_run.get(), group_runtime_inputs);

    // map output args to tensor map
    args_manager.updateWithSegmentOutputs(
        group_to_run->outputs(), group_runtime_outputs, run_order_id);
    num_live_args_after_segment_runs_.emplace_back((int64_t)args.size());
  }

  // add all expressions and compiled kernels to the host ir container
  if (hic != nullptr) {
    IrCloner ir_cloner(hic.get());
    FusionGuard::setCurFusion(hic.get());
    for (int64_t run_order_id = 0; run_order_id < num_groups; ++run_order_id) {
      auto group_to_run = runtime_workspace_.group_run_order.at(run_order_id);
      if (hic->hasKernelExecutor(run_order_id)) {
        auto in_clone = ir_cloner.clone(group_to_run->inputs());
        auto out_clone = ir_cloner.clone(group_to_run->outputs());
        auto heuristic_params = schedulers().at(run_order_id).get();
        auto launch_kernel = IrBuilder::create<hir::LaunchKernel>(
            run_order_id,
            heuristic_params->lparams,
            heuristic_params->cparams,
            std::vector<Val*>{in_clone},
            std::vector<Val*>{out_clone});
        hic->pushBackTopLevelExprs(launch_kernel);
      } else {
        // push back segment's exprs into the container as top level expressions
        for (auto* expr : group_to_run->exprs()) {
          auto cloned_expr = ir_cloner.clone(expr);
          hic->pushBackTopLevelExprs(cloned_expr);
        }
      }
    }
    for (const Val* in : segmented_fusion_->inputs()) {
      hic->addInput(ir_cloner.clone(in));
    }
    for (const Val* out : segmented_fusion_->outputs()) {
      hic->addOutput(ir_cloner.clone(out));
    }
  }

  if (num_groups != 1 && !isOptionDisabled(DisableOption::ParallelCompile)) {
    // Wait until all segments finish compiling
    getThreadPool()->waitWorkComplete();
    NVF_ERROR(
        !detect_exception_in_thread_pool.load(),
        "Detected exception while compiling fusion segments in parallel. ",
        "Error messages from all threads are printed below.\n",
        thread_pool_error_message,
        "\nUse NVFUSER_DISABLE=parallel_compile to simplify error message.");
  }

  if (hic != nullptr) {
    hie_ = std::make_unique<hir::HostIrEvaluator>(
        hir::HostIrEvaluator(std::move(hic)));
  }

  if (isProfilerEnabled()) {
    FusionProfiler::stopCompile();
  }
}

void FusionKernelRuntime::disableKernelLaunch() {
  NVF_CHECK(
      isCompiled(),
      "Tried to set parameters of executors before they were initialized.");
  for (auto& executor : executors_) {
    if (auto ke = dynamic_cast<KernelExecutor*>(executor.get())) {
      ke->setExecuteKernelFlag(false);
    }
  }
}

SegmentedFusion* FusionKernelRuntime::fusionSegments() const {
  return segmented_fusion_.get();
}

HeuristicParamsList* FusionKernelRuntime::schedulerHeuristics() const {
  return heuristics_.get();
}

const ExecutorLog& FusionKernelRuntime::getMostRecentExecutorLog() const {
  NVF_ERROR(profiling_, "Executor log is only produced in profiling mode");
  return most_recent_executor_log_;
}

std::optional<std::unique_ptr<HeuristicParamsList>> FusionKernelRuntime::
    getMaybeHeuristicsFor(
        const KernelArgumentHolder& args,
        std::optional<PrimDataType> forced_index_type) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::getMaybeHeuristicsFor");

  // The runtime group run order is different from the segmented_fusion group
  // order. Instead of using HeuristicParamsList::emplaceBack, we initialize
  // HeuristicParamsList with the desired number of groups.
  const int64_t num_groups = (int64_t)runtime_workspace_.group_run_order.size();
  std::unique_ptr<HeuristicParamsList> heuristics =
      std::make_unique<HeuristicParamsList>(num_groups);

  // We make a mutable copy of args so that we can use it in an
  // ArgumentManager
  ArgumentManager args_manager(
      args, runtime_workspace_, segmented_fusion_->inputs());
  // Follow group run order
  for (int64_t group_id : c10::irange(num_groups)) {
    auto group_to_run = runtime_workspace_.group_run_order.at(group_id);

    // Create fusion for this segmented group
    Fusion* fusion_to_run = group_to_run->getFusion();
    NVF_ERROR(fusion_to_run != nullptr);
    FusionGuard fg(fusion_to_run);

    // Get input arguments for SchedulerRuntimeInfo
    KernelArgumentHolder group_runtime_inputs =
        args_manager.translateValsToArgs(group_to_run->inputs());
    group_runtime_inputs.setDeviceIndex(args.getDeviceIndex());

    // Create PrecomputedValues for fusion segment
    std::unique_ptr<PrecomputedValues> evaluator_precomputed_values;
    {
      FUSER_PERF_SCOPE(
          "FusionKernelRuntime::getMaybeHeuristicsFor::PrecomputedValues");
      evaluator_precomputed_values =
          std::make_unique<PrecomputedValues>(fusion_to_run);
      evaluator_precomputed_values->bindInputs(group_runtime_inputs);
      // TODO Remove binding the original fusion inputs when creating
      // heuristics for fusion segment.
      evaluator_precomputed_values->bindValues(
          group_to_run->getCompleteFusionInputs(), args);
      evaluator_precomputed_values->evaluate();
    }

    // Get all tensorviews for segmented fusion
    std::vector<TensorView*> all_tvs_for_fusion_to_run =
        fusion_to_run->allTvs();

    SchedulerRuntimeInfo fusion_to_run_info(
        fusion_to_run,
        group_runtime_inputs,
        evaluator_precomputed_values.get(),
        all_tvs_for_fusion_to_run,
        forced_index_type);

    if (heuristics_ == nullptr) {
      // Add new scheduler entry for this segmented group
      heuristics->at(group_to_run->groupId()) =
          segmented_fusion_->makeInitialHeuristicParams(
              group_to_run, fusion_to_run_info);
    } else {
      // Try to get scheduler entry
      // NOTE: we are able to skip compile time checks here since the fusion
      // has already been segmented. During segmentation, each segment must
      // pass both canScheduleCompileTime and canScheduleRuntime for the
      // scheduler that accepts the segment. Since we might have different
      // runtime info than was used during segmentation, we cannot skip
      // canScheduleRuntime, but it is safe to skip canScheduleCompileTime. We
      // skip it here to avoid performing expensive fusion traversals on the
      // dynamic shape path.
      auto maybe_heuristic_params =
          group_to_run->getMaybeHeuristicParams(fusion_to_run_info);
      // If unavailable, then return std::nullopt
      if (!maybe_heuristic_params.has_value()) {
        return std::nullopt;
      }
      // Check if this scheduler entry matches the previous entry for this
      // segmented group. If no match, then return std::nullptr
      auto heuristic_params = std::move(maybe_heuristic_params.value());
      if (!heuristic_params->sameAs(
              heuristics_->at(group_to_run->groupId()).get())) {
        return std::nullopt;
      }
      // Add new scheduler entry for this segmented group
      heuristics->at(group_to_run->groupId()) = std::move(heuristic_params);
    }

    // Generate metadata for the fusion's outputs
    auto group_runtime_outputs = inferOutputSizes(
        fusion_to_run,
        group_runtime_inputs,
        evaluator_precomputed_values.get());

    args_manager.updateWithSegmentOutputs(
        group_to_run->outputs(), group_runtime_outputs, group_id);
  }
  return heuristics;
}

void FusionKernelRuntime::updateHeuristicsLaunchParams(
    HeuristicParamsList* update_heuristics) {
  auto scheduler_list_length = heuristics_->heuristicsList().size();
  NVF_ERROR(
      update_heuristics->heuristicsList().size() == scheduler_list_length);
  for (const auto i : c10::irange(scheduler_list_length)) {
    auto& heuristic_params = heuristics_->heuristicsList()[i];
    heuristic_params->lparams = update_heuristics->heuristicsList()[i]->lparams;
  }
}

const std::vector<std::unique_ptr<ExecutorAbstract>>& FusionKernelRuntime::
    executors() const {
  return executors_;
}

std::unordered_map<Val*, PolymorphicValue> FusionKernelRuntime::
    runSegmentsWithInputs(KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::runSegmentsWithInputs");
  NVF_ERROR(
      args.size() == segmented_fusion_->inputs().size(),
      "Inputs were not set up correctly, received ",
      args.size(),
      " inputs but expected ",
      segmented_fusion_->inputs().size());

  ArgumentManager args_manager(
      args, runtime_workspace_, segmented_fusion_->inputs());

  // group should share cache id.
  auto group_cache_id = args.getCacheId();
  const int64_t num_groups = (int64_t)runtime_workspace_.group_run_order.size();
  num_live_args_after_segment_runs_.reserve(num_groups);
  kernel_time_ms_ = 0;
  for (auto run_order_id : c10::irange(num_groups)) {
    // TODO: index mode should be updated per segmented kernel
    // Prepare input vector
    auto group_to_run = runtime_workspace_.group_run_order.at(run_order_id);
    KernelArgumentHolder group_runtime_inputs =
        args_manager.translateValsToArgs(group_to_run->inputs());
    group_runtime_inputs.setDeviceIndex(args.getDeviceIndex());
    if (group_cache_id.has_value()) {
      group_runtime_inputs.setCacheId(group_cache_id.value());
    }

    // TODO: currently we are still outputing PyTorch tensors, instead of
    // something abstract. This is quite unsatisfying.

    // Run graph segment
    KernelArgumentHolder group_runtime_outputs =
        runKernelWithInput(group_runtime_inputs, group_to_run);

    args_manager.updateWithSegmentOutputs(
        group_to_run->outputs(), group_runtime_outputs, run_order_id);
    num_live_args_after_segment_runs_.push_back((int64_t)args.size());
  }

  if (isProfilerEnabled()) {
    int64_t input_bytes = 0;
    for (auto* inp : fusionSegments()->inputs()) {
      if (inp->isA<TensorView>()) {
        const auto& tensor = args_manager.checkTensorMap(inp).as<at::Tensor>();
        input_bytes += static_cast<int64_t>(tensor.storage().nbytes());
      }
    }
    FusionProfiler::inputBytesAccessed(input_bytes);

    int64_t output_bytes = 0;
    for (auto* outp : fusionSegments()->outputs()) {
      if (outp->isA<TensorView>()) {
        const auto& tensor = args_manager.checkTensorMap(outp).as<at::Tensor>();
        output_bytes += static_cast<int64_t>(tensor.storage().nbytes());
      }
    }
    FusionProfiler::outputBytesAccessed(output_bytes);
  }

  return args_manager.getTensorMap();
}

KernelArgumentHolder FusionKernelRuntime::runKernelWithInput(
    KernelArgumentHolder& args,
    SegmentedGroup* sg) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::runKernelWithInput");
  std::lock_guard<std::mutex> guard(mutex_);
  // This function will be called once on un-segmented fusion,
  // for segmented fusion, this function will be called on each segment
  // In the case of segmented fusion, segmented group needs to be given so
  // a kernel is compiled and run for a segmented group
  // In the case of complete fusion, sg = nullptr, and the original fusion
  // is complied and run.
  NVF_ERROR(sg, "runKernelWithInput: need valid group to run");
  auto [launch_params, compile_params] = getKernelConfig(args, sg);
  auto group_id = sg->groupId();
  auto heuristic_params = schedulers().at(group_id).get();
  ExecutorAbstract* ea = executors_.at(group_id).get();

  if (profiling_) {
    most_recent_executor_log_.fusion_executor = ea;
    most_recent_executor_log_.params = heuristic_params->clone();
  }

  // TODO: This is a work around for the fallback execution path where a
  // kernel is not compiled. Perhaps the group/segment Id needs to be
  // specified to the executor at its constructor.  Currently, initialization
  // is ad hoc.
  if (auto ke = dynamic_cast<KernelExecutor*>(ea)) {
    ke->setGroupId(group_id);
  }
  auto outputs =
      ExecutorDispatch::run(ea, args, {}, launch_params, compile_params);

  return outputs;
}

void FusionKernelRuntime::compileKernel(
    const KernelArgumentHolder& args,
    SegmentedGroup* sg,
    hir::HostIrContainer* hic) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::compileKernel");
  auto group_id = sg->groupId();
  auto heuristic_params = schedulers().at(group_id).get();

  // Check that the heuristics are matched, in the case of segmented fusion
  NVF_ERROR(heuristic_params->scheduler_type == sg->schedulerType());

  if (hic != nullptr &&
      (sg->schedulerType() == SchedulerType::ExprEval ||
       sg->schedulerType() == SchedulerType::Communication)) {
    // When lowering to host IR, ExprEval and Communication segments are lowered
    // to top-level expressions in the host IR container, not executors. Only
    // kernels need to be compiled.
    return;
  }

  // Running a segment group as a single kernel,
  // make a fusion to run from segmented fusion
  auto fusion_to_run = segmented_fusion_->makeFusion(sg).second;
  if (isDebugDumpEnabled(DebugDumpOption::FusionIrPresched)) {
    fusion_to_run->printMath();
  }
  FusionGuard fg(fusion_to_run.get());
  if (auto_schedule_) {
    SchedulerEntry::makeSchedulerInstance(heuristic_params->scheduler_type)
        ->schedule(fusion_to_run.get(), heuristic_params);
  }
  NVF_ERROR(
      heuristic_params->cparams.index_type.has_value(),
      "Kernel index type is not defined.");

  if (hic != nullptr) {
    auto ke = std::make_unique<KernelExecutor>();
    ke->compile(
        fusion_to_run.get(),
        args,
        heuristic_params->lparams,
        heuristic_params->cparams,
        heuristic_params->scheduler_type);
    hic->setKernelExecutor(group_id, std::move(ke));
  } else {
    // Initialize associated executors
    executors_[group_id] = ExecutorDispatch::makeExecutor(
        fusion_to_run.get(), fusion_id_, concrete_id_, runtime_id_, group_id);

    ExecutorDispatch::compile(
        executors_.at(group_id).get(),
        fusion_to_run.get(),
        args,
        heuristic_params->lparams,
        heuristic_params->cparams,
        heuristic_params->scheduler_type);
  }
}

std::pair<LaunchParams, CompileParams> FusionKernelRuntime::getKernelConfig(
    const KernelArgumentHolder& args,
    SegmentedGroup* sg) {
  auto group_id = sg->groupId();
  auto heuristic_params = schedulers().at(group_id).get();

  // Check that the heuristics are matched, in the case of segmented fusion
  NVF_ERROR(!sg || heuristic_params->scheduler_type == sg->schedulerType());

  return std::make_pair(heuristic_params->lparams, heuristic_params->cparams);
}

const std::vector<std::unique_ptr<HeuristicParams>>& FusionKernelRuntime::
    schedulers() const {
  return heuristics_->heuristicsList();
}

} // namespace nvfuser

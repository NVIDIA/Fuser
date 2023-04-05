// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <kernel_cache.h>

#include <executor_params.h>
#include <instrumentation.h>
#include <ir_utils.h>
#include <parser.h>
#include <scheduler/debug_utils.h>
#include <scheduler/registry.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>

namespace nvfuser {

namespace {

#define THREAD_POOL_SIZE 10

// TODO: clean this up with some knobs
c10::ThreadPool* getThreadPool() {
  static c10::ThreadPool pool(THREAD_POOL_SIZE);
  return &pool;
}

void encodeBuffer(size_t value, std::string& buffer) {
  const char* v = reinterpret_cast<char*>(&value);
  for (const auto i : c10::irange(sizeof(size_t))) {
    (void)i; // Suppress unused variable warning
    buffer.push_back(*(v++));
  }
}

} // namespace

InputsIdLookup::IdLookupReturn InputsIdLookup::lookupId(
    const at::ArrayRef<c10::IValue>& inputs) {
  IdLookupReturn ret;

  // lock mutex_ because we are touching encoding_
  std::lock_guard<std::mutex> guard(mutex_);
  encoding_.clear();
  for (const auto& input : inputs) {
    if (input.isTensor()) {
      auto& input_tensor = input.toTensor();

      for (auto size : input_tensor.sizes()) {
        encodeBuffer(size, encoding_);
        encoding_.push_back(' ');
      }
      encoding_.push_back('X');
      encoding_.push_back(' ');
      for (auto stride : input_tensor.strides()) {
        encodeBuffer(stride, encoding_);
        encoding_.push_back(' ');
      }
      encoding_.push_back('a');
      encodeBuffer(
          SchedulerRuntimeInfo::computeAlignmentSize(
              (size_t)input_tensor.data_ptr()),
          encoding_);
      encoding_.push_back('d');
      encodeBuffer(input_tensor.device().index(), encoding_);
    } else {
      // encode s for scalar;
      encoding_.push_back('s');
    }
    encoding_.push_back(';');
  }

  auto& entry = encoding_lookup_[encoding_];

  if (entry.id == 0) {
    // no entry existed for given input set, set id for given entry
    entry.id = current_id_++;
    if (used_entry_.size() == max_cache_size_) {
      // pop least recently used cache;
      const auto& remove_iter = encoding_lookup_.find(used_entry_.back());
      used_entry_.pop_back();
      ret.evict_id = remove_iter->second.id;
      ret.eviction = true;
      encoding_lookup_.erase(remove_iter);
    }
  } else {
    // short-cut to leave LRU entry as is
    if (entry.lru_iter == used_entry_.begin()) {
      ret.id = entry.id;
      return ret;
    }

    used_entry_.erase(entry.lru_iter);
  }

  ret.id = entry.id;
  entry.lru_iter = used_entry_.insert(used_entry_.begin(), encoding_);
  return ret;
}

FusionExecutorCache::FusionExecutorCache(std::unique_ptr<Fusion> fusion)
    : fusion_(std::move(fusion)) {}

KernelArgumentHolder FusionExecutorCache::prepareInputs(
    const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionExecutorCache::prepareInputs");

  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder(inputs);

  // TODO: move InputsIdLookup inside KernelArgumentHolder;
  auto id_lookup_ret = inputs_id_lookup_.lookupId(inputs);
  if (id_lookup_ret.eviction) {
    evictCache(id_lookup_ret.evict_id);
  }

  args.setCacheId(id_lookup_ret.id);
  return args;
}

bool FusionExecutorCache::isCompiled(const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionExecutorCache::isCompiled");

  // Access kernels associated with the common device id
  KernelArgumentHolder args = prepareInputs(inputs);

  return getKernelRuntimeFor(args)->isCompiled();
}

void FusionExecutorCache::compileFusionAsync(
    const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionExecutorCache::compileFusionAsync");

  KernelArgumentHolder args = prepareInputs(inputs);
  auto kernel_runtime = getKernelRuntimeFor(args);

  kernel_runtime->startAsyncCompile(args);
}

// Note [ Permutation support in nvfuser ]
//
// Background:
// To support permutation in nvfuser with optimal performance, we would want to
// allow dimension collapsing in generated code on channels-last tensors, which
// greatly simplifies indexing. Current API in codegen only allows dimensional
// collapsing on neighboring axes. The unfortunate thing is that memory format
// design in PyTorch is implicitly marked by strides, while the semantics
// meaning of axes remain unchanged. i.e. A 4d tensor with axes [N, C, H, W]
// would have the same shape in both format, while contiguous tensor carries
// strides [C*H*W, H*W, W, 1] and channels-last tensor [H*W*C, 1, W*C, C]
//
// Approach:
// Part_1. To allow axes collapsing for permuted tensor in codegen, we can
// permute input tensor to have axes in decending order by their strides, so
// they would be viewed as `contiguous` in codegen, hence collapsed to simple
// indexing.
//
// Part_2. To ensure correct results, we need to ensure computation in
// NvFuser carries the same semantics as the TorchScript graph.
//
// Part_2_1. We need to maintain a bookkeeping where each codegen tensor is
// tagged with their permutation.
//
// Part_2_2. The parsing rule should handle and propagate the tag properly.
// e.g. Batch normalization has special rules for `channels_last` input tensor,
// so it should mark its output tensor with the right permutation.
//
// Part_3. The permuted, output tensor generated by codegen should be restored
// to the original layout before returning to TorchScript.
//
// For details on Part_2, refer to the implementation note. [ Permutation
// Bookkeeping and Propagation in Parser ]
std::vector<at::Tensor> FusionExecutorCache::runFusionWithInputs(
    const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionExecutorCache::runFusionWithInputs");

  // Permute input tensor for kernel execution.
  // See Part_1 in Note [ Channels-Last support in nvfuser ]
  at::ArrayRef<c10::IValue> perm_inputs = inputs;
  const auto& to_be_permuted_inputs = fusion_->getPermutationInputMap();
  std::vector<c10::IValue> inputs_vec;
  if (!to_be_permuted_inputs.empty()) {
    inputs_vec = inputs.vec();
    for (const auto& pair : to_be_permuted_inputs) {
      auto v = inputs_vec[pair.first];
      TORCH_CHECK(
          v.isTensor(), "input permutation can only be applied at tensor");
      auto tensor = v.toTensor();
      inputs_vec[pair.first] = tensor.permute(pair.second);
    }
    perm_inputs = inputs_vec;
  }

  KernelArgumentHolder args = prepareInputs(perm_inputs);

  auto kernel_runtime = getKernelRuntimeFor(args);
  most_recent_runtime_ = kernel_runtime;
  int seq_id = 0;
  // Record kernel input and output tensors so profiler can construct
  // the data flow graph
  RECORD_FUNCTION(
      "run_fused_kernel",
      std::vector<c10::IValue>(inputs.begin(), inputs.end()),
      seq_id);
  auto outputs = kernel_runtime->runWithInputs(args);
  RECORD_OUTPUTS(outputs);

  // Permute output tensor returned by kernel execution.
  // See Part_3 in Note [ Permutation support in nvfuser ]
  for (const auto& pair : fusion_->getPermutationOutputMap()) {
    if (size_t(pair.first) < outputs.size()) {
      outputs[pair.first] = outputs[pair.first].permute(pair.second);
    }
  }

  // Removing aliased outputs, since those are updated by the Fusion. It is not
  // semantically correct to actually return them as outputs from fusion.
  int offset = 0;
  const auto& indices = fusion_->getOutputAliasIndices();
  std::set<int> aliased_output_indices(indices.begin(), indices.end());
  for (const auto& v : aliased_output_indices) {
    outputs.erase(outputs.begin() + v - offset);
    offset++;
  }

  return outputs;
}

std::string FusionExecutorCache::getCode(
    FusionKernelRuntime* kernel_runtime,
    bool intrinsic_code) const {
  std::string kernel_code;
  TORCH_CHECK(kernel_runtime != nullptr, "Invalid fusion definition!");
  TORCH_CHECK(kernel_runtime->isCompiled(), "Fusion is not compiled!");

  bool first_kernel = true;
  for (const auto& exec : kernel_runtime->executors()) {
    if (first_kernel) {
      first_kernel = false;
    } else {
      kernel_code += "\n";
    }
    kernel_code += exec.kernelString();
  }

  if (intrinsic_code) {
    const auto& execs = kernel_runtime->executors();
    const FusionExecutor& fe = execs[0];
    auto index_type = fe.kernel()->indexType();
    // Make sure all the segment index types match. All segments currently
    // use the same index type but this code change in the future.
    for (const auto& exec : execs) {
      TORCH_CHECK(
          index_type == exec.kernel()->indexType(),
          "Index Type mismatch between Segment Executors: ",
          index_type,
          " ",
          exec.kernel()->indexType());
    }
    std::string full_code = fe.getStructuredCode(kernel_code, index_type);
    return full_code;
  } else {
    return kernel_code;
  }
}

std::string FusionExecutorCache::getMostRecentCode(bool intrinsic_code) const {
  return getCode(most_recent_runtime_, intrinsic_code);
}

std::string FusionExecutorCache::getCodeFor(
    const at::ArrayRef<c10::IValue>& inputs,
    bool intrinsic_code) {
  KernelArgumentHolder args = prepareInputs(inputs);
  auto kernel_runtime = getKernelRuntimeFor(args);
  return getCode(kernel_runtime, intrinsic_code);
}

std::string FusionExecutorCache::getScheduledIr(
    FusionKernelRuntime* kernel_runtime,
    bool tensor_transforms) const {
  TORCH_CHECK(kernel_runtime != nullptr, "Invalid fusion definition!");
  TORCH_CHECK(kernel_runtime->isCompiled(), "Fusion is not compiled!");
  std::stringstream ss;
  if (kernel_runtime->isSegmented()) {
    auto fs = kernel_runtime->fusionSegments();
    ss << "Segmented_Fusion Dump: -- Re-written complete fusion:{\n";
    fs->completeFusion()->print(ss, false);
    ss << "} // {Re-written complete fusion}\n";
    ss << fs << "\n";
  }
  for (auto& exec : kernel_runtime->executors()) {
    auto sched_ir = exec.kernel()->as<Fusion>();
    sched_ir->print(ss, tensor_transforms);
  }
  return ss.str();
}

std::string FusionExecutorCache::getMostRecentScheduledIr(
    bool tensor_transforms) const {
  return getScheduledIr(most_recent_runtime_, tensor_transforms);
}

std::string FusionExecutorCache::getScheduledIrFor(
    const at::ArrayRef<c10::IValue>& inputs,
    bool tensor_transforms) {
  KernelArgumentHolder args = prepareInputs(inputs);
  auto kernel_runtime = getKernelRuntimeFor(args);
  return getScheduledIr(kernel_runtime, tensor_transforms);
}

void FusionExecutorCache::evictCache(size_t cache_id) {
  auto it = id_to_kernel_runtime_.find(cache_id);
  TORCH_INTERNAL_ASSERT(it != id_to_kernel_runtime_.end());
  it->second->evictCache(cache_id);
  id_to_kernel_runtime_.erase(it);
}

FusionKernelRuntime* FusionExecutorCache::getKernelRuntimeFor(
    const KernelArgumentHolder& args) {
  // Check for id hit case
  auto unique_id = *args.getCacheId();
  auto id_it = id_to_kernel_runtime_.find(unique_id);
  if (id_it != id_to_kernel_runtime_.end()) {
    return id_it->second;
  }

  // Access kernels associated with the common device id
  auto& kernel_runtimes = kernel_runtimes_[args.getDeviceIndex()];

  // Check for re-use hit case
  //  a kernel runtime is re-usable if all the compiled
  //  kernels have the same heuristic parameters
  std::unique_ptr<FusionHeuristics> new_heuristics;

  auto reuse_it = std::find_if(
      kernel_runtimes.begin(),
      kernel_runtimes.end(),
      [&args, &new_heuristics](auto& kernel_runtime) {
        auto maybe_heuristics = kernel_runtime->getMaybeHeuristicsFor(args);
        if (!maybe_heuristics.has_value()) {
          return false;
        }
        new_heuristics = std::move(maybe_heuristics.value());
        return true;
      });

  FusionKernelRuntime* kernel_runtime = nullptr;
  if (reuse_it != kernel_runtimes.end()) {
    kernel_runtime = reuse_it->get();
    kernel_runtime->updateHeuristicsLaunchParams(new_heuristics.get());
  } else {
    // graph miss, need to re-build an optimized graph for this case
    kernel_runtimes.emplace_back(
        std::make_unique<FusionKernelRuntime>(fusion_.get(), args));
    kernel_runtime = kernel_runtimes.back().get();
    if (profiling_) {
      kernel_runtime->profile(true);
    }
  }

  id_to_kernel_runtime_[unique_id] = kernel_runtime;
  return kernel_runtime;
}

FusionKernelRuntime::FusionKernelRuntime(
    Fusion* fusion,
    const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::FusionKernelRuntime");

  // Make a copy of fusion and do segmentation and translation
  //  on this copy
  auto fusion_copy = std::make_unique<Fusion>(*fusion);

  // Run segmentation on the copied fusion
  SchedulerRuntimeInfo runtime_info(fusion_copy.get(), args, true);

  // Initialize the evaluator simplifer
  precomputed_values_ = std::make_unique<PrecomputedValues>(fusion_copy.get());

  //! Try to schedule the complete fusion
  scheduler_debug_utils::canScheduleMessage(
      "***Runtime***: Try to schedule fusion un-segmented:\n");

  const auto maybe_complete_fusion_heuristic =
      SchedulerEntry::proposeHeuristics(fusion_copy.get(), runtime_info);

  //! Decide if this fusion is segmented or not
  const bool segmented = !maybe_complete_fusion_heuristic.has_value();

  if (segmented) {
    // Take ownership and segment transformed fusion
    segmented_fusion_ =
        SegmentCandidateFinder::segment(std::move(fusion_copy), args);
  } else {
    segmented_fusion_ = SegmentedFusion::fromCompleteFusion(
        std::move(fusion_copy), maybe_complete_fusion_heuristic.value(), args);
  }

  heuristics_ = segmented_fusion_->makeInitialHeuristics(args);
  executors_ = std::vector<FusionExecutor>(segmented_fusion_->groups().size());
  if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
    segmented_fusion_->print();
  }

  // Even if we go through the segmented path we may still end up
  //  with a segmented fusion with one group. This case still
  //  counts as un-segmented.
  is_segmented_ = segmented_fusion_->groups().size() > 1;

  // Pre-compute the executor order so that the run time path
  //  would go directly to kernel launch.
  prepareRuntimeOrder();
}

std::vector<at::Tensor> FusionKernelRuntime::runKernelWithInput(
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
  TORCH_INTERNAL_ASSERT(sg, "runKernelWithInput: need valid group to run");
  auto [launch_params, compile_params] = compileKernel(args, sg);
  auto group_id = sg->groupId();
  auto scheduler_entry = schedulers().at(group_id).get();
  auto& executor = executors_.at(group_id);

  if (profiling_) {
    most_recent_executor_log_.fusion_executor = &executor;
    most_recent_executor_log_.params = scheduler_entry->params()->clone();
  }

  if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    executor.setMeasureKernelTimeFlag(true);
  }

  auto outputs = executor.runFusion(args, launch_params, compile_params);

  // Print relevant information all at once for easy debuging of perf
  if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    std::cout << "\nRun kernel:\n";
    if (sg) {
      segmented_fusion_->makeFusion(sg)->printMath();
    } else {
      segmented_fusion_->completeFusion()->printMath();
    }
    std::cout << "With inputs:\n";
    for (auto i : c10::irange(args.size())) {
      std::cout << "  " << args[i]->toString() << std::endl;
    }
    std::cout << "Compiler log: " << executor.compilerLog() << "\n";
    std::cout << scheduler_entry->params()->toString() << "\n";
    std::cout << "With arguments: " << executor.lastLaunchParams().toString();
    std::cout << executor.kernelName() << " " << executor.bytesProcessed()
              << " bytes/ " << std::setprecision(3) << executor.kernelTimeMs()
              << " ms "
              << ((double)executor.bytesProcessed() /
                  ((double)executor.kernelTimeMs() / 1000)) /
            (double)1.0e9
              << " GB/s" << std::endl;
    executor.setMeasureKernelTimeFlag(false);
  }

  return outputs;
}

void FusionKernelRuntime::prepareRuntimeOrder() {
  // Setup group run order:
  std::unordered_set<Val*> available_input;

  // setup the order tensor dimensions are bound
  for (const size_t i : c10::irange(segmented_fusion_->inputs().size())) {
    auto input_val = segmented_fusion_->inputs()[i];
    available_input.insert(input_val);

    if (auto input_tv = dynamic_cast<TensorView*>(input_val)) {
      auto root_dom = TensorDomain::noReductions(input_tv->getRootDomain());
      for (const size_t dim : c10::irange(root_dom.size())) {
        const auto extent = root_dom[dim]->extent();
        available_input.insert(extent);
        runtime_workspace_.group_extent_binding_order.push_back(extent);
      }
    }
  }

  // Keep track of groups that has run
  std::vector<bool> group_ran(segmented_fusion_->groups().size(), false);

  while (!std::all_of(
      group_ran.begin(), group_ran.end(), [](bool b) { return b; })) {
    bool one_ran = false;

    // Find the first segment with all inputs available to run
    for (const size_t group_i :
         c10::irange(segmented_fusion_->groups().size())) {
      auto& group = segmented_fusion_->groups()[group_i];
      if (group_ran[group_i]) {
        continue;
      }
      const auto& group_inputs = group->inputs();
      bool ready_to_run = std::all_of(
          group_inputs.begin(),
          group_inputs.end(),
          [&available_input](Val* val) { return available_input.count(val); });

      if (ready_to_run) {
        runtime_workspace_.group_run_order.push_back(group);
        const auto& group_outputs = group->outputs();

        // Insert graph segment output to tensor map
        for (const size_t group_out_i : c10::irange(group_outputs.size())) {
          available_input.insert(group_outputs[group_out_i]);
        }
        group_ran[group_i] = true;
        one_ran = true;
      }
    }
    TORCH_INTERNAL_ASSERT(
        one_ran,
        "Couldn't run all groups, something must have gone wrong in segmentation.");
  }
}

// passing args by value, since we will be modify this
void FusionKernelRuntime::startAsyncCompile(
    const KernelArgumentHolder& args_old) {
  // only single compilation is supported at this moment.
  std::unique_lock<std::mutex> unique_lock(mutex_, std::try_to_lock);
  TORCH_CHECK(
      unique_lock.owns_lock(),
      "Calling startAsyncCompile on a FusionKernelRuntime that has already",
      " started a compilation thread is not supported.",
      " - unique_lock");
  std::unique_lock<std::mutex> unique_lock2(compiling_, std::try_to_lock);
  TORCH_CHECK(
      unique_lock2.owns_lock(),
      "Calling startAsyncCompile on a FusionKernelRuntime that has already",
      " started a compilation thread is not supported.",
      " - unique_lock2");

  // TODO: Compilation can happen in parallel! We can infer the output sizes
  // on the non-compiled kernel and build the entire tensor_map prior to
  // asyc compilation.

  // PyTorch's threadpool uses std::function, which requires the target to be
  // copy-constructible. Adding a std::unique_lock to the lambda's capture list
  // prevents it from being copyable. The std::unique_lock can be moved, but it
  // cannot be copied. Thus, we need the second mutex.
  auto compile_fusion = [args = args_old, this]() mutable {
    std::lock_guard<std::mutex> guard(compiling_);

    // locking mutex_ since we are touching executors_ during compilation.
    // CUDAGuard uses runtime API directly, which is thread safe.
    c10::cuda::CUDAGuard dg((int8_t)args.getDeviceIndex());

    FUSER_PERF_SCOPE("FusionKernelRuntime::startAsyncCompile");
    runSegmentsWithInputs(args, true /* is_dry_run */);
  };

  getThreadPool()->run(compile_fusion);
}

KernelArgumentHolder FusionKernelRuntime::dryRunKernelWithInput(
    const KernelArgumentHolder& args,
    SegmentedGroup* sg) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::dryRunKernelWithInput");
  TORCH_INTERNAL_ASSERT(sg, "compileKernel: need valid group to run");
  auto [launch_params, compile_params] = compileKernel(args, sg);
  return executors_[sg->groupId()].inferOutputSizes(args, launch_params);
}

std::pair<LaunchParams, CompileParams> FusionKernelRuntime::compileKernel(
    const KernelArgumentHolder& args,
    SegmentedGroup* sg) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::compileKernel");
  auto group_id = sg->groupId();
  auto scheduler_entry = schedulers().at(group_id).get();

  // Check that the heuristics are matched, in the case of segmented fusion
  TORCH_INTERNAL_ASSERT(!sg || scheduler_entry->heuristic() == sg->heuristic());

  if (!executors_.at(group_id).compiled()) {
    FUSER_PERF_SCOPE("FusionKernelRuntime::compileKernel::Compile");
    // Running a segment group as a single kernel,
    // make a fusion to run from segmented fusion
    std::unique_ptr<Fusion> fusion_to_run = segmented_fusion_->makeFusion(sg);
    FusionGuard fg(fusion_to_run.get());

    scheduler_entry->schedule(fusion_to_run.get());

    TORCH_INTERNAL_ASSERT(
        scheduler_entry->params()->cparams.index_type.has_value(),
        "Kernel index type is not defined.");
    executors_.at(group_id).compileFusion(
        fusion_to_run.get(),
        args,
        scheduler_entry->params()->lparams,
        scheduler_entry->params()->cparams);
  }

  return std::make_pair(
      scheduler_entry->params()->lparams, scheduler_entry->params()->cparams);
}

std::unordered_map<Val*, const ArgAbstract*> FusionKernelRuntime::
    mapFusionInputsToArgs(KernelArgumentHolder& args) {
  std::unordered_map<Val*, const ArgAbstract*> tensor_map;
  int extent_index = 0;
  auto original_args_size = args.size();
  // Bind args in the tensor_map
  for (const auto i : c10::irange(original_args_size)) {
    tensor_map.emplace(segmented_fusion_->inputs()[i], args[i]);
    // Bind tensorview inputs values in case some segmented group
    //  needs it down the road.
    // TODO: we probably have done this already up to this point
    //      should consider caching the expression evaluators, both
    //      more convenient and safer than replication
    if (auto tensor_arg_abstract =
            dynamic_cast<const TensorArgAbstract*>(args[i])) {
      // Note this is very ugly way. We are pushing every single extent to args,
      // because we don't have a better place to hold them.
      auto rank = tensor_arg_abstract->getRank();
      for (const auto dim : c10::irange(rank)) {
        args.push(tensor_arg_abstract->getSize((int)dim));
        tensor_map.emplace(
            runtime_workspace_.group_extent_binding_order[extent_index++],
            args.back());
      }
    }
  }
  return tensor_map;
}

std::vector<at::Tensor> FusionKernelRuntime::runWithInputs(
    KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::runWithInputs");

  if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    std::cout << "=================RUNNING FUSION SEGMENTS================="
              << std::endl;
  }

  c10::Device device(c10::DeviceType::CUDA, (int8_t)args.getDeviceIndex());
  std::unordered_map<Val*, const ArgAbstract*> tensor_map =
      runSegmentsWithInputs(args, false /* is_dry_run */);

  if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    std::cout << "============= FINISHED RUNNING FUSION SEGMENTS ============"
              << std::endl;
  }

  // Produce final global output
  std::vector<at::Tensor> fusion_outputs;
  for (auto output : segmented_fusion_->outputs()) {
    const auto iter = tensor_map.find(output);
    if (iter != tensor_map.end()) {
      // Note [ trivial forwarding ]
      //
      // Background:
      // NvFuser codegen does not handle aliases. When we have a fusion that
      // forwards an input to output without any operations on it, this is
      // a no-op for codegen and the output tensor is never written to. However,
      // the codegen cannot "forward" an input to output, since all outputs are
      // allocated in integration. If we do not special case it, we'll ended up
      // having a "fresh" tensor allocated for the forwarded-input.
      //
      // Approach:
      // There are two aspects of the support:
      // 1) Codegen handles forwarding implicitly. Forwarded inputs do not
      // have any producer in the IR, so the output argument is not used in
      // the code. However, it is required to be a kernel argument, which acts
      // as a place-holder, so we can map the arguments correctly.
      //
      // 2) Integration handles the trivial forwarding of inputs. When we put
      // together `fusion_outputs` for a given fusion and the outputs are
      // fusion inputs, we directly return the input tensor.
      auto arg = dynamic_cast<const TensorArgAbstract*>(iter->second);
      fusion_outputs.push_back(arg->getTensor());
    } else {
      bool empty_type_check = output->getDataType().has_value() &&
          output->getDataType().value() == DataType::Float;

      // Only support two cases of empty tensor here, since this is hot path.
      auto out_tv = output->as<TensorView>();

      // TODO: should be only one of the two once the "empty"
      //  definition has been unified throughout the ops.
      bool empty_tensor_check = out_tv->isZeroDim() || out_tv->isEmptyTensor();

      // This is the check for an empty tensor;
      TORCH_INTERNAL_ASSERT(
          empty_tensor_check && empty_type_check,
          "Is empty tensor? ",
          !empty_tensor_check,
          " Is empty type check? ",
          !empty_type_check,
          " Output empty tensor check failed for tensor: ",
          out_tv->toString(),
          " In function: ",
          __FUNCTION__);

      // TODO: would need to clean up this part when
      //   we have a unified and consistent way to generate
      //   size-0 tensors.
      const auto tensor_options =
          at::TensorOptions().dtype(at::kFloat).device(device);
      fusion_outputs.emplace_back(at::empty({0}, tensor_options));
    }
  }
  return fusion_outputs;
}

std::unordered_map<Val*, const ArgAbstract*> FusionKernelRuntime::
    runSegmentsWithInputs(KernelArgumentHolder& args, bool is_dry_run) {
  TORCH_INTERNAL_ASSERT(
      args.size() == segmented_fusion_->inputs().size(),
      "Inputs were not set up correctly, received ",
      args.size(),
      " inputs but expected ",
      segmented_fusion_->inputs().size());

  std::unordered_map<Val*, const ArgAbstract*> tensor_map =
      mapFusionInputsToArgs(args);

  auto update_outputs = [&args, &tensor_map](
                            auto group_outputs, auto group_runtime_outputs) {
    // Insert graph segment output to tensor map
    TORCH_INTERNAL_ASSERT(
        group_outputs.size() == group_runtime_outputs.size(),
        "Output size does not match.");

    // Trivial forwarding outputs an empty tensor to save bandwidth. We skip
    // updating the tensor_map because we want all future use of inputs on
    // the original tensor input. See note [Trivial Forwarding]
    for (const size_t group_out_i : c10::irange(group_outputs.size())) {
      if (!group_outputs[group_out_i]->isFusionInput()) {
        args.push(group_runtime_outputs[group_out_i]);
        tensor_map.emplace(group_outputs[group_out_i], args.back());
      }
    }
  };

  // group should share cache id.
  auto group_cache_id = args.getCacheId();
  for (auto group_to_run : runtime_workspace_.group_run_order) {
    // TODO: index mode should be updated per segmented kernel
    // Prepare input vector
    KernelArgumentHolder group_runtime_inputs(args.getIndexMode());
    group_runtime_inputs.setDeviceIndex(args.getDeviceIndex());
    if (group_cache_id.has_value()) {
      group_runtime_inputs.setCacheId(group_cache_id.value());
    }
    for (auto input : group_to_run->inputs()) {
      group_runtime_inputs.push(tensor_map.at(input));
    }

    // TODO: currently we are still outputing PyTorch tensors, instead of
    // something abstract. This is quite unsatisfying.

    // Run graph segment
    if (is_dry_run) {
      KernelArgumentHolder group_runtime_outputs =
          dryRunKernelWithInput(group_runtime_inputs, group_to_run);
      update_outputs(group_to_run->outputs(), group_runtime_outputs);
    } else {
      std::vector<at::Tensor> group_runtime_outputs =
          runKernelWithInput(group_runtime_inputs, group_to_run);
      update_outputs(group_to_run->outputs(), group_runtime_outputs);
    }
  }
  return tensor_map;
}

const std::vector<FusionKernelRuntime::SchedulerEntryPtr>& FusionKernelRuntime::
    schedulers() {
  return heuristics_->heuristicsList();
}

void FusionKernelRuntime::updateHeuristicsLaunchParams(
    FusionHeuristics* update_heuristics) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::updateHeuristicsLaunchParams");
  auto scheduler_list_length = heuristics_->heuristicsList().size();
  TORCH_INTERNAL_ASSERT(
      update_heuristics->heuristicsList().size() == scheduler_list_length);
  for (const auto i : c10::irange(scheduler_list_length)) {
    auto& schedulerPtr = heuristics_->heuristicsList()[i];
    schedulerPtr->updateLaunchConstraint(
        update_heuristics->heuristicsList()[i]->params()->lparams);
  }
}

c10::optional<FusionKernelRuntime::HeuristicsPtr> FusionKernelRuntime::
    getMaybeHeuristicsFor(const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::getMaybeHeuristicsFor");
  auto complete_fusion = segmented_fusion_->completeFusion();
  SchedulerRuntimeInfo runtime_info(complete_fusion, args);
  precomputed_values_->bindInputs(args);
  precomputed_values_->evaluate();
  runtime_info.expressionEvaluator().bindPrecomputedValues(
      precomputed_values_.get());

  c10::optional<FusionKernelRuntime::HeuristicsPtr> ret;
  ret = std::make_unique<FusionHeuristics>();
  size_t total_groups = segmented_fusion_->groups().size();
  for (const auto group_index : c10::irange(total_groups)) {
    auto group = segmented_fusion_->groups()[group_index];

    auto maybe_scheduler_entry = group->getMaybeSchedulerEntry(runtime_info);
    if (!maybe_scheduler_entry.has_value()) {
      return c10::nullopt;
    }
    auto scheduler_entry = std::move(maybe_scheduler_entry.value());
    if (!scheduler_entry->sameAs(
            heuristics_->heuristicsList()[group_index].get())) {
      return c10::nullopt;
    }
    ret.value()->emplaceBack(std::move(scheduler_entry));
  }

  return ret;
}

void GraphCache::createFusion(const std::shared_ptr<torch::jit::Graph>& graph) {
  FUSER_PERF_SCOPE("GraphCache::createFusion");

  fusion_executor_cache_ =
      std::make_unique<FusionExecutorCache>(parseJitIR(graph));

  num_of_outputs_ = graph->outputs().size();
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
GraphCache::GraphCache(const std::shared_ptr<torch::jit::Graph>& graph) {
  FUSER_PERF_SCOPE("GraphCache::GraphCache");
  TORCH_INTERNAL_ASSERT(
      torch::jit::IsNewExecutorEnabled(),
      "legacy executor is not supported by nvfuser");

  GRAPH_DEBUG("GraphCache constructor: ", this);
  GRAPH_DUMP("GraphCache created for graph", graph);
  createFusion(graph);
}

std::vector<at::Tensor> GraphCache::runGraphWithInputs(
    const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("GraphCache::runGraphWithInputs");

  GRAPH_DEBUG("running GraphCache: ", this);
  auto outputs = fusion_executor_cache_->runFusionWithInputs(inputs);
  TORCH_INTERNAL_ASSERT(
      outputs.size() == num_of_outputs_,
      "FusionExecutorCache returned ",
      outputs.size(),
      " outputs, doesn't match computational graph, which requires ",
      num_of_outputs_);

  return outputs;
}

std::string KernelArgumentHolder::toString() const {
  std::stringstream ss;
  for (const auto& arg : arguments_) {
    ss << arg->toString() << "\n";
  }
  return ss.str();
}

} // namespace nvfuser

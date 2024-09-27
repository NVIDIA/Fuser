// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <kernel_cache.h>

#include <mutex>
#include <sstream>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>

#include <debug.h>
#include <driver_api.h>
#include <dynamic_transform.h>
#include <fusion_executor/allocations.h>
#include <fusion_executor/executor_params.h>
#include <fusion_executor/executor_utils.h>
#include <fusion_profiler.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <multidevice/communicator.h>
#include <options.h>
#include <preseg_passes/pre_segmenter.h>
#include <scheduler/debug_utils.h>
#include <scheduler/registry.h>
#include <utils.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace nvfuser {

namespace {

// Replace CUDA tensor with Meta tensor because storing tensors can cause
// out-of-memory issues. Other arguments are returned as-is.
std::shared_ptr<PolymorphicValue> convertMetadataArg(
    std::shared_ptr<PolymorphicValue> arg) {
  if (arg->is<at::Tensor>()) {
    if (const auto& tensor = arg->as<at::Tensor>(); tensor.is_cuda()) {
      auto meta_tensor = at::Tensor(at::detail::empty_strided_meta(
          tensor.sizes(),
          tensor.strides(),
          tensor.scalar_type(),
          c10::nullopt,
          c10::Device(c10::DeviceType::Meta, 0),
          c10::nullopt));
      return std::make_shared<PolymorphicValue>(std::move(meta_tensor));
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

// Copy bytes of value to back of buffer. This is templated in order to avoid
// implicit cast such as int64_t -> size_t that might lose information.
template <typename T>
void encodeBuffer(T value, std::string& buffer) {
  const char* v = reinterpret_cast<char*>(&value);
  for (const auto i : c10::irange(sizeof(T))) {
    (void)i; // Suppress unused variable warning
    buffer.push_back(*(v++));
  }
}

// This ArgumentManager do two things
// (1) add outputs from a segment to the global fusion args to pass it to next
// segment (2) delete args no longer being used to save memory. For task (2), it
// checks the input and output arguments of each segment and make a map from val
// to the segment_id where the val is lastly used. The arguments representing
// these vals are then deleted after the segment runs.
class ArgumentManager {
 public:
  ArgumentManager(
      KernelArgumentHolder& args,
      const RuntimeWorkSpace& runtime_workspace,
      const std::vector<Val*>& fusion_inputs)
      : fusion_args_(args) {
    // map from val to args
    mapFusionInputsToArgs(
        fusion_inputs, runtime_workspace.group_extent_binding_order);
    setLastUsedSegmentID(runtime_workspace.group_run_order);
  }
  const std::unordered_map<Val*, const PolymorphicValue*>& getTensorMap() {
    return tensor_map_;
  }
  const PolymorphicValue* checkTensorMap(Val* v) {
    return tensor_map_.at(v);
  }
  // T is assumed to be either std::vector<at::Tensor> or KernelArgumentHolder
  // (from dry run)
  // TODO: make the output type uniform no matter it's a real or dry run
  template <typename T>
  void updateWithSegmentOutputs(
      const std::vector<Val*>& group_outputs,
      const T& group_runtime_outputs,
      const int64_t group_id) {
    addOutputsToArgsAndTensorMap(group_outputs, group_runtime_outputs);
    deleteUnusedArgs(group_id);
  }

 private:
  KernelArgumentHolder& fusion_args_;
  // map from val to args
  std::unordered_map<Val*, const PolymorphicValue*> tensor_map_;
  // map segment_id to vector of fusion vals lastly used at this segment
  std::unordered_map<int64_t, std::vector<Val*>> vals_last_used_at_segment_;

  void mapFusionInputsToArgs(
      const std::vector<Val*>& fusion_inputs,
      const std::vector<Val*>& group_extent_binding_order) {
    int extent_index = 0;
    auto original_args_size = fusion_args_.size();
    // Bind args in the tensor_map
    for (const auto i : c10::irange(original_args_size)) {
      tensor_map_.emplace(fusion_inputs[i], fusion_args_[i]);
      // Bind tensorview inputs values in case some segmented group
      //  needs it down the road.
      // TODO: we probably have done this already up to this point
      //      should consider caching the expression evaluators, both
      //      more convenient and safer than replication
      if (fusion_args_[i]->is<at::Tensor>()) {
        // Note this is very ugly way. We are pushing every single extent to
        // args, because we don't have a better place to hold them.
        auto rank = fusion_args_[i]->as<at::Tensor>().dim();
        for (const auto dim : c10::irange(rank)) {
          fusion_args_.push(
              PolymorphicValue(fusion_args_[i]->as<at::Tensor>().size(dim)));
          tensor_map_.emplace(
              group_extent_binding_order[extent_index++], fusion_args_.back());
        }
      }
    }
  }

  void setLastUsedSegmentID(
      const std::vector<SegmentedGroup*>& group_run_order) {
    // never delete global fusion inputs and outputs
    auto isFusionInputOrOutput = [](Val* val) {
      return val->isFusionInput() || val->isFusionOutput();
    };
    // map val to segment_id where arg is lastly used
    std::unordered_map<Val*, int64_t> last_used_segment_map;
    const int64_t num_groups = (int64_t)group_run_order.size();
    // only need to set lifetime of vals if there are more than 3 groups
    if (num_groups >= 3) {
      // start from the 2nd group, since the input of the first group is always
      // the global input and its outputs are always used by at least one of the
      // following groups
      for (auto run_order_id : c10::irange(1l, num_groups)) {
        auto group_to_run = group_run_order.at(run_order_id);
        // set/update life of vals in inputs of this group
        for (auto val : group_to_run->inputs()) {
          // skip fusion inputs and outputs, they may be used by other fusions
          // or code
          if (!isFusionInputOrOutput(val)) {
            last_used_segment_map[val] = run_order_id;
          }
        }
        // set/update life of vals in outputs of this group
        // skip the last group since its outputs are always the global outputs
        if (run_order_id < num_groups - 1) {
          for (auto val : group_to_run->outputs()) {
            // skip fusion inputs and outputs, they may be used by other fusions
            // or code
            if (!isFusionInputOrOutput(val)) {
              last_used_segment_map[val] = run_order_id;
            }
          }
        }
      }
      // convert to vals_last_used_at_segment_, so we don't need to iterate over
      // all vals when erasing
      for (auto item : last_used_segment_map) {
        vals_last_used_at_segment_[item.second].push_back(item.first);
      }
    }
  }

  void deleteUnusedArgs(int64_t run_order_id) {
    // erase args corresponding to vals lastly used in this segment
    if (run_order_id >= 1 && vals_last_used_at_segment_.count(run_order_id)) {
      for (auto val : vals_last_used_at_segment_[run_order_id]) {
        fusion_args_.erase(tensor_map_.at(val));
        tensor_map_.erase(val);
      }
    }
  }

  template <typename T>
  void addOutputsToArgsAndTensorMap(
      const std::vector<Val*>& group_outputs,
      const T& group_runtime_outputs) {
    // Insert graph segment output to tensor map
    NVF_ERROR(
        group_outputs.size() == group_runtime_outputs.size(),
        "Output size does not match.");

    for (const size_t group_out_i : c10::irange(group_outputs.size())) {
      Val* output = group_outputs[group_out_i];
      const PolymorphicValue*& runtime_output = tensor_map_[output];
      if (runtime_output != nullptr) {
        // A trivial forwarding output or a dupliated output shares the same
        // `Val*` as another fusion input/output. In those cases, we keep
        // mapping it to the same runtime output.
        continue;
      }

      if constexpr (std::is_pointer_v<
                        decltype(group_runtime_outputs[group_out_i])>) {
        fusion_args_.push(*group_runtime_outputs[group_out_i]);
      } else {
        fusion_args_.push(group_runtime_outputs[group_out_i]);
      }
      runtime_output = fusion_args_.back();
    }
  }
};

} // namespace

flatbuffers::Offset<serde::InputsIdLookup> InputsIdLookup::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See definitions in serde/fusion_cache.fbs for table
  // InputsIdLookup and struct EncodingEntry

  using fb_string = flatbuffers::Offset<flatbuffers::String>;

  // For serialization, we require a consistent ordering for the
  // encoding_lookup_ map.
  std::unordered_map<std::string, size_t> str_key_ordering;

  // 1. Serialize used_entry_ list
  std::vector<fb_string> lru_cache_fb;
  for (const auto& str : used_entry_) {
    lru_cache_fb.push_back(builder.CreateString(str));
    str_key_ordering.emplace(str, str_key_ordering.size());
  }

  // 2. Serialize encoding_lookup_ map
  std::vector<fb_string> encoding_lookup_keys_fb;
  std::vector<serde::EncodingEntry> encoding_lookup_values_fb;
  for (auto&& [key, value] : encoding_lookup_) {
    encoding_lookup_keys_fb.push_back(builder.CreateString(key));
    encoding_lookup_values_fb.emplace_back(value.id, str_key_ordering.at(key));
  }

  return serde::CreateInputsIdLookupDirect(
      builder,
      max_cache_size_,
      current_id_,
      &lru_cache_fb,
      &encoding_lookup_keys_fb,
      &encoding_lookup_values_fb);
}

void InputsIdLookup::deserialize(const serde::InputsIdLookup* buffer) {
  // See definitions in serde/fusion_cache.fbs for tables
  // InputsIdLookup and EncodingEntry
  NVF_ERROR(buffer != nullptr, "serde::InputsIdLookup is nullptr.");
  using list_iter = std::list<std::string>::iterator;
  std::vector<list_iter> used_entry_iterators;

  max_cache_size_ = buffer->max_cache_size();
  current_id_ = buffer->current_id();
  for (auto fb_str : *buffer->lru_cache()) {
    used_entry_.emplace_back(fb_str->str());
    used_entry_iterators.emplace_back(std::prev(used_entry_.end()));
  }

  for (auto idx : c10::irange(buffer->encoding_lookup_keys()->size())) {
    auto fb_encoding_lookup_str = buffer->encoding_lookup_keys()->Get(idx);
    auto fb_encoding_entry = buffer->encoding_lookup_values()->Get(idx);

    EncodingEntry entry{
        fb_encoding_entry->id(),
        used_entry_iterators.at(fb_encoding_entry->lru_iter())};
    encoding_lookup_.emplace(fb_encoding_lookup_str->str(), entry);
  }
}

InputsIdLookup::IdLookupReturn InputsIdLookup::lookupId(
    const at::ArrayRef<c10::IValue>& inputs,
    const std::unordered_set<size_t>& scalar_inputs_to_record,
    int8_t device) {
  IdLookupReturn ret;

  // lock mutex_ because we are touching encoding_
  std::lock_guard<std::mutex> guard(mutex_);
  encoding_.clear();
  encodeBuffer(device, encoding_);
  for (const auto i : c10::irange(inputs.size())) {
    auto input = inputs[i];
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
      // NOTE: device is set for the whole set of inputs first using device arg
    } else {
      // encode s for scalar;
      encoding_.push_back('s');
      if (scalar_inputs_to_record.find(i) != scalar_inputs_to_record.end()) {
        // Add value of scalars here only if it is one of the scalars
        // provided, as these are used in determining concretization.
        // Note that although most commonly these will be Int or Bool scalars,
        // any DataType might appear via `cast` and `where`, so we handle all
        // cases here.
        if (input.isInt()) {
          encodeBuffer(input.toInt(), encoding_);
        } else if (input.isBool()) {
          encodeBuffer(input.toBool(), encoding_);
        } else if (input.isDouble()) {
          encodeBuffer(input.toDouble(), encoding_);
        } else if (input.isComplexDouble()) {
          encodeBuffer(input.toComplexDouble(), encoding_);
        } else {
          NVF_THROW(
              "Unhandled input type when creating input ID. Cannot record ",
              input);
        }
      }
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

void prepareRuntimeOrder(
    SegmentedFusion* segmented_fusion,
    RuntimeWorkSpace& runtime_workspace) {
  // Setup group run order:
  std::unordered_set<Val*> available_input;

  // setup the order tensor dimensions are bound
  for (const size_t i : c10::irange(segmented_fusion->inputs().size())) {
    auto input_val = segmented_fusion->inputs()[i];
    available_input.insert(input_val);

    if (auto input_tv = dynamic_cast<TensorView*>(input_val)) {
      auto logical_dom =
          TensorDomain::noReductions(input_tv->getLogicalDomain());
      for (const size_t dim : c10::irange(logical_dom.size())) {
        const auto extent = logical_dom[dim]->getMaybeExpandedExtent();
        available_input.insert(extent);
        runtime_workspace.group_extent_binding_order.push_back(extent);
      }
    }
  }

  // Keep track of groups that has run
  std::vector<bool> group_ran(segmented_fusion->groups().size(), false);

  while (!std::all_of(
      group_ran.begin(), group_ran.end(), [](bool b) { return b; })) {
    bool one_ran = false;

    // Find the first segment with all inputs available to run
    for (const size_t group_i :
         c10::irange(segmented_fusion->groups().size())) {
      auto& group = segmented_fusion->groups()[group_i];
      if (group_ran[group_i]) {
        continue;
      }
      const auto& group_inputs = group->inputs();
      bool ready_to_run = std::all_of(
          group_inputs.begin(),
          group_inputs.end(),
          [&available_input](Val* val) { return available_input.count(val); });

      if (ready_to_run) {
        runtime_workspace.group_run_order.push_back(group);
        const auto& group_outputs = group->outputs();

        // Insert graph segment output to tensor map
        for (const size_t group_out_i : c10::irange(group_outputs.size())) {
          available_input.insert(group_outputs[group_out_i]);
        }
        group_ran[group_i] = true;
        one_ran = true;
      }
    }
    NVF_ERROR(
        one_ran,
        "Couldn't run all groups, something must have gone wrong in segmentation.");
  }
}

FusionExecutorCache::FusionExecutorCache(
    std::unique_ptr<Fusion> fusion,
    int64_t fusion_id,
    bool auto_schedule)
    : fusion_(std::move(fusion)),
      exact_map_(std::make_unique<ExactLogicalDomainMap>(fusion_.get())),
      fusion_id_{fusion_id},
      auto_schedule_(auto_schedule) {}

KernelArgumentHolder FusionExecutorCache::prepareInputs(
    const at::ArrayRef<c10::IValue>& inputs,
    std::optional<int8_t> selected_device) {
  FUSER_PERF_SCOPE("FusionExecutorCache::prepareInputs");

  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder(inputs, selected_device);

  // TODO: move InputsIdLookup inside KernelArgumentHolder;
  // NOTE: We must ensure that the cache id is in fact unique. Dynamic fusions
  // may contain transformations that depend on input scalars, not just on the
  // extents of tensor inputs, so we must at times include those scalars in the
  // unique id. Currently, we include all integer scalar inputs for dynamic
  // fusions. This may not be ideal in all cases, since it will prevent
  // short-circuiting here, resulting in avoidable rebuilds of concretization
  // info.
  auto id_lookup_ret = inputs_id_lookup_.lookupId(
      inputs,
      initialInfo().scalarInputsAffectingConcretization(),
      args.getDeviceIndex());
  if (id_lookup_ret.eviction) {
    evictCache(id_lookup_ret.evict_id);
  }

  args.setCacheId(id_lookup_ret.id);
  return args;
}

bool FusionExecutorCache::isCompiled(
    const at::ArrayRef<c10::IValue>& inputs,
    int8_t device) {
  FUSER_PERF_SCOPE("FusionExecutorCache::isCompiled");

  // Access kernels associated with the common device id
  KernelArgumentHolder args = prepareInputs(inputs);
  args.setDeviceIndex(device);

  return getKernelRuntimeFor(args)->isCompiled();
}

std::vector<at::Tensor> FusionExecutorCache::runFusionWithInputs(
    const at::ArrayRef<c10::IValue>& inputs,
    std::optional<PrimDataType> forced_index_type,
    std::optional<int8_t> selected_device) {
  FUSER_PERF_SCOPE("FusionExecutorCache::runFusionWithInputs");
  // NOTE: This should be the first code in the method to capture all host time
  if (isProfilerEnabled()) {
    FusionProfiler::start(!isProfilerEnabledWithCupti());
  }

  KernelArgumentHolder args = prepareInputs(inputs, selected_device);
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

  int seq_id = 0;
  // Record kernel input and output tensors so profiler can construct
  // the data flow graph
  RECORD_FUNCTION(
      "run_fused_kernel",
      std::vector<c10::IValue>(inputs.begin(), inputs.end()),
      seq_id);
  auto outputs = kernel_runtime->runWithInputs(args);
  RECORD_OUTPUTS(outputs);

  // Kernel time measurement is off by default
  kernel_runtime->disableKernelTimeMeasurement();

  // Removing aliased outputs, since those are updated by the Fusion. It is not
  // semantically correct to actually return them as outputs from
  // fusion.
  NVF_ERROR(fusion->outputs().size() == outputs.size());
  size_t new_size = 0;
  for (size_t out_index = 0; out_index < outputs.size(); out_index++) {
    Val* out = fusion->outputs()[out_index];
    if (!fusion->getOutputAlias(out).hide_output) {
      outputs[new_size] = outputs[out_index];
      new_size++;
    }
  }
  outputs.resize(new_size);

  // NOTE: This should be the last code in the method to capture all host time
  if (isProfilerEnabled()) {
    FusionProfiler::stop();
  }
  if (isProfilerPrintingEnabled()) {
    debug() << FusionProfiler::profile();
  }

  return outputs;
}

std::string FusionExecutorCache::getCode(
    FusionKernelRuntime* kernel_runtime,
    bool intrinsic_code) const {
  std::string kernel_code;
  NVF_CHECK(kernel_runtime != nullptr, "Invalid fusion definition!");
  NVF_CHECK(kernel_runtime->isCompiled(), "Fusion is not compiled!");

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
      NVF_CHECK(
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
  NVF_ERROR(it != id_to_kernel_runtime_.end());
  it->second->evictCache(cache_id);
  id_to_kernel_runtime_.erase(it);
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

// getKernelRuntimeFor inspects the inputs to find a usable FusionKernelRuntime
// as quickly as possible. To do so we cache at multiple levels:
//   A. If we have seen these inputs before, we re-use the FusionKernelRuntime
//   we used last time. Here, we mean the same input tensor sizes, as well as
//   same input scalars if they are used to compute an intermediate or output
//   tensor size.
//   B. We check how we should concretize the dynamic fusion using these
//   inputs. If we have not concretized the fusion this way previously, then we
//   concretize it and create a new FusionKernelRuntime, which means segmenting
//   and compiling new kernels. Otherwise, we check whether we can re-use any of
//   the previously-segmented runtimes.
//      i. We look at all FusionKernelRuntimes that have been used with
//      this concretized fusion.
//      ii. For each of those runtimes, we compare the heuristic parameters for
//      each segment to those that we compute using the current inputs.
//   If we do not find any runtimes whose heuristic parameters match, then we
//   create a new FusionKernelRuntime, which means segmenting and compiling all
//   new kernels.
//
// In summary, we have the following paths, in order of hottest to coldest:
//   1. Input ID cache hit: re-use runtime used last time these inputs were seen
//   2. Concretization match, runtime heuristic params match: re-use runtime
//   after checking concretization/heuristics.
//   3. Concretization match but no runtime heuristic params match. Segment
//   to create new FusionKernelRuntime
//   4. Concretization is unseen: Segment to create a new FusionKernelRuntime
// For re-used shapes, path 1 is most relevant. For dynamic shape problems with
// a large number of unique shapes, path 2 is important. Paths 3 and 4 are slow
// since they both involve re-segmentation and re-compilation of the Fusion.
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
            &initial_info, &expr_eval, exact_map_.get()));
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

  // Check if we missed the KernelRuntime cache (Path 2) and need to generate a
  // new kernel runtime (Path 3/4)
  // By default, we try to avoid recompiling whenever possible. However, this
  // can lead to suboptimal code if we only check that a compiled kernel is able
  // to run with some inputs, instead of whether it is optimal to do so. The
  // NVFUSER_DISABLE=kernel_reuse option is a coarse tool that just enforces
  // that whenever we encounter a new set of input shapes we segment and compile
  // a new FusionKernelRuntime. Effectively, this option disables Paths 2 and 3
  // above so that we only have Path 1 (hottest re-use path) and Path 4 (full
  // recompile).
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
      // Initial info is used during concretization and is owned by conc_fusion.
      // After concretization, we stop managing it so that we won't keep cloning
      // it for every subsequent Fusion copy.
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

    for (auto kernel_idx : c10::irange(device_runtimes.size())) {
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
      // Each FusionKernelRuntime stores a metadata copy of its initial inputs.
      // We deserialize the arguments of the first FusionKernelRuntime to
      // recompute the concretization info.
      KernelArgumentHolder args;
      args.deserialize(fb_device_runtimes->runtimes()->begin()->args());
      auto expr_eval = executor_utils::bindInputs(args, fusion_.get());
      cached_conc_info_.emplace_back(
          std::make_unique<DynamicTransformConcretizationInfo>(
              &initial_info, &expr_eval, exact_map_.get()));
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
      // to create the FusionExecutor objects.
      device_runtimes.back()->deserialize(
          fb_fusion_kernel_runtime, args.getDeviceIndex());

      all_runtimes.emplace_back(device_runtimes.back().get());
    }
  }

  // 2. Rebuild input id to kernel cache
  for (auto idx : c10::irange(buffer->kernel_cache_keys()->size())) {
    size_t key = buffer->kernel_cache_keys()->Get(idx);
    size_t value_id = buffer->kernel_cache_values()->Get(idx);
    id_to_kernel_runtime_.emplace(key, all_runtimes.at(value_id));
  }
}

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
      fusion->printMath();
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
        SegmentCandidateFinder::segment(std::move(fusion), &args, runtime_info);
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

  executors_ = std::vector<FusionExecutor>(segmented_fusion_->groups().size());
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

flatbuffers::Offset<serde::FusionKernelRuntime> FusionKernelRuntime::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See table definition for FusionKernelRuntime in serde/fusion_cache.fbs

  // 1. Serialize FusionExecutor objects
  std::vector<flatbuffers::Offset<serde::FusionExecutor>> executors_fb;
  executors_fb.reserve(executors_.size());
  for (auto& executor : executors_) {
    executors_fb.push_back(executor.serialize(builder));
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

  // 1. Deserialize FusionExecutor objects
  for (auto idx : c10::irange(buffer->executors()->size())) {
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

    executors_.at(group_id).deserialize(
        buffer->executors()->Get(group_id),
        fusion_to_run.get(),
        device_index,
        heuristic_params->cparams,
        heuristic_params->scheduler_type,
        fusion_id_,
        concrete_id_,
        runtime_id_,
        group_id);
  }
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
  NVF_ERROR(sg, "runKernelWithInput: need valid group to run");
  auto [launch_params, compile_params] = getKernelConfig(args, sg);
  auto group_id = sg->groupId();
  auto heuristic_params = schedulers().at(group_id).get();
  auto& executor = executors_.at(group_id);

  if (profiling_) {
    most_recent_executor_log_.fusion_executor = &executor;
    most_recent_executor_log_.params = heuristic_params->clone();
  }

  // TODO: This is a work around for the fallback execution path where a kernel
  // is not compiled. Perhaps the gorup/segment Id needs to be specified to the
  // executor at its constructor.  Currently, initialization is ad hoc.
  if (executor.groupId() < 0) {
    executor.setGroupId(group_id);
  }
  auto outputs = executor.runFusion(args, launch_params, compile_params);

  return outputs;
}

// passing args by value because we will be modify this
void FusionKernelRuntime::compileFusionParallel(KernelArgumentHolder args) {
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

  std::atomic<bool> detect_exception_in_thread_pool{false};
  std::string thread_pool_error_message;
  std::mutex thread_pool_error_message_mutex;
  for (int64_t run_order_id = 0; run_order_id < num_groups; ++run_order_id) {
    auto group_to_run = runtime_workspace_.group_run_order.at(run_order_id);

    // TODO: index mode should be updated per segmented kernel
    // Prepare input vector
    KernelArgumentHolder group_runtime_inputs;
    group_runtime_inputs.setDeviceIndex(args.getDeviceIndex());
    if (group_cache_id.has_value()) {
      group_runtime_inputs.setCacheId(group_cache_id.value());
    }
    for (auto input : group_to_run->inputs()) {
      group_runtime_inputs.push(*args_manager.checkTensorMap(input));
    }

    if (num_groups == 1 || isOptionDisabled(DisableOption::ParallelCompile)) {
      FUSER_PERF_SCOPE("FusionKernelRuntime::compileFusionParallel");
      c10::cuda::CUDAGuard dg(args.getDeviceIndex());
      c10::Device device(c10::DeviceType::CUDA, args.getDeviceIndex());
      compileKernel(group_runtime_inputs, group_to_run);
    } else {
      // launch compileKernel thread here
      getThreadPool()->run([this,
                            args,
                            group_runtime_inputs,
                            group_to_run,
                            &detect_exception_in_thread_pool,
                            &thread_pool_error_message,
                            &thread_pool_error_message_mutex]() {
        FUSER_PERF_SCOPE("FusionKernelRuntime::compileFusionParallel");
        try {
          c10::cuda::CUDAGuard dg(args.getDeviceIndex());
          c10::Device device(c10::DeviceType::CUDA, args.getDeviceIndex());
          compileKernel(group_runtime_inputs, group_to_run);
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
    num_live_args_after_segment_runs_.push_back((int64_t)args.size());
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
  if (isProfilerEnabled()) {
    FusionProfiler::stopCompile();
  }
}

void FusionKernelRuntime::compileKernel(
    const KernelArgumentHolder& args,
    SegmentedGroup* sg) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::compileKernel");
  auto group_id = sg->groupId();
  auto heuristic_params = schedulers().at(group_id).get();

  // Check that the heuristics are matched, in the case of segmented fusion
  NVF_ERROR(!sg || heuristic_params->scheduler_type == sg->schedulerType());
  NVF_ERROR(!executors_.at(group_id).isCompiled());

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
  executors_.at(group_id).compileFusion(
      fusion_to_run.get(),
      args,
      heuristic_params->lparams,
      heuristic_params->cparams,
      heuristic_params->scheduler_type,
      fusion_id_,
      concrete_id_,
      runtime_id_,
      group_id);
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

std::vector<at::Tensor> FusionKernelRuntime::runWithInputs(
    KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::runWithInputs");

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
  std::vector<at::Tensor> fusion_outputs;
  fusion_outputs.reserve(segmented_fusion_->outputs().size());
  for (Val* output : segmented_fusion_->outputs()) {
    NVF_ERROR(
        tensor_map.count(output),
        "Segmented fusion output ",
        output->toString(),
        " does not exist in `tensor_map`.");
    const PolymorphicValue* runtime_output = tensor_map.at(output);
    fusion_outputs.push_back(runtime_output->as<at::Tensor>());
  }
  return fusion_outputs;
}

std::unordered_map<Val*, const PolymorphicValue*> FusionKernelRuntime::
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
    KernelArgumentHolder group_runtime_inputs;
    group_runtime_inputs.setDeviceIndex(args.getDeviceIndex());
    if (group_cache_id.has_value()) {
      group_runtime_inputs.setCacheId(group_cache_id.value());
    }
    for (auto input : group_to_run->inputs()) {
      group_runtime_inputs.push(*args_manager.checkTensorMap(input));
    }

    // TODO: currently we are still outputing PyTorch tensors, instead of
    // something abstract. This is quite unsatisfying.

    // Run graph segment
    std::vector<at::Tensor> group_runtime_outputs =
        runKernelWithInput(group_runtime_inputs, group_to_run);
    args_manager.updateWithSegmentOutputs(
        group_to_run->outputs(), group_runtime_outputs, run_order_id);
    num_live_args_after_segment_runs_.push_back((int64_t)args.size());
  }

  if (isProfilerEnabled()) {
    int64_t input_bytes = 0;
    for (auto inp : fusionSegments()->inputs()) {
      if (dynamic_cast<TensorView*>(inp)) {
        auto aten_ten = args_manager.checkTensorMap(inp);
        input_bytes +=
            static_cast<int64_t>(aten_ten->as<at::Tensor>().storage().nbytes());
      }
    }
    FusionProfiler::inputBytesAccessed(input_bytes);
    int64_t output_bytes = 0;
    for (auto outp : fusionSegments()->outputs()) {
      if (dynamic_cast<TensorView*>(outp)) {
        auto aten_ten = args_manager.checkTensorMap(outp);
        output_bytes +=
            static_cast<int64_t>(aten_ten->as<at::Tensor>().storage().nbytes());
      }
    }
    FusionProfiler::outputBytesAccessed(output_bytes);
  }

  return args_manager.getTensorMap();
}

const std::vector<std::unique_ptr<HeuristicParams>>& FusionKernelRuntime::
    schedulers() const {
  return heuristics_->heuristicsList();
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

  // We make a mutable copy of args so that we can use it in an ArgumentManager
  KernelArgumentHolder mutable_args(args);
  ArgumentManager args_manager(
      mutable_args, runtime_workspace_, segmented_fusion_->inputs());
  // Follow group run order
  for (int64_t group_id : c10::irange(num_groups)) {
    auto group_to_run = runtime_workspace_.group_run_order.at(group_id);

    // Create fusion for this segmented group
    Fusion* fusion_to_run = group_to_run->getFusion();
    NVF_ERROR(fusion_to_run != nullptr);
    FusionGuard fg(fusion_to_run);

    // Get input arguments for SchedulerRuntimeInfo
    KernelArgumentHolder group_runtime_inputs;
    for (auto input : group_to_run->inputs()) {
      group_runtime_inputs.push(*args_manager.checkTensorMap(input));
    }

    // Create PrecomputedValues for fusion segment
    std::unique_ptr<PrecomputedValues> evaluator_precomputed_values;
    {
      FUSER_PERF_SCOPE(
          "FusionKernelRuntime::getMaybeHeuristicsFor::PrecomputedValues");
      evaluator_precomputed_values =
          std::make_unique<PrecomputedValues>(fusion_to_run);
      evaluator_precomputed_values->bindInputs(group_runtime_inputs);
      // TODO Remove binding the original fusion inputs when creating heuristics
      // for fusion segment.
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

} // namespace nvfuser

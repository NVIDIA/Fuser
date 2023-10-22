// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <kernel_cache.h>

#include <debug.h>
#include <driver_api.h>
#include <dynamic_transform.h>
#include <executor_params.h>
#include <executor_utils.h>
#include <fusion_profiler.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <optimization/pre_segmenter.h>
#include <options.h>
#include <scheduler/debug_utils.h>
#include <scheduler/registry.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <utils.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>

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
  // T is assumed to be either std::vector<at::Tensro> or KernelArgumentHolder
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
      for (auto group_id : c10::irange(1l, num_groups)) {
        auto group_to_run = group_run_order.at(group_id);
        // set/update life of vals in inputs of this group
        for (auto val : group_to_run->inputs()) {
          // skip fusion inputs and outputs, they may be used by other fusions
          // or code
          if (!isFusionInputOrOutput(val)) {
            last_used_segment_map[val] = group_id;
          }
        }
        // set/update life of vals in outputs of this group
        // skip the last group since its outputs are always the global outputs
        if (group_id < num_groups - 1) {
          for (auto val : group_to_run->outputs()) {
            // skip fusion inputs and outputs, they may be used by other fusions
            // or code
            if (!isFusionInputOrOutput(val)) {
              last_used_segment_map[val] = group_id;
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
  void deleteUnusedArgs(int64_t group_id) {
    // erase args corresponding to vals lastly used in this segment
    if (group_id >= 1 && vals_last_used_at_segment_.count(group_id)) {
      for (auto val : vals_last_used_at_segment_[group_id]) {
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

    // Trivial forwarding outputs an empty tensor to save bandwidth. We skip
    // updating the tensor_map because we want all future use of inputs on
    // the original tensor input. See note [Trivial Forwarding]
    for (const size_t group_out_i : c10::irange(group_outputs.size())) {
      if (!group_outputs[group_out_i]->isFusionInput()) {
        if constexpr (std::is_pointer_v<
                          decltype(group_runtime_outputs[group_out_i])>) {
          fusion_args_.push(*group_runtime_outputs[group_out_i]);
        } else {
          fusion_args_.push(group_runtime_outputs[group_out_i]);
        }
        tensor_map_.emplace(group_outputs[group_out_i], fusion_args_.back());
      }
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
          NVF_ERROR(
              false,
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

FusionExecutorCache::FusionExecutorCache(std::unique_ptr<Fusion> fusion)
    : fusion_(std::move(fusion)) {}

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
    const at::ArrayRef<c10::IValue>& inputs,
    std::optional<PrimDataType> forced_index_type,
    std::optional<int8_t> selected_device) {
  FUSER_PERF_SCOPE("FusionExecutorCache::runFusionWithInputs");

  FUSION_PROFILER_START;

  // Permute input tensor for kernel execution.
  // See Part_1 in Note [ Channels-Last support in nvfuser ]
  at::ArrayRef<c10::IValue> perm_inputs = inputs;
  const auto& to_be_permuted_inputs = fusion_->getPermutationInputMap();
  std::vector<c10::IValue> inputs_vec;
  if (!to_be_permuted_inputs.empty()) {
    inputs_vec = inputs.vec();
    for (const auto& pair : to_be_permuted_inputs) {
      auto v = inputs_vec[pair.first];
      NVF_CHECK(
          v.isTensor(), "input permutation can only be applied at tensor");
      auto tensor = v.toTensor();
      inputs_vec[pair.first] = tensor.permute(pair.second);
    }
    perm_inputs = inputs_vec;
  }

  KernelArgumentHolder args = prepareInputs(perm_inputs, selected_device);
  auto kernel_runtime = getKernelRuntimeFor(args, forced_index_type);

  FUSION_PROFILER_CREATE_SEGMENTS(kernel_runtime->executors().size());

  if (!kernel_runtime->isCompiled()) {
    kernel_runtime->compileFusionParallel(args);
  }

  if (measure_kernel_time_) {
    kernel_runtime->enableKernelTimeMeasurement();
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

  // Permute output tensor returned by kernel execution.
  // See Part_3 in Note [ Permutation support in nvfuser ]
  for (const auto& pair : fusion->getPermutationOutputMap()) {
    if (size_t(pair.first) < outputs.size()) {
      outputs[pair.first] = outputs[pair.first].permute(pair.second);
    }
  }

  // Removing aliased outputs, since those are updated by the Fusion. It is not
  // semantically correct to actually return them as outputs from
  // fusion.
  int offset = 0;
  const auto& indices = fusion->getIndicesOfAliasedOutputs();
  std::set<int> aliased_output_indices(indices.begin(), indices.end());
  for (const auto& v : aliased_output_indices) {
    outputs.erase(outputs.begin() + v - offset);
    offset++;
  }

  FUSION_PROFILER_STOP;
  FUSION_PROFILER_PRINT;

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

FusionKernelRuntime* FusionExecutorCache::getKernelRuntimeFor(
    const KernelArgumentHolder& args,
    std::optional<PrimDataType> forced_index_type) {
  // Check for id hit case
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
            &initial_info, &expr_eval));
    conc_info = cached_conc_info_.back().get();
  }

  // Initialize or fetch vector of FusionKernelRuntime objects associated with
  // each pair of device ID and
  auto& kernel_runtimes =
      kernel_runtimes_
          .try_emplace(std::make_pair(args.getDeviceIndex(), conc_info))
          .first->second;

  // Check for re-use hit case
  //  a kernel runtime is re-usable if all the compiled
  //  kernels have the same heuristic parameters
  std::unique_ptr<FusionHeuristics> new_heuristics;

  FusionKernelRuntime* kernel_runtime = nullptr;

  bool reusing = false;
  // By default, we try to avoid recompiling whenever possible. However, this
  // can lead to suboptimal code if we only check that a compiled kernel is able
  // to run with some inputs, instead of whether it is optimal to do so. The
  // NVFUSER_DISABLE=kernel_reuse option is a coarse tool that just enforces
  // that whenever we encounter a new set of input shapes we segment and compile
  // a new FusionKernelRuntime.
  if (!isOptionDisabled(DisableOption::KernelReuse)) {
    auto reuse_it = std::find_if(
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
    if (reuse_it != kernel_runtimes.end()) {
      kernel_runtime = reuse_it->get();
      kernel_runtime->updateHeuristicsLaunchParams(new_heuristics.get());
      reusing = true;
    }
  }

  if (!reusing) {
    // cache miss, need to re-build an optimized graph for this case

    // Clone fusion_ so that we can safely use an ExpressionEvaluator on it, for
    // the purposes of computing the concretization info.
    auto conc_fusion = std::make_unique<Fusion>(*fusion_);
    if (initial_info.isDynamic()) {
      const auto& conc_initial_info =
          conc_fusion->getManaged<DynamicTransformInitialInfo>("initial_info");
      NVF_ERROR(conc_info);
      conc_info->setInitialInfo(&conc_initial_info);

      if (isDebugDumpEnabled(DebugDumpOption::FusionIrConcretized)) {
        debug() << "Fusion before concretization:" << std::endl;
        conc_fusion->printMath();
      }

      DynamicTransform::concretizeFusion(conc_fusion.get(), conc_info);
      // Initial info is used during concretization and is owned by conc_fusion.
      // After concretization, we stop managing it so that we won't keep cloning
      // it for every subsequent Fusion copy.
      conc_fusion->stopManaging("initial_info");

      if (isDebugDumpEnabled(DebugDumpOption::FusionIrConcretized)) {
        debug() << "Concretized Fusion:" << std::endl;
        conc_fusion->printMath();
      }
    }
    FusionGuard fg(conc_fusion.get());
    kernel_runtimes.emplace_back(std::make_unique<FusionKernelRuntime>(
        std::move(conc_fusion), args, forced_index_type));
    kernel_runtime = kernel_runtimes.back().get();

    if (profiling_) {
      kernel_runtime->profile(true);
    }
  }

  id_to_kernel_runtime_[unique_id] = kernel_runtime;
  return kernel_runtime;
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

  for (auto&& [config, device_runtimes] : kernel_runtimes_) {
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
    auto&& [device_id, dynamic_info] = config;
    fb_kernel_runtimes.push_back(CreateKernelRuntimeStateDirect(
        builder, device_id, (dynamic_info != nullptr), &fb_device_runtimes));
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
      inputs_id_lookup_.serialize(builder),
      &fb_kernel_runtimes,
      &kernel_cache_keys,
      &kernel_cache_values);
}

void FusionExecutorCache::deserialize(
    const serde::FusionExecutorCache* buffer) {
  // See definitions in serde/fusion_cache.fbs for tables
  // FusionExecutorCache and KernelRuntimes

  NVF_ERROR(buffer != nullptr, "serde::FusionExecutorCache is nullptr.");

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

    std::vector<std::unique_ptr<FusionKernelRuntime>> device_runtimes;

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
              &initial_info, &expr_eval));
      conc_info = cached_conc_info_.back().get();
    }

    for (auto runtime : *fb_device_runtimes->runtimes()) {
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
      args.deserialize(runtime->args());

      // 2. Construct new FusionKernelRuntime
      device_runtimes.emplace_back(
          std::make_unique<FusionKernelRuntime>(std::move(conc_fusion), args));

      // 3. For FusionKernelRuntime, we have a separate deserialize function
      // to create the FusionExecutor objects.
      device_runtimes.back()->deserialize(runtime);

      all_runtimes.emplace_back(device_runtimes.back().get());
    }

    kernel_runtimes_.emplace(
        std::make_pair(fb_device_runtimes->device_id(), conc_info),
        std::move(device_runtimes));
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
    std::optional<PrimDataType> forced_index_type) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::FusionKernelRuntime");

  NVF_ERROR(
      !fusion->hasDynamicTransform(),
      "Fusion must be concretized before constructing FusionKernelRuntime");

  // Store metadata copy of arguments for serialization
  std::transform(
      args.cbegin(),
      args.cend(),
      args_metadata_.getBackInserter(),
      convertMetadataArg);

  optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
      fusion.get());

  if (isDebugDumpEnabled(DebugDumpOption::FusionIrPreseg)) {
    debug() << "Fusion IR after pre-segmenter optimization passes:"
            << std::endl;
    fusion->printMath();
  }

  all_tvs_ = ir_utils::allTvs(fusion.get());

  // Run segmentation on the copied fusion
  SchedulerRuntimeInfo runtime_info(
      fusion.get(), args, nullptr, all_tvs_, forced_index_type);

  // Initialize the evaluator simplifer
  precomputed_values_ = std::make_unique<PrecomputedValues>(fusion.get());

  segmented_fusion_ =
      SegmentCandidateFinder::segment(std::move(fusion), args, runtime_info);

  heuristics_ = segmented_fusion_->makeInitialHeuristics(args, runtime_info);

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

flatbuffers::Offset<serde::FusionKernelRuntime> FusionKernelRuntime::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See table definition for FusionKernelRuntime in serde/fusion_cache.fbs

  // 1. Serialize FusionExecutor objects
  std::vector<flatbuffers::Offset<serde::FusionExecutor>> executors_fb;
  executors_fb.reserve(executors_.size());
  for (auto& executor : executors_) {
    executors_fb.push_back(executor.serialize(builder));
  }

  return serde::CreateFusionKernelRuntimeDirect(
      builder, args_metadata_.serialize(builder), &executors_fb);
}

void FusionKernelRuntime::deserialize(
    const serde::FusionKernelRuntime* buffer) {
  // See table definition in FusionKernelRuntime in serde/fusion_cache.fbs

  NVF_ERROR(buffer != nullptr, "serde::FusionKernelRuntime is nullptr.");
  NVF_ERROR(runtime_workspace_.group_run_order.size() == executors_.size());

  // 1. Deserialize FusionExecutor objects
  for (auto idx : c10::irange(buffer->executors()->size())) {
    auto sg = runtime_workspace_.group_run_order.at(idx);

    // Create and schedule Fusion for this SegmentedGroup
    auto group_id = sg->groupId();
    auto scheduler_entry = schedulers().at(group_id).get();
    NVF_ERROR(
        !sg || scheduler_entry->heuristic() == sg->heuristic(),
        "Heuristics do not match.");
    std::unique_ptr<Fusion> fusion_to_run = segmented_fusion_->makeFusion(sg);
    FusionGuard fg(fusion_to_run.get());
    scheduler_entry->schedule(fusion_to_run.get());

    executors_.at(group_id).deserialize(
        buffer->executors()->Get(group_id),
        fusion_to_run.get(),
        scheduler_entry->params()->cparams);
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
  auto scheduler_entry = schedulers().at(group_id).get();
  auto& executor = executors_.at(group_id);

  if (profiling_) {
    most_recent_executor_log_.fusion_executor = &executor;
    most_recent_executor_log_.params = scheduler_entry->params()->clone();
  }

  if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose) ||
      measure_kernel_time_) {
    executor.setMeasureKernelTimeFlag(true);
  }

  SEGMENT_PROFILER_INPUT_BYTES_ACCESSED(group_id,
      ([&args, &executor]() { return executor.inputBytesProcessed(args);}));
  SEGMENT_PROFILER_START_KERNEL(args.getDeviceIndex(), group_id)
  auto outputs = executor.runFusion(args, launch_params, compile_params);
  SEGMENT_PROFILER_STOP_KERNEL(group_id);
  SEGMENT_PROFILER_OUTPUT_BYTES_ACCESSED(group_id,
      ([&outputs, &executor]() { return executor.outputBytesProcessed(outputs);}));
  
  // Accumulate the kernel time of each segment
  kernel_time_ms_ += executor.kernelTimeMs();

  // Print relevant information all at once for easy debuging of perf
  if (isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    debug() << "\nRun kernel:\n";
    if (sg) {
      segmented_fusion_->makeFusion(sg)->printMath();
    } else {
      segmented_fusion_->completeFusion()->printMath();
    }
    debug() << "With inputs:\n";
    for (auto i : c10::irange(args.size())) {
      debug() << "  " << args[i] << std::endl;
    }
    debug() << "Compiler log: " << executor.compiledKernel().compile_log
            << "\n";
    debug() << scheduler_entry->params()->toString() << "\n";
    debug() << "With arguments: " << executor.lastLaunchParams().toString();
    debug() << executor.kernelName() << " " << executor.bytesProcessed()
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
        const auto extent = root_dom[dim]->getMaybeExpandedExtent();
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
    NVF_ERROR(
        one_ran,
        "Couldn't run all groups, something must have gone wrong in segmentation.");
  }
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
  SEGMENT_PROFILER_START_PARALLEL_COMPILE(num_groups);
  for (int64_t group_id = 0; group_id < num_groups; ++group_id) {
    auto group_to_run = runtime_workspace_.group_run_order.at(group_id);

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
      SEGMENT_PROFILER_START_COMPILE(args.getDeviceIndex(), group_id);
      c10::cuda::CUDAGuard dg(args.getDeviceIndex());
      c10::Device device(c10::DeviceType::CUDA, args.getDeviceIndex());
      compileKernel(group_runtime_inputs, group_to_run);
      SEGMENT_PROFILER_STOP_COMPILE(group_id);
    } else {
      // launch compileKernel thread here
      getThreadPool()->run([this, args, group_runtime_inputs, group_to_run]() {
        FUSER_PERF_SCOPE("FusionKernelRuntime::compileFusionParallel");
        c10::cuda::CUDAGuard dg(args.getDeviceIndex());
        c10::Device device(c10::DeviceType::CUDA, args.getDeviceIndex());
        compileKernel(group_runtime_inputs, group_to_run);
      });
    }

    auto fusion_to_run = segmented_fusion_->makeFusion(group_to_run);
    auto group_runtime_outputs =
        executors_[group_to_run->groupId()].inferOutputSizes(
            fusion_to_run.get(), group_runtime_inputs);

    // map output args to tensor map
    args_manager.updateWithSegmentOutputs(
        group_to_run->outputs(), group_runtime_outputs, group_id);
    num_live_args_after_segment_runs_.push_back((int64_t)args.size());
  }

  if (num_groups != 1 && !isOptionDisabled(DisableOption::ParallelCompile)) {
    // wait until all segments finish compiling
    getThreadPool()->waitWorkComplete();
  }
  SEGMENT_PROFILER_STOP_PARALLEL_COMPILE(num_groups);
}

void FusionKernelRuntime::compileKernel(
    const KernelArgumentHolder& args,
    SegmentedGroup* sg) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::compileKernel");
  auto group_id = sg->groupId();
  auto scheduler_entry = schedulers().at(group_id).get();

  // Check that the heuristics are matched, in the case of segmented fusion
  NVF_ERROR(!sg || scheduler_entry->heuristic() == sg->heuristic());
  NVF_ERROR(!executors_.at(group_id).isCompiled());

  // Running a segment group as a single kernel,
  // make a fusion to run from segmented fusion
  auto fusion_to_run = segmented_fusion_->makeFusion(sg);
  FusionGuard fg(fusion_to_run.get());
  scheduler_entry->schedule(fusion_to_run.get());
  NVF_ERROR(
      scheduler_entry->params()->cparams.index_type.has_value(),
      "Kernel index type is not defined.");
  SEGMENT_PROFILER_START_COMPILE(args.getDeviceIndex(), group_id);
  executors_.at(group_id).compileFusion(
      fusion_to_run.get(),
      args,
      scheduler_entry->params()->lparams,
      scheduler_entry->params()->cparams);
  SEGMENT_PROFILER_STOP_COMPILE(group_id);
}

std::pair<LaunchParams, CompileParams> FusionKernelRuntime::getKernelConfig(
    const KernelArgumentHolder& args,
    SegmentedGroup* sg) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::getKernelConfig");
  auto group_id = sg->groupId();
  auto scheduler_entry = schedulers().at(group_id).get();

  // Check that the heuristics are matched, in the case of segmented fusion
  NVF_ERROR(!sg || scheduler_entry->heuristic() == sg->heuristic());
  NVF_ERROR(executors_.at(group_id).isCompiled());

  return std::make_pair(
      scheduler_entry->params()->lparams, scheduler_entry->params()->cparams);
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
      NVF_ERROR(iter->second->is<at::Tensor>());
      fusion_outputs.push_back(iter->second->as<at::Tensor>());
    } else {
      bool empty_type_check = output->getDataType().has_value() &&
          output->getDataType().value() == DataType::Float;

      // Only support two cases of empty tensor here, since this is hot path.
      auto out_tv = output->as<TensorView>();

      // TODO: should be only one of the two once the "empty"
      //  definition has been unified throughout the ops.
      bool empty_tensor_check = out_tv->isZeroDim() || out_tv->isEmptyTensor();

      // This is the check for an empty tensor;
      NVF_ERROR(
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

std::unordered_map<Val*, const PolymorphicValue*> FusionKernelRuntime::
    runSegmentsWithInputs(KernelArgumentHolder& args) {
  NVF_ERROR(
      args.size() == segmented_fusion_->inputs().size(),
      "Inputs were not set up correctly, received ",
      args.size(),
      " inputs but expected ",
      segmented_fusion_->inputs().size());

  bool compute_overall_bw =
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose);

  int64_t total_bytes_processed = 0;

  ArgumentManager args_manager(
      args, runtime_workspace_, segmented_fusion_->inputs());

  // group should share cache id.
  auto group_cache_id = args.getCacheId();
  const int64_t num_groups = (int64_t)runtime_workspace_.group_run_order.size();
  num_live_args_after_segment_runs_.reserve(num_groups);
  kernel_time_ms_ = 0;
  for (auto group_id : c10::irange(num_groups)) {
    // TODO: index mode should be updated per segmented kernel
    // Prepare input vector
    auto group_to_run = runtime_workspace_.group_run_order.at(group_id);
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
        group_to_run->outputs(), group_runtime_outputs, group_id);
    num_live_args_after_segment_runs_.push_back((int64_t)args.size());

    if (compute_overall_bw) {
      const auto& executor = executors_.at(group_id);
      for (auto bytes : executor.bytesInputsProcessed()) {
        total_bytes_processed += bytes;
      }
      for (auto bytes : executor.bytesOutputsProcessed()) {
        total_bytes_processed += bytes;
      }
    }
  }

  FUSION_PROFILER_INPUT_BYTES_ACCESSED(([&]() {
    size_t input_bytes = 0;
    for (auto inp : fusionSegments()->inputs()) {
      if (auto tv = dynamic_cast<TensorView*>(inp)) {
        auto aten_ten = args_manager.checkTensorMap(inp);
        input_bytes += aten_ten->as<at::Tensor>().numel() *
            dataTypeSize(tv->dtype());
      }
    }
    return input_bytes;
  }));
  FUSION_PROFILER_OUTPUT_BYTES_ACCESSED(([&]() {
    size_t output_bytes = 0;
    for (auto outp : fusionSegments()->outputs()) {
      if (auto tv = dynamic_cast<TensorView*>(outp)) {
        auto aten_ten = args_manager.checkTensorMap(outp);
        output_bytes += aten_ten->as<at::Tensor>().numel() *
            dataTypeSize(tv->dtype());
      }
    }
    return output_bytes;
  }));

  if (compute_overall_bw) {
    // Get peak bandwidth for device
    int clock = 0, width = 0;
    std::string gpuname;
    gpuname.reserve(100);
    NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
        &clock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, args.getDeviceIndex()));
    NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
        &width,
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        args.getDeviceIndex()));
    NVFUSER_CUDA_SAFE_CALL(
        cuDeviceGetName(gpuname.data(), 100, args.getDeviceIndex()));
    // Peak bandwidth calculation:
    // Bus width is given in bits, so dividing by 8 converts to bytes.
    // Clock is given in kHz. 1 GB = 1e9 bytes (don't report GiB = 1024^3 bytes)
    // A factor of 2 is multiplied to account for double data rate (DDR):
    // (clock in kHz * width in bits) * (1000 Hz / kHz) * (1 GB / 8e9 bits) * 2
    // factor = 2.5e-7
    double peak_bw = 2.5e-7 * (double)clock * (double)width;

    int64_t total_io_bytes_processed = 0;
    for (auto inp : fusionSegments()->inputs()) {
      if (auto tv = dynamic_cast<TensorView*>(inp)) {
        auto aten_ten = args_manager.checkTensorMap(inp);
        total_io_bytes_processed +=
            (int64_t)aten_ten->as<at::Tensor>().numel() *
            dataTypeSize(tv->dtype());
      }
    }
    for (auto outp : fusionSegments()->outputs()) {
      if (auto tv = dynamic_cast<TensorView*>(outp)) {
        auto aten_ten = args_manager.checkTensorMap(outp);
        total_io_bytes_processed +=
            (int64_t)aten_ten->as<at::Tensor>().numel() *
            dataTypeSize(tv->dtype());
      }
    }

    // Effective bw in GB/s
    double eff_bw = 1e-6 * (double)total_io_bytes_processed / kernel_time_ms_;

    double percent_peak = eff_bw / peak_bw * 100;

    auto formatBytes = [](double bytes) {
      std::stringstream ss;
      if (bytes < 1e3) {
        ss << bytes << " B";
        return ss.str();
      }
      ss << std::setprecision(2);
      if (bytes >= 1e12) {
        ss << (bytes / 1e12) << " TB";
      } else if (bytes >= 1e9) {
        ss << (bytes / 1e9) << " GB";
      } else if (bytes >= 1e6) {
        ss << (bytes / 1e6) << " MB";
      } else if (bytes >= 1e3) {
        ss << (bytes / 1e3) << " kB";
      }
      return ss.str();
    };

    debug() << "Total bytes processed: "
            << formatBytes((double)total_bytes_processed) << std::endl;
    debug() << "Bytes that were complete fusion inputs or outputs: "
            << formatBytes((double)total_io_bytes_processed) << " ("
            << ((double)total_io_bytes_processed /
                (double)total_bytes_processed * 100.0)
            << "% of total)" << std::endl;

    debug() << "Total CUDA kernel time (" << num_groups
            << " kernels): " << kernel_time_ms_ << " ms" << std::endl;
    debug() << "Theoretical peak bandwidth (" << gpuname << "): " << peak_bw
            << " GB/s" << std::endl;
    debug()
        << "Complete fusion effective bandwidth (counts CUDA kernel time only): "
        << eff_bw << " GB/s (";
    debug() << std::setprecision(2) << percent_peak << "\% of theoretical peak)"
            << std::endl;
  }

  return args_manager.getTensorMap();
}

const std::vector<FusionKernelRuntime::SchedulerEntryPtr>& FusionKernelRuntime::
    schedulers() const {
  return heuristics_->heuristicsList();
}

void FusionKernelRuntime::updateHeuristicsLaunchParams(
    FusionHeuristics* update_heuristics) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::updateHeuristicsLaunchParams");
  auto scheduler_list_length = heuristics_->heuristicsList().size();
  NVF_ERROR(
      update_heuristics->heuristicsList().size() == scheduler_list_length);
  for (const auto i : c10::irange(scheduler_list_length)) {
    auto& schedulerPtr = heuristics_->heuristicsList()[i];
    schedulerPtr->updateLaunchConstraint(
        update_heuristics->heuristicsList()[i]->params()->lparams);
  }
}

std::optional<FusionKernelRuntime::HeuristicsPtr> FusionKernelRuntime::
    getMaybeHeuristicsFor(
        const KernelArgumentHolder& args,
        std::optional<PrimDataType> forced_index_type) {
  FUSER_PERF_SCOPE("FusionKernelRuntime::getMaybeHeuristicsFor");
  auto complete_fusion = segmented_fusion_->completeFusion();
  precomputed_values_->bindInputs(args);
  precomputed_values_->evaluate();
  SchedulerRuntimeInfo runtime_info(
      complete_fusion,
      args,
      precomputed_values_.get(),
      all_tvs_,
      forced_index_type);

  std::optional<FusionKernelRuntime::HeuristicsPtr> ret;
  ret = std::make_unique<FusionHeuristics>();
  size_t total_groups = segmented_fusion_->groups().size();
  for (const auto group_index : c10::irange(total_groups)) {
    auto group = segmented_fusion_->groups()[group_index];

    auto maybe_scheduler_entry = group->getMaybeSchedulerEntry(runtime_info);
    if (!maybe_scheduler_entry.has_value()) {
      return std::nullopt;
    }
    auto scheduler_entry = std::move(maybe_scheduler_entry.value());
    if (!scheduler_entry->sameAs(
            heuristics_->heuristicsList()[group_index].get())) {
      return std::nullopt;
    }
    ret.value()->emplaceBack(std::move(scheduler_entry));
  }

  return ret;
}

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <kernel_cache_utils.h>

#include <fusion_executor/executor_kernel_arg.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <polymorphic_value.h>

#include <unordered_set>

namespace nvfuser {

namespace {

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
} // namespace

ArgumentManager::ArgumentManager(
    KernelArgumentHolder& args,
    const RuntimeWorkSpace& runtime_workspace,
    const std::vector<Val*>& fusion_inputs)
    : fusion_args_(args) {
  // map from val to args
  mapFusionInputsToArgs(
      fusion_inputs, runtime_workspace.group_extent_binding_order);
  setLastUsedSegmentID(runtime_workspace.group_run_order);
}

const std::unordered_map<Val*, const PolymorphicValue*>& ArgumentManager::
    getTensorMap() {
  return tensor_map_;
}
const PolymorphicValue* ArgumentManager::checkTensorMap(Val* v) {
  return tensor_map_.at(v);
}

template <typename T>
void ArgumentManager::addOutputsToArgsAndTensorMap(
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

template <typename T>
void ArgumentManager::updateWithSegmentOutputs(
    const std::vector<Val*>& group_outputs,
    const T& group_runtime_outputs,
    const int64_t group_id) {
  addOutputsToArgsAndTensorMap(group_outputs, group_runtime_outputs);
  deleteUnusedArgs(group_id);
}

template void ArgumentManager::addOutputsToArgsAndTensorMap<
    std::vector<at::Tensor>>(
    const std::vector<Val*>& group_outputs,
    const std::vector<at::Tensor>& group_runtime_outputs);

template void ArgumentManager::updateWithSegmentOutputs<
    std::vector<at::Tensor>>(
    const std::vector<Val*>&,
    const std::vector<at::Tensor>&,
    const int64_t);

template void ArgumentManager::addOutputsToArgsAndTensorMap<
    KernelArgumentHolder>(
    const std::vector<Val*>& group_outputs,
    const KernelArgumentHolder& group_runtime_outputs);

template void ArgumentManager::updateWithSegmentOutputs<KernelArgumentHolder>(
    const std::vector<Val*>&,
    const KernelArgumentHolder&,
    const int64_t);

void ArgumentManager::mapFusionInputsToArgs(
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

void ArgumentManager::setLastUsedSegmentID(
    const std::vector<SegmentedGroup*>& group_run_order) {
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
        if (!val->isFusionInput() && !val->isFusionOutput()) {
          last_used_segment_map[val] = run_order_id;
        }
      }
      // set/update life of vals in outputs of this group
      // skip the last group since its outputs are always the global outputs
      if (run_order_id < num_groups - 1) {
        for (auto val : group_to_run->outputs()) {
          // skip fusion inputs and outputs, they may be used by other fusions
          // or code
          if (!val->isFusionInput() && !val->isFusionOutput()) {
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

void ArgumentManager::deleteUnusedArgs(int64_t run_order_id) {
  // erase args corresponding to vals lastly used in this segment
  if (run_order_id >= 1 && vals_last_used_at_segment_.count(run_order_id)) {
    for (auto val : vals_last_used_at_segment_[run_order_id]) {
      fusion_args_.erase(tensor_map_.at(val));
      tensor_map_.erase(val);
    }
  }
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

} // namespace nvfuser

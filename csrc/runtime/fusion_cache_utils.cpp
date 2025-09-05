// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <runtime/fusion_cache_utils.h>

#include <tensor_metadata.h>
#include <unordered_set>

#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <polymorphic_value.h>
#include <runtime/executor_kernel_arg.h>

namespace nvfuser {

namespace {

// Copy bytes of value to back of buffer. This is templated in order to avoid
// implicit cast such as int64_t -> size_t that might lose information.
template <typename T>
void encodeBuffer(T value, std::string& buffer) {
  const char* v = reinterpret_cast<char*>(&value);
  for (const auto i : arange(sizeof(T))) {
    (void)i; // Suppress unused variable warning
    buffer.push_back(*(v++));
  }
}
} // namespace

ArgumentManager::ArgumentManager(
    const KernelArgumentHolder& args,
    const RuntimeWorkSpace& runtime_workspace,
    const std::vector<Val*>& fusion_inputs) {
  // map from val to args
  mapFusionInputsToArgs(
      fusion_inputs, args, runtime_workspace.group_extent_binding_order);
  setLastUsedSegmentID(runtime_workspace.group_run_order);
}

const PolymorphicValue& ArgumentManager::checkTensorMap(Val* v) const {
  return tensor_map_.at(v);
}

KernelArgumentHolder ArgumentManager::translateValsToArgs(
    const std::vector<Val*>& vals) const {
  std::vector<PolymorphicValue> arg_values;
  arg_values.reserve(vals.size());

  for (auto val : vals) {
    auto it = tensor_map_.find(val);
    NVF_ERROR(
        it != tensor_map_.end(),
        "Could not find value ",
        val->toString(),
        " in tensor map");
    arg_values.push_back(it->second);
  }

  KernelArgumentHolder holder;
  for (auto arg : arg_values) {
    holder.push(std::move(arg));
  }
  return holder;
}


std::vector<std::optional<bool>> _computeContiguity(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  NVF_CHECK(
      sizes.size() == strides.size(),
      "compute_contiguity: Sizes and strides must have the same number of "
      "dimensions");
  // Not a broadcast means neither the stride == 0 (size can be non-zero)
  // or the size == 1 that each can indicate a broadcast
  auto not_broadcast = [&](auto i) { return strides[i] != 0 && sizes[i] != 1; };
  // Contiguity defaults to vector of all None's
  std::vector<std::optional<bool>> contiguity(sizes.size(), std::nullopt);
  if (contiguity.empty()) { // zero-dim tensor
    return contiguity;
  }
  int64_t last = (int64_t)sizes.size() - 1; // inner most dimension
  // Contiguity normallly is determined by the current dimension and one
  // dimension to the right.  The innermost dimension, that is not broadcasted,
  // does not have any dimension to it's right and needs to be specially marked
  // contiguous.
  for (; last >= 0; --last) {
    if (not_broadcast(last)) {
      contiguity[last] = (strides.at(last) == 1);
      break;
    }
  }
  // Dimensions are marked contiguous by inspecting the current dimension and
  // one to the right towards the inner dimension while skipping over broadcast
  // dimensions.
  for (int64_t i = 0; i < last;) {
    if (not_broadcast(i)) {
      auto l = i++;
      for (; i <= last; i++) {
        if (not_broadcast(i)) {
          break;
        }
      }
      contiguity[l] = (strides[l] == strides[i] * sizes[i]);
    } else {
      i++;
    }
  }
  return contiguity;
}

void ArgumentManager::updateWithSegmentOutputs(
    const std::vector<Val*>& group_outputs,
    const KernelArgumentHolder& group_runtime_outputs,
    const int64_t group_id) {
  // Insert graph segment output to tensor map
  NVF_ERROR_EQ(
      std::ssize(group_outputs),
      group_runtime_outputs.size(),
      "Output size does not match.");
  for (const size_t group_out_i : arange(group_outputs.size())) {
    tensor_map_.emplace(
        group_outputs[group_out_i], group_runtime_outputs[group_out_i]);
    auto tv = dynamic_cast<TensorView*>(group_outputs[group_out_i]);
    if (tv) {
      tv->printTransforms();
      const at::Tensor& tensor = group_runtime_outputs[group_out_i].as<at::Tensor>();
      // const std::vector<int64_t> sizes = tensor.sizes().vec();
      // const std::vector<int64_t> strides = tensor.strides().vec();
      const auto [sizes, strides] = inferAndValidateAllocationSizesAndStrides(tensor, tv, ExpressionEvaluator());
      std::vector<std::optional<bool>> contiguity = _computeContiguity(sizes, strides);
      std::cout << "sizes: " << sizes << std::endl;
      std::cout << "strides: " << strides << std::endl;
      std::cout << "contiguity: " << contiguity << std::endl;
      tv->domain()->setContiguity(contiguity);
    }
  }

  // Delete args corresponding to vals lastly used in this segment
  if (group_id >= 1 && vals_last_used_at_segment_.count(group_id)) {
    for (auto val : vals_last_used_at_segment_[group_id]) {
      tensor_map_.erase(val);
    }
  }
}

void ArgumentManager::mapFusionInputsToArgs(
    const std::vector<Val*>& fusion_inputs,
    const KernelArgumentHolder& args,
    const std::vector<Val*>& group_extent_binding_order) {
  int extent_index = 0;
  auto original_args_size = args.size();
  // Bind args in the tensor_map
  for (const auto i : arange(original_args_size)) {
    tensor_map_.emplace(fusion_inputs[i], args[i]);
    // Bind tensorview inputs values in case some segmented group
    //  needs it down the road.
    if (args[i].is<at::Tensor>()) {
      auto rank = args[i].as<at::Tensor>().dim();
      for (const auto dim : arange(rank)) {
        tensor_map_.emplace(
            group_extent_binding_order[extent_index++],
            args[i].as<at::Tensor>().size(dim));
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
    for (auto run_order_id : arange(1l, num_groups)) {
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

  for (auto idx : arange(buffer->encoding_lookup_keys()->size())) {
    auto fb_encoding_lookup_str = buffer->encoding_lookup_keys()->Get(idx);
    auto fb_encoding_entry = buffer->encoding_lookup_values()->Get(idx);

    EncodingEntry entry{
        fb_encoding_entry->id(),
        used_entry_iterators.at(fb_encoding_entry->lru_iter())};
    encoding_lookup_.emplace(fb_encoding_lookup_str->str(), entry);
  }
}

InputsIdLookup::IdLookupReturn InputsIdLookup::lookupId(
    const KernelArgumentHolder& args,
    const std::unordered_set<size_t>& scalar_inputs_to_record) {
  IdLookupReturn ret;

  // lock mutex_ because we are touching encoding_
  std::lock_guard<std::mutex> guard(mutex_);
  encoding_.clear();
  encodeBuffer(args.getDeviceIndex(), encoding_);

  for (const auto i : arange(args.size())) {
    const auto& arg = args[i];
    if (arg.is<at::Tensor>()) {
      const auto& input_tensor = arg.as<at::Tensor>();

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
          SchedulerRuntimeInfo::computeAlignmentSizeBit(
              (size_t)input_tensor.data_ptr()),
          encoding_);
    } else {
      // encode s for scalar;
      encoding_.push_back('s');
      if (scalar_inputs_to_record.find(i) != scalar_inputs_to_record.end()) {
        // Add value of scalars here only if it is one of the scalars
        // provided, as these are used in determining concretization.
        // Note that although most commonly these will be Int or Bool scalars,
        // any DataType might appear via `cast` and `where`, so we handle all
        // cases here.
        if (arg.is<int64_t>()) {
          encodeBuffer(arg.as<int64_t>(), encoding_);
        } else if (arg.is<bool>()) {
          encodeBuffer(arg.as<bool>(), encoding_);
        } else if (arg.is<double>()) {
          encodeBuffer(arg.as<double>(), encoding_);
        } else if (arg.is<std::complex<double>>()) {
          encodeBuffer(arg.as<std::complex<double>>(), encoding_);
        } else {
          NVF_THROW(
              "Unhandled input type when creating input ID. Cannot record ",
              PolymorphicValue_functions::toString(arg));
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

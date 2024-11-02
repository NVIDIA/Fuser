// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <python_frontend/fusion_definition.h>
#include <python_frontend/translation.h>

namespace nvfuser::python_frontend {

int64_t SegmentationState::setupSegmentation(
    Fusion* fusion,
    const std::unordered_map<const Val*, int64_t>& map_value_to_original_fid,
    const at::ArrayRef<c10::IValue>& inputs) {
  // Check state
  NVF_ERROR(fusion != nullptr);
  NVF_ERROR(segment_fusion_ == nullptr);
  NVF_ERROR(segmented_fusion_ == nullptr);
  NVF_ERROR(group_run_order_.empty());
  NVF_ERROR(map_cloned_value_to_fid_.empty());
  NVF_ERROR(cloned_extents_.empty());

  int8_t device = getCommonDeviceCUDA(inputs);
  NVF_CHECK(
      inputs.empty() || device > -1, "Inputs are not all on the same device!");

  // Clone CPP Fusion
  segment_fusion_ = std::make_unique<Fusion>();
  IrCloner original_to_cloned_map = Fusion::copy(fusion, segment_fusion_.get());

  // Get arguments
  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder(inputs, device);

  // Concretize fusion with input arguments. Then, map original symbolic values
  // to new concrete values when building map_cloned_value_to_fid_
  std::unordered_map<Val*, Val*> symbolic_to_concrete_map =
      DynamicTransform::concretizeFusion(segment_fusion_.get(), args);

  // Track mapping from cloned CPP fusion and FusionDefinition indices.
  std::transform(
      map_value_to_original_fid.begin(),
      map_value_to_original_fid.end(),
      std::inserter(map_cloned_value_to_fid_, map_cloned_value_to_fid_.end()),
      [&](const auto& item) {
        const Val* original_value = item.first;
        int64_t fid = item.second;
        Val* cloned_val = original_to_cloned_map.clone(original_value);
        if (symbolic_to_concrete_map.count(cloned_val)) {
          cloned_val = symbolic_to_concrete_map.at(cloned_val);
        }
        return std::make_pair(cloned_val, fid);
      });

  // Track the extents for input TensorViews in cloned CPP Fusion.
  cloned_extents_ = getExtents(segment_fusion_.get());

  // Create runtime infomation
  SchedulerRuntimeInfo runtime_info(
      segment_fusion_.get(),
      args,
      /*precomputed_values=*/nullptr,
      segment_fusion_->allTvs());

  // Run segmentation algorithm
  segmented_fusion_ = SegmentCandidateFinder::segment(
      std::move(segment_fusion_), &args, runtime_info);

  // Get the order for fusion segments
  prepareGroupOrder();

  // Return the number of segments
  return (int64_t)segmented_fusion_->groups().size();
}

std::unordered_map<int64_t, int64_t> SegmentationState::buildSegment(
    FusionDefinition& other,
    int64_t segment_id) {
  NVF_ERROR(
      !other.completed(),
      "Expected an incomplete definition before translation.");
  NVF_ERROR(
      segmented_fusion_ != nullptr,
      "SegmentedFusion is not initialized. Run setupSegmentation first.");
  NVF_ERROR(
      segment_id >= 0 &&
          segment_id < (int64_t)segmented_fusion_->groups().size(),
      "The segment id is not valid");

  // Create new fusion segment
  SegmentedGroup* sg = group_run_order_.at(segment_id);
  NVF_ERROR(sg != nullptr);
  std::pair<IrCloner, std::unique_ptr<Fusion>> cloner_segment_pair =
      segmented_fusion_->makeFusion(sg);
  IrCloner original_to_segment_map = cloner_segment_pair.first;
  std::unique_ptr<Fusion> fusion_segment =
      std::move(cloner_segment_pair.second);

  std::unordered_map<const nvfuser::Val*, size_t>
      map_translated_val_to_segment_fid =
          translate(fusion_segment.get(), &other);

  // Step 1: Get FusionDefinition index for original inputs and outputs.
  // Use std::transform on inputs and outputs
  const std::vector<Val*>& original_inputs = sg->inputs();
  const std::vector<Val*>& original_outputs = sg->outputs();

  std::vector<int64_t> original_fid;
  original_fid.reserve(original_inputs.size() + original_outputs.size());

  std::transform(
      original_inputs.begin(),
      original_inputs.end(),
      std::back_inserter(original_fid),
      [&](Val* v) { return map_cloned_value_to_fid_.at(v); });

  std::transform(
      original_outputs.begin(),
      original_outputs.end(),
      std::back_inserter(original_fid),
      [&](Val* v) { return map_cloned_value_to_fid_.at(v); });

  // Step 2: ir_cloner maps original fusion statements to translated statements.
  // Use std::transform
  std::vector<Val*> segment_inputs_outputs;
  segment_inputs_outputs.reserve(
      original_inputs.size() + original_outputs.size());

  std::transform(
      original_inputs.begin(),
      original_inputs.end(),
      std::back_inserter(segment_inputs_outputs),
      [&](Val* v) { return original_to_segment_map.clone(v); });

  std::transform(
      original_outputs.begin(),
      original_outputs.end(),
      std::back_inserter(segment_inputs_outputs),
      [&](Val* v) { return original_to_segment_map.clone(v); });

  // Step 3: Map translated statements to its FusionDefinition index.
  std::vector<int64_t> segment_fid;
  segment_fid.reserve(segment_inputs_outputs.size());
  std::transform(
      segment_inputs_outputs.begin(),
      segment_inputs_outputs.end(),
      std::back_inserter(segment_fid),
      [&](Val* v) { return map_translated_val_to_segment_fid.at(v); });

  // Step 4: Map original FusionDefinition index to translated Fusion Definition
  // index for inputs and outputs.
  NVF_ERROR(original_fid.size() == segment_fid.size());

  // Create map from original fid to segment fid.
  std::unordered_map<int64_t, int64_t> segment_fid_to_original_fid_map;
  for (size_t idx : c10::irange(original_fid.size())) {
    segment_fid_to_original_fid_map.emplace(
        segment_fid.at(idx), original_fid.at(idx));
  }

  // short-circuit: No extra extents required for python definition
  if (fusion_segment->inputs().size() == other.inputs().size()) {
    return segment_fid_to_original_fid_map;
  }

  // The python definition can require the size of tensor dimensions from
  // original input arguments, which the original segment does not.

  // Step 1a: Create a map from segment to original extents.
  // Step 1a: Create a map from segment fid to segment extents.
  std::unordered_map<Val*, Val*> segment_to_original_extents;
  std::unordered_map<size_t, Val*> segment_fid_to_translated_val;
  for (Val* original_extent : cloned_extents_) {
    Val* segment_extent = original_to_segment_map.clone(original_extent);

    // short-circuit: some extents are not used in segment
    if (map_translated_val_to_segment_fid.count(segment_extent) == 0) {
      continue;
    }

    size_t segment_fid = map_translated_val_to_segment_fid.at(segment_extent);
    segment_to_original_extents.emplace(segment_extent, original_extent);
    segment_fid_to_translated_val.emplace(segment_fid, segment_extent);
  }

  // Step 2: Find the set difference between all segment input fid and known
  // segment fids.
  std::vector<int64_t> missing_segment_fid;
  for (int64_t input_fid : other.inputs()) {
    if (segment_fid_to_original_fid_map.count(input_fid) == 0) {
      missing_segment_fid.push_back(input_fid);
    }
  }

  // Step 3: Get segment Val for missing segment input fids.
  std::vector<Val*> missing_segment_val;
  missing_segment_val.reserve(missing_segment_fid.size());
  std::transform(
      missing_segment_fid.begin(),
      missing_segment_fid.end(),
      std::back_inserter(missing_segment_val),
      [&](int64_t segment_fid) {
        return segment_fid_to_translated_val.at(segment_fid);
      });

  // Step 4: Map segment Val to cloned Val
  std::vector<Val*> missing_cloned_val;
  missing_cloned_val.reserve(missing_segment_val.size());
  std::transform(
      missing_segment_val.begin(),
      missing_segment_val.end(),
      std::back_inserter(missing_cloned_val),
      [&](Val* segment_val) {
        return segment_to_original_extents.at(segment_val);
      });

  // Step 5: Transform cloned Val to original fid.
  std::vector<int64_t> missing_cloned_fid;
  missing_cloned_fid.reserve(missing_cloned_val.size());
  std::transform(
      missing_cloned_val.begin(),
      missing_cloned_val.end(),
      std::back_inserter(missing_cloned_fid),
      [&](Val* original_val) {
        return map_cloned_value_to_fid_.at(original_val);
      });

  // Step 6: Add mapping from segment to original fid.
  for (size_t idx : c10::irange(missing_segment_fid.size())) {
    segment_fid_to_original_fid_map.emplace(
        missing_segment_fid.at(idx), missing_cloned_fid.at(idx));
  }

  return segment_fid_to_original_fid_map;
}

} // namespace nvfuser::python_frontend

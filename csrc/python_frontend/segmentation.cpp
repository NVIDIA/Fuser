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

  // Step 1) Clone preschedFusion CPP Fusion.
  segment_fusion_ = std::make_unique<Fusion>();

  // The IRCloner returned by Fusion::copy acts as map from the original fusion
  // to the cloned fusion.
  IrCloner original_to_cloned_map = Fusion::copy(fusion, segment_fusion_.get());

  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder(inputs, device);

  // Step 2) Concretize fusion with input arguments.
  std::unordered_map<Val*, Val*> symbolic_to_concrete_map =
      DynamicTransform::concretizeFusion(segment_fusion_.get(), args);

  // Step 3) Given the map_value_to_original_fid, the IRCloner returned by
  // Fusion::copy, AND the symbolic_to_concrete map returned by
  // concretization pass, create a mapping from cloned Vals to original fusion
  // state indices.
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

  // Return the number of segments created by segmentation algorithm.
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

  // Step 1) Use segment id to get SegmentedGroup from group_run_order_.
  SegmentedGroup* sg = group_run_order_.at(segment_id);
  NVF_ERROR(sg != nullptr);

  // Step 2) Create CPP Fusion for SegmentedGroup. The IrCloner acts as a map
  // from fusion segment to the original fusion.
  std::pair<IrCloner, std::unique_ptr<Fusion>> cloner_segment_pair =
      segmented_fusion_->makeFusion(sg);
  IrCloner cloned_to_segment_map = cloner_segment_pair.first;
  std::unique_ptr<Fusion> fusion_segment =
      std::move(cloner_segment_pair.second);

  // Step 3) Translate CPP Fusion to Python FusionDefinition
  std::unordered_map<const nvfuser::Val*, size_t>
      map_translated_val_to_segment_fid =
          translate(fusion_segment.get(), &other);

  // Step 4) Create map from segment fusion indices to original fusion indices.
  // Step 4a) Get FusionDefinition index for cloned inputs and outputs. Map them
  // to their original fusion indices.
  const std::vector<Val*>& cloned_inputs = sg->inputs();
  const std::vector<Val*>& cloned_outputs = sg->outputs();

  std::vector<int64_t> original_fid;
  original_fid.reserve(cloned_inputs.size() + cloned_outputs.size());

  std::transform(
      cloned_inputs.begin(),
      cloned_inputs.end(),
      std::back_inserter(original_fid),
      [&](Val* v) { return map_cloned_value_to_fid_.at(v); });

  std::transform(
      cloned_outputs.begin(),
      cloned_outputs.end(),
      std::back_inserter(original_fid),
      [&](Val* v) { return map_cloned_value_to_fid_.at(v); });

  // Step 4b) ir_cloner maps cloned fusion Vals to segment Vals.
  std::vector<Val*> segment_inputs_outputs;
  segment_inputs_outputs.reserve(cloned_inputs.size() + cloned_outputs.size());

  std::transform(
      cloned_inputs.begin(),
      cloned_inputs.end(),
      std::back_inserter(segment_inputs_outputs),
      [&](Val* v) { return cloned_to_segment_map.clone(v); });

  std::transform(
      cloned_outputs.begin(),
      cloned_outputs.end(),
      std::back_inserter(segment_inputs_outputs),
      [&](Val* v) { return cloned_to_segment_map.clone(v); });

  // Step 4c) Map segment Vals to their FusionDefinition index.
  std::vector<int64_t> segment_fid;
  segment_fid.reserve(segment_inputs_outputs.size());
  std::transform(
      segment_inputs_outputs.begin(),
      segment_inputs_outputs.end(),
      std::back_inserter(segment_fid),
      [&](Val* v) { return map_translated_val_to_segment_fid.at(v); });

  // Step 4d) Map original indices to segment indices.
  NVF_ERROR(original_fid.size() == segment_fid.size());
  std::unordered_map<int64_t, int64_t> segment_fid_to_original_fid_map;
  for (size_t idx : c10::irange(original_fid.size())) {
    segment_fid_to_original_fid_map.emplace(
        segment_fid.at(idx), original_fid.at(idx));
  }

  // Step 4e) short-circuit: No extra extents required for python definition.
  if (fusion_segment->inputs().size() == other.inputs().size()) {
    return segment_fid_to_original_fid_map;
  }

  // The python segment can require the size of tensor dimensions from original
  // fusion's input arguments, which the CPP segment does not.

  // Step 4f) Create a map from segment to cloned extents.
  // Step 4g) Create a map from segment indices to segment extents.
  std::unordered_map<Val*, Val*> segment_to_cloned_extents;
  std::unordered_map<size_t, Val*> segment_fid_to_translated_val;
  for (Val* cloned_extent : cloned_extents_) {
    Val* segment_extent = cloned_to_segment_map.clone(cloned_extent);

    // short-circuit: some extents are not used in segment
    if (map_translated_val_to_segment_fid.count(segment_extent) == 0) {
      continue;
    }

    size_t segment_fid = map_translated_val_to_segment_fid.at(segment_extent);
    segment_to_cloned_extents.emplace(segment_extent, cloned_extent);
    segment_fid_to_translated_val.emplace(segment_fid, segment_extent);
  }

  // Step 4h) Find the set difference between all segment input indices and
  // known input segment indices.
  std::vector<int64_t> missing_segment_fid;
  for (int64_t input_fid : other.inputs()) {
    if (segment_fid_to_original_fid_map.count(input_fid) == 0) {
      missing_segment_fid.push_back(input_fid);
    }
  }

  // Step 4i) Get segment Val for missing segment input indices.
  std::vector<Val*> missing_segment_val;
  missing_segment_val.reserve(missing_segment_fid.size());
  std::transform(
      missing_segment_fid.begin(),
      missing_segment_fid.end(),
      std::back_inserter(missing_segment_val),
      [&](int64_t segment_fid) {
        return segment_fid_to_translated_val.at(segment_fid);
      });

  // Step 4j) Map segment Vals to cloned Vals
  std::vector<Val*> missing_cloned_val;
  missing_cloned_val.reserve(missing_segment_val.size());
  std::transform(
      missing_segment_val.begin(),
      missing_segment_val.end(),
      std::back_inserter(missing_cloned_val),
      [&](Val* segment_val) {
        return segment_to_cloned_extents.at(segment_val);
      });

  // Step 4k) Transform cloned Vals to their original fusion indices.
  std::vector<int64_t> missing_cloned_fid;
  missing_cloned_fid.reserve(missing_cloned_val.size());
  std::transform(
      missing_cloned_val.begin(),
      missing_cloned_val.end(),
      std::back_inserter(missing_cloned_fid),
      [&](Val* cloned_val) { return map_cloned_value_to_fid_.at(cloned_val); });

  // Step 4l) Add missing mappings from segment to original indices.
  for (size_t idx : c10::irange(missing_segment_fid.size())) {
    segment_fid_to_original_fid_map.emplace(
        missing_segment_fid.at(idx), missing_cloned_fid.at(idx));
  }

  // Return the mapping from the index space of segment FusionDefinition to the
  // index space of the original FusionDefinition.
  return segment_fid_to_original_fid_map;
}

void SegmentationState::prepareGroupOrder() {
  NVF_ERROR(segmented_fusion_ != nullptr);

  // Gather initial inputs for SegmentedFusion.
  std::unordered_set<Val*> available_input;
  std::copy(
      segmented_fusion_->inputs().begin(),
      segmented_fusion_->inputs().end(),
      std::inserter(available_input, available_input.end()));

  // The size of the tensor dimensions can be used as an input of the segments.
  // NvFuser does not support returning scalar values. Segmentation must pass
  // those sizes as segment arguments manually.
  std::vector<Val*> extents = getExtents(segmented_fusion_->completeFusion());
  std::copy(
      extents.begin(),
      extents.end(),
      std::inserter(available_input, available_input.end()));

  // Track the run status of all SegmentedGroups in SegmentedFusion
  std::vector<bool> group_ran(segmented_fusion_->groups().size(), false);

  // While not all the SegmentedGroups are run:
  while (!std::all_of(
      group_ran.begin(), group_ran.end(), [](bool b) { return b; })) {
    bool ran_any_group = false;

    // Find the first segment with all inputs available to run
    for (size_t group_i : c10::irange(segmented_fusion_->groups().size())) {
      SegmentedGroup* group = segmented_fusion_->groups().at(group_i);

      // short-circuit: Already ran this segmented group.
      if (group_ran.at(group_i)) {
        continue;
      }

      const std::vector<Val*>& group_inputs = group->inputs();
      bool ready_to_run = std::all_of(
          group_inputs.begin(),
          group_inputs.end(),
          [&available_input](Val* val) { return available_input.count(val); });

      // short-circuit: This segmented group is not ready to run.
      if (!ready_to_run) {
        continue;
      }

      // Add SegmentedGroup to group_run_order_.
      group_run_order_.push_back(group);

      // Mark all outputs of SegmentedGroup as ready.
      const std::vector<Val*>& group_outputs = group->outputs();
      for (size_t group_out_i : c10::irange(group_outputs.size())) {
        available_input.insert(group_outputs.at(group_out_i));
      }
      group_ran[group_i] = true;
      ran_any_group = true;
    }
    NVF_ERROR(
        ran_any_group,
        "Failed to run all groups; An error must have occured in segmentation.");
  }
}

} // namespace nvfuser::python_frontend

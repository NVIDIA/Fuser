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
    const std::unordered_map<const Val*, int64_t>&
        map_presched_value_to_original_python_index,
    const KernelArgumentHolder& args) {
  // Check state
  NVF_ERROR(fusion != nullptr);
  NVF_ERROR(cloned_original_fusion_ == nullptr);
  NVF_ERROR(segmented_fusion_ == nullptr);
  NVF_ERROR(group_run_order_.empty());
  NVF_ERROR(map_cloned_concretized_value_to_original_python_index_.empty());
  NVF_ERROR(cloned_original_extents_.empty());

  // Step 1) Clone preschedFusion CPP Fusion.
  cloned_original_fusion_ = std::make_unique<Fusion>();

  // The IRCloner returned by Fusion::copy acts as map from the original fusion
  // to the cloned fusion.
  IrCloner original_to_cloned_map =
      Fusion::copy(fusion, cloned_original_fusion_.get());

  // Step 2) Given the map_presched_value_to_original_python_index AND the
  // IRCloner returned by Fusion::copy, create a mapping from cloned CPP values
  // to original fusion state indices.
  std::unordered_map<Val*, int64_t> map_cloned_value_to_original_python_index;
  map_cloned_value_to_original_python_index.reserve(
      map_presched_value_to_original_python_index.size());
  std::transform(
      map_presched_value_to_original_python_index.begin(),
      map_presched_value_to_original_python_index.end(),
      std::inserter(
          map_cloned_value_to_original_python_index,
          map_cloned_value_to_original_python_index.end()),
      [&](const auto& item) {
        const Val* original_value = item.first;
        int64_t python_index = item.second;
        Val* cloned_value = original_to_cloned_map.clone(original_value);
        return std::make_pair(cloned_value, python_index);
      });

  // Step 3) Concretize fusion with input arguments.
  std::unordered_map<Val*, Val*> symbolic_to_concrete_map =
      DynamicTransform::concretizeFusion(cloned_original_fusion_.get(), args);

  // Given the map_cloned_value_to_original_python_index AND the
  // symbolic_to_concrete map returned by the concretization pass, create a
  // mapping from cloned, concretized CPP values to original fusion state
  // indices.
  map_cloned_concretized_value_to_original_python_index_.reserve(
      map_cloned_value_to_original_python_index.size());
  std::transform(
      map_cloned_value_to_original_python_index.begin(),
      map_cloned_value_to_original_python_index.end(),
      std::inserter(
          map_cloned_concretized_value_to_original_python_index_,
          map_cloned_concretized_value_to_original_python_index_.end()),
      [&](const auto& item) {
        Val* maybe_concretized_value = item.first;
        int64_t python_index = item.second;
        if (symbolic_to_concrete_map.count(maybe_concretized_value) > 0) {
          maybe_concretized_value =
              symbolic_to_concrete_map.at(maybe_concretized_value);
        }
        return std::make_pair(maybe_concretized_value, python_index);
      });

  // Track the extents for input TensorViews in cloned CPP Fusion.
  cloned_original_extents_ = getExtents(cloned_original_fusion_.get());

  // Create runtime infomation
  SchedulerRuntimeInfo runtime_info(
      cloned_original_fusion_.get(),
      args,
      /*precomputed_values=*/nullptr,
      cloned_original_fusion_->allTvs());

  // Run segmentation algorithm
  segmented_fusion_ = SegmentCandidateFinder::segment(
      std::move(cloned_original_fusion_), args, runtime_info);

  // Get the order for fusion segments
  prepareGroupOrder();

  // Return the number of segments created by segmentation algorithm.
  return (int64_t)segmented_fusion_->groups().size();
}

// setupSegmentation transforms the Prescheduled, Symbolic Fusion to Cloned,
// Concretized Fusion. Both CPP fusions corresponds with Original
// FusionDefinition.
//
// The segmentation pass runs on cloned, concretized fusion to create
// SegmentedFusion. Each SegmentedGroup in the SegmentedFusion creates a segment
// CPP fusion that is translated to a python definition.
//
//
// NOTE: Steps 4a through 4d are run for every fusion segment. However,
// sometimes the python definition needs the extents of the original fusion's
// input tensors as extra arguments. Steps 4f to 4l creates mappings for these
// missing extents.
//
// Details:
//  1) Use segment id to get SegmentedGroup from group_run_order_.
//  2) Create CPP Fusion for SegmentedGroup.
//  * IrCloner acts as a map from fusion segment to the original fusion.
//  3) Translate CPP Fusion to Python FusionDefinition
//  4) Create map from segment fusion indices to original fusion indices.
//    a) Get cloned Vals for SegmentedGroup's inputs and outputs. Map them
//       to their original fusion indices.
//    b) Map cloned Vals to their segment Vals
//    c) Map segment Vals to their fusion indices.
//    d) Map original indices to segment indices.
//    e) Return map if the number of input arguments for python definition
//       matches the number of input arguments for CPP fusion.
//    f) Create a map from segment to cloned extents.
//    g) Create a map from segment fusion indices to cloned extents.
//    h) Find segment inputs that are missing from segment to original
//       indices map.
//    i) Get segment Vals for the missing segment fusion indices.
//    j) Map segment Vals to cloned Vals.
//    k) Map cloned Vals to their corresponding fusion indices.
//    l) Add missing mappings to segment to original indices map.
//  5) Return the mapping from the segmented FusionDefinition index space to
//  original FusionDefinition index space.
std::unordered_map<int64_t, int64_t> SegmentationState::buildSegment(
    FusionDefinition& segment_fd,
    int64_t segment_id) {
  NVF_ERROR(
      !segment_fd.completed(),
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
      map_segment_cpp_value_to_python_index =
          translate(fusion_segment.get(), &segment_fd);

  // Step 4) Create map from segment fusion indices to original fusion indices.
  // Step 4a) Get FusionDefinition index for cloned inputs and outputs. Map them
  // to their original fusion indices.
  const std::vector<Val*>& cloned_inputs = sg->inputs();
  const std::vector<Val*>& cloned_outputs = sg->outputs();

  std::vector<int64_t> original_python_index;
  original_python_index.reserve(cloned_inputs.size() + cloned_outputs.size());

  std::transform(
      cloned_inputs.begin(),
      cloned_inputs.end(),
      std::back_inserter(original_python_index),
      [&](Val* v) {
        return map_cloned_concretized_value_to_original_python_index_.at(v);
      });

  std::transform(
      cloned_outputs.begin(),
      cloned_outputs.end(),
      std::back_inserter(original_python_index),
      [&](Val* v) {
        return map_cloned_concretized_value_to_original_python_index_.at(v);
      });

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
  std::vector<int64_t> segment_python_index;
  segment_python_index.reserve(segment_inputs_outputs.size());
  std::transform(
      segment_inputs_outputs.begin(),
      segment_inputs_outputs.end(),
      std::back_inserter(segment_python_index),
      [&](Val* v) { return map_segment_cpp_value_to_python_index.at(v); });

  // Step 4d) Map original indices to segment indices.
  NVF_ERROR(original_python_index.size() == segment_python_index.size());
  std::unordered_map<int64_t, int64_t> segment_to_original_python_index_map;
  for (size_t idx : arange(original_python_index.size())) {
    segment_to_original_python_index_map.emplace(
        segment_python_index.at(idx), original_python_index.at(idx));
  }

  // Step 4e) short-circuit: No extra extents required for python definition.
  if (fusion_segment->inputs().size() == segment_fd.inputs().size()) {
    return segment_to_original_python_index_map;
  }

  // The python segment can require the size of tensor dimensions from original
  // fusion's input arguments, which the CPP segment does not.

  // Step 4f) Create a map from segment to cloned extents.
  // Step 4g) Create a map from segment indices to segment extents.
  std::unordered_map<Val*, Val*> segment_to_cloned_extents;
  std::unordered_map<size_t, Val*> segment_python_index_to_cpp_val;
  for (Val* cloned_extent : cloned_original_extents_) {
    Val* segment_extent = cloned_to_segment_map.clone(cloned_extent);

    // short-circuit: some extents are not used in segment
    if (map_segment_cpp_value_to_python_index.count(segment_extent) == 0) {
      continue;
    }

    size_t segment_python_index =
        map_segment_cpp_value_to_python_index.at(segment_extent);
    segment_to_cloned_extents.emplace(segment_extent, cloned_extent);
    segment_python_index_to_cpp_val.emplace(
        segment_python_index, segment_extent);
  }

  // Step 4h) Find the set difference between all segment input indices and
  // known input segment indices.
  std::vector<int64_t> missing_segment_python_index;
  for (int64_t input_python_index : segment_fd.inputs()) {
    if (segment_to_original_python_index_map.count(input_python_index) == 0) {
      missing_segment_python_index.push_back(input_python_index);
    }
  }

  // Step 4i) Get segment Val for missing segment input indices.
  std::vector<Val*> missing_segment_val;
  missing_segment_val.reserve(missing_segment_python_index.size());
  std::transform(
      missing_segment_python_index.begin(),
      missing_segment_python_index.end(),
      std::back_inserter(missing_segment_val),
      [&](int64_t segment_python_index) {
        return segment_python_index_to_cpp_val.at(segment_python_index);
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
  std::vector<int64_t> missing_cloned_python_index;
  missing_cloned_python_index.reserve(missing_cloned_val.size());
  std::transform(
      missing_cloned_val.begin(),
      missing_cloned_val.end(),
      std::back_inserter(missing_cloned_python_index),
      [&](Val* cloned_val) {
        return map_cloned_concretized_value_to_original_python_index_.at(
            cloned_val);
      });

  // Step 4l) Add missing mappings from segment to original indices.
  for (size_t idx : arange(missing_segment_python_index.size())) {
    segment_to_original_python_index_map.emplace(
        missing_segment_python_index.at(idx),
        missing_cloned_python_index.at(idx));
  }

  // Return the mapping from the index space of segment FusionDefinition to the
  // index space of the original FusionDefinition.
  return segment_to_original_python_index_map;
}

void SegmentationState::prepareGroupOrder() {
  NVF_ERROR(segmented_fusion_ != nullptr);

  // Gather initial inputs for SegmentedFusion.
  std::unordered_set<Val*> available_input(
      segmented_fusion_->inputs().begin(), segmented_fusion_->inputs().end());

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
    for (size_t group_i : arange(segmented_fusion_->groups().size())) {
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
      for (size_t group_out_i : arange(group_outputs.size())) {
        available_input.insert(group_outputs.at(group_out_i));
      }
      group_ran[group_i] = true;
      ran_any_group = true;
    }
    NVF_ERROR(
        ran_any_group,
        "Failed to run any group; An error must have occured in segmentation.");
  }
}

} // namespace nvfuser::python_frontend

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

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <python_frontend/fusion_state.h>
#include <python_frontend/fusion_definition.h>
#include <runtime/fusion_executor_cache.h>

namespace nvfuser::python_frontend {

class FusionDefinition;

class SegmentationState {
 public:
  //! Run segmentation algorithm on FusionDefinition. Returns the number of
  //! segments.
  int64_t setupSegmentation(
      Fusion* fusion,
      const std::unordered_map<const Val*, int64_t>& map_value_to_original_fid,
      const at::ArrayRef<c10::IValue>& inputs);

  //! Given SegmentedFusion and vector of FusionDefinition objects for the
  //! fusion segments, create the fusion segments and clone their state to the
  //! FusionDefinitions.
  NVF_API std::unordered_map<int64_t, int64_t> buildSegment(
      FusionDefinition& other,
      int64_t segment_id);

  //! Perform a topological sort on SegmentedFusion to segment order.
  void prepareGroupOrder();

 private:
  //! Clone of original fusion for segmentation
  std::unique_ptr<Fusion> segment_fusion_ = nullptr;

  //! This FusionDefinition may require multiple kernels if it cannot be handled
  //! by a single heuristic scheduler. SegmentedFusion takes a fusion and runs
  //! the segmentation algorithm.
  std::unique_ptr<SegmentedFusion> segmented_fusion_ = nullptr;

  //! Pre-determined order to run the segmented groups
  std::vector<SegmentedGroup*> group_run_order_;

  //! Create copy of fusion for segmentation algorithm. IrCloner is a map
  //! between values in original and cloned fusions.
  std::unordered_map<const Val*, int64_t> map_cloned_value_to_fid_;

  //! Extents for TensorView input arguments for cloned Fusion
  std::vector<Val*> cloned_extents_;
};

} // namespace nvfuser::python_frontend

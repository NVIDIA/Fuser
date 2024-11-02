// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_state.h>
#include <runtime/fusion_executor_cache.h>

namespace nvfuser::python_frontend {

class FusionDefinition;

//! ===========================================================================
//
//! setupSegmentation runs the segmentation algorithm on CPP Fusion to create
//! SegmentedFusion. It returns the number of segments in SegmentedFusion.
//!
//! Details:
//!  1) Clone preschedFusion CPP Fusion.
//!  2) Concretize fusion using input arguments.
//!  3) Given the map_value_to_original_fid, the IRCloner returned by
//!     Fusion::copy, AND symbolic_to_concrete map returned by concretization
//!     pass, create a mapping from cloned Vals to original fusion state indices
//!  4) Get extents for cloned fusion
//!  5) Create SchedulerRuntimeInfo
//!  6) Run segmentation algorithm using cloned fusion, input arguments, and
//!  scheduler runtime infomation.
//!  7) Get sequential order of fusion segments using prepareGroupOrder.
//!  8) Return the number of segments created by segmentation algorithm.
//! ===========================================================================
//!
//! buildSegment creates the CPP Fusion for a given segment id, translate it to
//! the python FusionDefinition, then returns a mapping from segment fusion
//! state indices to the original fusion state indices.
//!
//! Why do we need a map from the segment's fusion index space to the original
//! fusion index space?
//!
//! * The original FusionDefinition is decomposed into a sequence of segment
//! FusionDefinitions.
//! * Each FusionDefinition has an independent index space.
//! * At runtime, the original FusionDefinition acts an argument manager,
//! gathering input arguments and storing output results.
//! * To perform this function, it requires a map from the segment index space
//! to the original index space.
//!
//! NOTE: Steps 4a through 4d are run for every fusion segment. However,
//! sometimes the python definition needs the extents of the original fusion's
//! input tensors as extra arguments. Steps 4f to 4l creates mappings for these
//! missing extents.
//!
//! Details:
//!  1) Use segment id to get SegmentedGroup from group_run_order_.
//!  2) Create CPP Fusion for SegmentedGroup.
//!  * IrCloner acts as a map from fusion segment to the original fusion.
//!  3) Translate CPP Fusion to Python FusionDefinition
//!  4) Create map from segment fusion indices to original fusion indices.
//!    a) Get original Vals for SegmentedGroup's inputs and outputs.
//!    b) Map original Vals to their original fusion indices.
//!    c) Map original Vals to their segment Vals
//!    d) Map segment Vals to their fusion indices.
//!    e) Return map if the number of input arguments for python definition
//!       matches the number of input arguments for CPP fusion.
//!    f) Create a map from segment to original extents.
//!    g) Create a map from segment fusion indices to original extents.
//!    h) Find segment inputs that are missing from segment to original
//!       indices map.
//!    i) Get segment CPP Vals for the missing segment fusion indices.
//!    j) Map segment CPP Vals to original CPP Vals.
//!    k) Map original CPP Vals to their corresponding fusion indices.
//!    l) Add missing mappings to segment to original indices map.
//!  5) Return the mapping from the segmented FusionDefinition index space to
//!  original FusionDefinition index space.
//!
//! ===========================================================================
//
//! prepareGroupOrder is similar to prepareRuntimeOrder. It generates the
//! sequential order of SegmentedGroups in SegmentedFusion.
//!
//! Details:
//!  1) Gather initial inputs for SegmentedFusion.
//!  2) Gather IterDomain extents from the tensor input arguments.
//!  3) Track the run status of all SegmentedGroups in SegmentedFusion
//!  4) While not all the SegmentedGroups are run:
//!  5)   For each SegmentedGroup:
//!  6)     Skip SegmentedGroup if it is already run
//!  7)     Skip SegmentedGroup if inputs are not ready
//!  8)     Add SegmentedGroup to group_run_order_. Mark all outputs of
//!         SegmentedGroup as ready.
//!  9)   End For
//!  10)  Fail if none of the SegmentedGroups are available to run.
//!  11) End While
//! ===========================================================================
class SegmentationState {
 public:
  // Run segmentation algorithm on FusionDefinition.
  int64_t setupSegmentation(
      Fusion* fusion,
      const std::unordered_map<const Val*, int64_t>& map_value_to_original_fid,
      const at::ArrayRef<c10::IValue>& inputs);

  // Given an empty FusionDefinition and a segment id, buildSegment creates the
  // CPP Fusion, translates it to the python FusionDefinition, then return a
  // mapping from segment fusion state indices to the original fusion state
  // indices.
  NVF_API std::unordered_map<int64_t, int64_t> buildSegment(
      FusionDefinition& other,
      int64_t segment_id);

  // Perform a topological sort on SegmentedFusion to segment order.
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

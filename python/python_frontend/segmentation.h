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

//! Overview:
//! Segmentation decomposes a fusion into a directed acyclic graph (DAG) of
//! sub-fusions. After applying the segmentation algorithm, we can translate
//! the sub-fusions into their corresponding python definitions. Then, given the
//! fusion's input arguments, the segments are run in the correct order to
//! produce the output results.
//!
//! Each FusionDefinition contains a set of states representing tensors, vectors
//! and scalars. Every state has a unique index, which matches the insertion
//! order of the state in the FusionDefinition. These indices form a linear
//! index space for each FusionDefinition.
//!
//! The original FusionDefinition stores the sequence of sub-fusions and acts as
//! an argument manager. It gathers the input arguments before running the
//! sub-fusion and stores its results. To perform this function, it requires a
//! map from the segment index space to the original index space. This mapping
//! is generated while creating the python definition for each sub-fusion.
//!
//! Algorithm:
//! Step 1: setupSegmentation runs the segmentation algorithm on the CPP Fusion
//! to create the SegmentedFusion. Then, sub-fusions are ordered according to
//! their dependencies by the prepareGroupOrder function. It returns the number
//! of segments in SegmentedFusion.
//!
//! Step 2: buildSegment creates the CPP Fusion for a given segment id,
//! translates it to a python FusionDefinition, then returns a mapping from the
//! segment fusion state indices to the original fusion state indices.
//!
//! ===========================================================================
//!
//! Example 1: A simple fusion with two iota operations.
//!
//! Original Fusion:
//! def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
//!     S0 = fd.define_scalar(2, dtype=DataType.Int)
//!     S1 = fd.define_scalar(0, dtype=DataType.Int)
//!     S2 = fd.define_scalar(2, dtype=DataType.Int)
//!     T3 = fd.ops.iota(S0, S1, S2, dtype=DataType.Int)
//!     S4 = fd.define_scalar(3, dtype=DataType.Int)
//!     S5 = fd.define_scalar(100, dtype=DataType.Int32)
//!     S6 = fd.define_scalar(1, dtype=DataType.Int32)
//!     T7 = fd.ops.iota(S4, S5, S6, dtype=DataType.Int32)
//!     fd.add_output(T3)
//!     fd.add_output(T7)
//!
//! After Segmentation:
//! The original fusion is divided into two segments. There is no dependencies
//! between either segment so they can run in any order.
//!
//! First Segment:
//! def nvfuser_fusion_id2(fd : FusionDefinition) -> None :
//!     S0 = fd.define_scalar(2, dtype=DataType.Int)
//!     S1 = fd.define_scalar(0, dtype=DataType.Int)
//!     S2 = fd.define_scalar(2, dtype=DataType.Int)
//!     T3 = fd.ops.iota(S0, S1, S2, dtype=DataType.Int)
//!     fd.add_output(T3)
//!
//! Second Segment:
//! def nvfuser_fusion_id3(fd : FusionDefinition) -> None :
//!     S0 = fd.define_scalar(3, dtype=DataType.Int)
//!     S1 = fd.define_scalar(100, dtype=DataType.Int32)
//!     S2 = fd.define_scalar(1, dtype=DataType.Int32)
//!     T3 = fd.ops.iota(S0, S1, S2, dtype=DataType.Int32)
//!     fd.add_output(T3)
//!
//! The first segment corresponds with [S0, S1, S2, T3] in the original fusion.
//! The second segment corresponds with [S4, S5, S6, S7] in the original fusion.
//!
//! Neither segment requires any input arguments from the original fusion.
//!
//! For the first segment, the segment's T3 is mapped to the original's T3.
//! Segment Index : Original Index Mapping
//! --------------------------------------
//! T3 : T3
//!
//! For the second segment the segment's T3 is mapped to the original's T7.
//! Segment Index : Original Index Mapping
//! --------------------------------------
//! T3 : T7
//!
//! ===========================================================================
//!
//! Example 2: A reduction + broadcast + pointwise fusion.
//!
//! Original Fusion:
//! def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
//!     T0 = fd.define_tensor(shape=[-1, -1],
//!                           contiguity=[True, True],
//!                           dtype=DataType.Float,
//!                           is_cpu=False)
//!     T1 = fd.define_tensor(shape=[-1, -1],
//!                           contiguity=[True, True],
//!                           dtype=DataType.Float,
//!                           is_cpu=False)
//!     T2 = fd.ops.sum(T0, dims=[1], keepdim=False, dtype=DataType.Float)
//!     T3 = fd.ops.broadcast(T2, is_broadcast_dim=[False, True])
//!     T4 = fd.ops.add(T1, T3)
//!     fd.add_output(T4)
//!
//! After Segmentation:
//! The reduction scheduler does not support fusing any operations with an
//! inner reduction, so the original fusion is divided into two segments.
//! Segment 2 depends on Segment 1, so there is a strict segment ordering.
//!
//! First Segment:
//! def nvfuser_fusion_id2(fd : FusionDefinition) -> None :
//!    T0 = fd.define_tensor(shape=[-1, -1],
//!                          contiguity=[True, True],
//!                          dtype=DataType.Float,
//!                          is_cpu=False)
//!    T1 = fd.ops.sum(T0, dims=[1], keepdim=False, dtype=DataType.Float)
//!    T2 = fd.ops.broadcast(T1, is_broadcast_dim=[False, True])
//!    fd.add_output(T2)
//!
//! Second Segment:
//! def nvfuser_fusion_id3(fd : FusionDefinition) -> None :
//!    T0 = fd.define_tensor(shape=[-1, -1],
//!                          contiguity=[True, True],
//!                          dtype=DataType.Float,
//!                          is_cpu=False)
//!    T1 = fd.define_tensor(shape=[-1, 1],
//!                          contiguity=[True, None],
//!                          dtype=DataType.Float,
//!                          is_cpu=False)
//!    T2 = fd.ops.add(T0, T1)
//!    fd.add_output(T2)
//!
//! The first segment contains the reduction and broadcast operations, which
//! corresponds with [T0, T2, T3] in the original fusion. Therefore, the segment
//! index to original index map has two entries.
//!
//! Segment Index : Original Index Mapping
//! --------------------------------------
//! T0 : T0 --- The first tensor argument for the original fusion.
//! T2 : T3 --- The broadcasted, reduction tensor is this segment's output.
//!
//! The second segment is the pointwise addition with the broadcasted reduction.
//! It corresponds with [T1, T3, T4] in the original fusion.
//!
//! Segment Index : Original Index Mapping
//! --------------------------------------
//! T0 : T1 --- The second tensor argument for the original fusion.
//! T1 : T3 --- The broadcasted, reduction tensor, which is the output from the
//!             first segment.
//! T2 : T4 --- The pointwise addition, which is the output for the original
//!             fusion.
//! ===========================================================================
class SegmentationState {
 public:
  // setupSegmentation runs the segmentation algorithm on CPP Fusion to create
  // SegmentedFusion. It returns the number of segments in SegmentedFusion.
  //
  // Details:
  //  1) Clone preschedFusion CPP Fusion.
  //  2) Concretize fusion using input arguments.
  //  3) Given the map_presched_value_to_original_python_index, the IRCloner
  //  returned by Fusion::copy, AND symbolic_to_concrete map returned by
  //  concretization pass, create a mapping from cloned Vals to original fusion
  //  state indices.
  //  4) Get extents for cloned fusion.
  //  5) Create SchedulerRuntimeInfo.
  //  6) Run segmentation algorithm using cloned fusion, input arguments, and
  //  scheduler runtime information.
  //  7) Get sequential order of fusion segments using prepareGroupOrder.
  //  8) Return the number of segments created by segmentation algorithm.
  int64_t setupSegmentation(
      Fusion* fusion,
      const std::unordered_map<const Val*, int64_t>&
          map_presched_value_to_original_python_index,
      const KernelArgumentHolder& inputs);

  // Given an empty FusionDefinition and a segment id, buildSegment creates the
  // CPP Fusion, translates it to the python FusionDefinition, then returns a
  // mapping from segment fusion state indices to the original fusion state
  // indices.
  //
  // The mapping is constructed from the segment's python definition ->
  // segment's CPP Fusion -> original's CPP Fusion -> original's python
  // definition.
  //
  // NOTE: Sometimes the python definition requires the extents from the
  // original fusion's input tensors as extra arguments. Therefore, the input
  // arguments for the python definition and the CPP Fusion may not exactly
  // match.
  NVF_API std::unordered_map<int64_t, int64_t> buildSegment(
      FusionDefinition& segment_fd,
      int64_t segment_id);

 private:
  // prepareGroupOrder is similar to prepareRuntimeOrder. It generates the
  // topological order of SegmentedGroups in SegmentedFusion.
  //
  // Details:
  //  1) Gather initial inputs for SegmentedFusion.
  //  2) Gather IterDomain extents from the tensor input arguments.
  //  3) Track the run status of all SegmentedGroups in SegmentedFusion
  //  4) While not all the SegmentedGroups are run:
  //  5)   For each SegmentedGroup:
  //  6)     Skip SegmentedGroup if it is already run
  //  7)     Skip SegmentedGroup if inputs are not ready
  //  8)     Add SegmentedGroup to group_run_order_. Mark all outputs of
  //         SegmentedGroup as ready.
  //  9)   End For
  //  10)  Fail if none of the SegmentedGroups are available to run.
  //  11) End While
  void prepareGroupOrder();

 private:
  // Clone of original fusion for segmentation
  std::unique_ptr<Fusion> cloned_original_fusion_ = nullptr;

  // This FusionDefinition may require multiple kernels if it cannot be handled
  // by a single heuristic scheduler. SegmentedFusion takes a fusion and runs
  // the segmentation algorithm.
  std::unique_ptr<SegmentedFusion> segmented_fusion_ = nullptr;

  // Pre-determined order to run the segmented groups
  std::vector<SegmentedGroup*> group_run_order_;

  // Map values from cloned, concretized fusion to the indices of the original
  // python definition.
  std::unordered_map<const Val*, int64_t>
      map_cloned_concretized_value_to_original_python_index_;

  // Extents for TensorView input arguments for cloned Fusion
  std::vector<Val*> cloned_original_extents_;
};

} // namespace nvfuser::python_frontend

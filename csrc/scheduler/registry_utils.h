// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <scheduler/all_schedulers.h>

namespace nvfuser {

class TensorView;
class ComputeAtLogicalDomainMap;
class ComputeAtMap;
class ExpressionEvaluator;
class KernelArgumentHolder;

namespace registry_utils {

bool checkPatternEquivalence(
    TensorView* out_tv0,
    TensorView* out_tv1,
    const ComputeAtLogicalDomainMap& logical_map);

// Reusing some code from lowering specifically in lower_trivial_broadcast.cpp
// ConcretizedBroadcastDomains::maybeNonUniquelyConcretized this checks if
// there's a broadcast iteration domain that's being broadcasted to seemingly
// different extents, meaning we don't know in the kernel if the dimension is
// being broadcasted to one size multiple times or different sizes. This is a
// hard to optimize problem and likely indicates we shouldn't be fusing.
bool hasNonUniqueBcast(Fusion* fusion);

// TODO: remove this requirement entirely
bool rejectScheduleForMemoryPromotion(
    Fusion* fusion,
    SchedulerType scheduler_type);

bool isConnectedFusionGraph(Fusion* fusion);

// Returns if a fusion cannot transformed into a consistent format since we
// can't transform forward through view operations, for exmaple:
//
// tv0[I0, I1, I2]
// tv1[I0*I1, I2] = view(tv0)
// tv2[I0, I1*I2] = view(tv0)
//
// If we start transform propagation at either tv1 or tv2, it would require
// "replaying forward" through the other. If we started at tv1 we'd have to be
// able to take tv2[I0, I1*I2] and transform it to [I0*I1, I2], however this
// would "undo" the view transformation which we do not support today.
//
// Returns true if a scenario like above is found in the fusion.
bool requiresForwardViewReplay(Fusion* fusion, ComputeAtMap& ca_map);

// Returns if view interferes with how we want to treat the reference, being
// at least a 2D reduction schedule but maybe a 3D reduction schedule.
bool reductionInterferingView(
    Fusion* fusion,
    const ComputeAtMap& ca_map,
    TensorView* reduction_reference);

// Check inputs, outputs and intermediates
// Intermediates are contiguous, so strides are not necessary
// Strides are required for inputs and also maybe for outputs as
// they may be non-contiguous. However, in our current interface,
// output strides are not available, so if there's any outputs that
// are non contiguous, need to fall back to 64-bit indexing
PrimDataType getIndexTypeOfKernel(
    Fusion* fusion,
    const std::vector<TensorView*>& all_tvs,
    const KernelArgumentHolder& inputs,
    ExpressionEvaluator& ee);

// Check if the block scales output of Block Quantization Op
// is a segment output.
bool hasNonTerminalBlockQuantizeOp(Fusion* fusion);

class SchedulerTopologyChecker {
 public:
  // Checks if any broadcasts are resolved after a reduction that don't follow
  // the normalization pattern
  static bool hasNonNormalizePostReductionBCast(Fusion* fusion);

  // Checks if any broadcasts are resolved after a reduction, this shouldn't
  // be accepted in the single reduction or multi-reduction scheduler
  static bool hasPostReductionBCast(Fusion* fusion);

  // Checks if there's any unsupported operations post reduction. If outer
  // reduction we can fuse some pointwise ops if they don't require
  // broadcasting (checked in hasPostReductionBCast). For inner reductions we
  // cannot fuse any binary like operation (includes operations like shift
  // that we're not fusing right now) involving "new" inputs (not going
  // through a reduction).
  static bool supportedPostReductionFusion(
      Fusion* fusion,
      std::vector<TensorView*> reduction_tvs);

  // Checks if there's any gather-like ops that result in non-resolved
  // broadcast domains and then get squeezed before reaching reduction
  // TVs. The reduction scheduler uses reduction TVs as a scheduling
  // reference, so that won't be able to schedule the broadcast ID if
  // squeezed and its corresponding index-accessed producer ID, and
  // any IDs that the producer ID depends on.
  //
  // This analysis has some similarity as DomainMap. Can be
  // consolidated?
  static bool hasGatherToBroadcastBeforeReduction(
      Fusion* fusion,
      const std::vector<TensorView*>& reduction_tvs);

  static bool hasResizeAndIndexOps(Fusion* fusion);

  // Checks if fusion satisfies the buffer requirement for special operations.
  // E.g. for PreprocessGroupedMatmulInputSf, the runtime function requires both
  // offsets (inputs) and the output TensorView to reside on global memory. This
  // is because indexing is not done during lowering, but rather by runtime
  // function. Keeping offsets and outputs in global memory allows random access
  // without synchronization by threads. We currently rejects fusion where the
  // runtime requirements are not satisfied.
  static bool rejectScheduleFusionGlobalBufferRequirement(
      Fusion* fusion,
      SchedulerType scheduler_type);

  // Checks if a series of reshape ops creates a cycle in the ID
  // graph. It is not currently supported. For example,
  // propagateReshapeTransforms won't work as it won't find any
  // terminating reshape IDs.
  static bool hasCyclicReshape(Fusion* fusion);

  // Checks if there are incompatible reshapes in the fusion.
  // reshapes are propagated to other tvs, replaying one reshape
  // should not cause conflicts with other reshapes, e.g. two ids
  // are the same disjoint val set can't be split by different factors.
  static bool hasIncompatibleTransforms(Fusion* fusion);
};

} // namespace registry_utils

} // namespace nvfuser

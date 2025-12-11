// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/ops.h>

#include <algorithm>
#include <optional>
#include <ranges>
#include <vector>

#include <host_ir/ir.h>
#include <ir/utils.h>
#include <multidevice/allocation_utils.h>
#include <multidevice/propagation.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <transform_replay.h>
#include <utils.h>

namespace nvfuser::hir {

TensorView* shardByStream(TensorView* source, Val* stream_index, Expr* e) {
  NVF_ERROR(
      getShardedIterDomain(
          source, ParallelType::Stream, DomainType::kAllocation) == nullptr,
      "Source allocation shouldn't be sharded on stream: ",
      source);

  auto* destination =
      ops::newValLike(source, *source->getDataType())->as<TensorView>();

  if (std::ranges::find(e->inputs(), source) != e->inputs().end()) {
    // Propagate the allocation domain from `source` to `destination`.
    // Consider adding a config to TransformReplay::selfReplay to control what
    // to propagate, so we don't have to reset the loop domain.
    TransformReplay::selfReplay(source->domain(), destination->domain());
    destination->setLoopDomain(destination->getLogicalDomain());

    // Propagate the loop domain from `e` to `destination`. There are two
    // technical challenges:
    // 1. Loop domains are associated with TensorViews, not Exprs. So we
    // find e's reference output, `ref_out`, and propagate its loop domain.
    // 2. shardLoopLike requires the source and destination to be connected by
    // an Expr. So we create a temporary Expr to connect them and then
    // remove it right after.
    Expr* temp_e = ir_utils::replaceValInExprInputs(e, source, destination);
    // Because HostIrContainer is non-SSA, `e->outputs()`'s definition is still
    // `e`, not `temp_e`, at this point.
    auto* ref_out = findMostParallelTensorView(
        ir_utils::filterByType<TensorView>(e->outputs()));
    NVF_ERROR(ref_out != nullptr, "`e` has no output TensorViews: ", e);
    ref_out->setDefinition(temp_e);
    shardLoopLike(
        ref_out,
        destination,
        deviceAndStreamParallelTypes(),
        PropagateDirection::kBackward);
    temp_e->fusion()->removeExpr(temp_e);
    // Fusion::removeExpr sets all outputs' definitions to nullptr, so we need
    // to restore them. Use-defs are important for haveDifferentShardings to
    // work.  Alternative, we could have Fusion::removeExpr(expr) not to set the
    // definition to nullptr if the definition is not `expr`.
    for (auto* out : e->outputs()) {
      out->setDefinition(e);
    }
  } else {
    NVF_ERROR(
        std::ranges::find(e->outputs(), source) != e->outputs().end(),
        "`source` ",
        source->toInlineString(),
        " is neither an input nor an output of `e`: ",
        e);
    // When `source` is an output of `e`, we simply propagate `source`'s loop
    // domain (and therefore `e`'s loop domain) to `destination`.
    // TransformReplay::selfReplay doesn't require the two TensorDomains to form
    // a producer-consumer relationship, so the logic is much simpler than when
    // `source` is an input of `e`.
    TransformReplay::selfReplay(source->domain(), destination->domain());
  }

  shardAllocationAsLoop(destination, {ParallelType::Stream});
  NVF_ERROR(
      getShardedIterDomain(
          destination, ParallelType::Stream, DomainType::kAllocation) !=
          nullptr,
      "Destination allocation should be sharded on stream after "
      "shardAllocationAsLoop: ",
      destination);

  // Refine the contiguity flags so `out` aliases `in`. This is done similar
  // to AliasFinder::handle(const SliceOp*). We scan through the allocation
  // domain in minor-to-major order. If an IterDomain is parallelized on
  // Stream (thus "sliced"), the next non-broadcast-non-reduction IterDomain
  // has to be marked non-contiguous. For example,
  //
  //   [m, n]
  //      /\.
  //     s  n/s
  //   contiguity = [t, t, t]
  //
  // will become contiguity = [f, t, t].
  //
  //    [m, n]
  //    /\.
  //   s m/s
  //   contiguity = [t, t, t]
  //
  // will remain [t, t, t] because the stream-parallel IterDomain is allocated
  // outermost.
  //
  // Contiguity refinement is done after shardAllocationAsLoop because
  // FinalizeMultideviceDomainPass (which also uses shardAllocationAsLoop)
  // doesn't want to compute contiguity this way.
  //
  // Let's say the loop domains look like the following during finalization:
  // ```
  //   in: [m, n]
  //          / \.
  //         s
  //         |
  //         | op1
  //         v
  //    x: [m, n]
  //          / \.
  //         s
  //         |
  //         | op2
  //         v
  //  out: [m, n]
  //          / \.
  //         s
  // ```
  // `x`'s allocation should be parallelized on stream because `op2` isn't
  // resharding, and the new contiguity ought to be `[t, t, t]`. However, the
  // code below would refine the contiguity to `[f, t, t]`.
  //
  // The issue stems from a time-dependent contract on allocation domains:
  // pre-finalization we treat allocations as unsharded, while
  // post-finalization we commit them to the target sharding. Because
  // shardByStream runs after finalization, it follows a different set of
  // assumptions than the finalization pass.  We anticipated that a "when" in
  // the contract could cause mismatches; this example validates that concern.
  // I don't see an easy fix. The principled approach is to remove the "when"
  // and pay the cost to make sharding propagation and decomposition reason
  // about both loop and allocation consistently.
  std::vector<IterDomain*> new_allocation =
      destination->getMaybeAllocationDomain();
  std::vector<std::optional<bool>> new_contiguity =
      destination->getContiguity();
  bool next_will_be_noncontiguous = false;
  for (auto [i, alloc_id] : enumerate(new_allocation) | std::views::reverse) {
    std::optional<bool>& contiguity = new_contiguity[i];

    if (alloc_id->isBroadcast() || alloc_id->isReduction()) {
      contiguity = std::nullopt;
    } else if (next_will_be_noncontiguous) {
      contiguity = false;
      next_will_be_noncontiguous = false;
    }

    if (alloc_id->isStream()) {
      next_will_be_noncontiguous = true;
    }
  }
  destination->setContiguity(new_contiguity);

  IrBuilder::create<ShardByStream>(destination, source, stream_index);
  return destination;
}

} // namespace nvfuser::hir

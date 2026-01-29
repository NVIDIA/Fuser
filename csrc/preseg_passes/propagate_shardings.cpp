// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "preseg_passes/propagate_shardings.h"

#include <vector>

#include "ir/interface_nodes.h"
#include "ir/iostream.h"
#include "ir/utils.h"
#include "multidevice/propagation.h"
#include "scheduler/utils.h"

namespace nvfuser::preseg_passes {

namespace {

// Sort the given tvs by the number of device/stream dimensions in descending
// order. Break ties by rank of device mesh.
template <typename Range>
std::vector<TensorView*> sortTvsByParallelDims(const Range& tvs) {
  auto num_parallel_dims = [](TensorView* tv) {
    return std::count_if(
        tv->getLoopDomain().begin(),
        tv->getLoopDomain().end(),
        [](IterDomain* id) {
          return !id->isReduction() && (id->isStream() || id->isDeviceDim());
        });
  };

  std::vector<TensorView*> tvs_vec(tvs.begin(), tvs.end());

  std::ranges::stable_sort(tvs_vec, [&num_parallel_dims](auto a, auto b) {
    return std::make_pair(num_parallel_dims(a), a->getDeviceMesh().rank()) >
        std::make_pair(num_parallel_dims(b), b->getDeviceMesh().rank());
  });

  return tvs_vec;
}

// Order the inputs of the expression based on their priority.
// For linear op, we use weights and bias before input.
// For matmul op, we use weights before input.
// For other ops, we sort the inputs by the number of device/stream dimensions
// in descending order.
std::vector<TensorView*> getOrderedReferenceInputs(Expr* expr) {
  const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
  if (auto* linear_op = dynamic_cast<LinearOp*>(expr)) {
    // Use weights and bias before input.
    if (linear_op->hasBias()) {
      return {linear_op->inB(), linear_op->bias(), linear_op->inA()};
    } else {
      return {linear_op->inB(), linear_op->inA()};
    }
  }

  if (auto* matmul_op = dynamic_cast<MatmulOp*>(expr)) {
    // Use weights before input.
    return {matmul_op->inB(), matmul_op->inA()};
  }

  // Sort inputs by number of device/stream dimensions in descending order
  return sortTvsByParallelDims(inputs);
}

// Returns the set of parallel types not seen on the loop domain of the given
// tvs and hence, can be propagated.
std::unordered_set<ParallelType> getParallelTypesToPropagate(TensorView* tv) {
  std::unordered_set<ParallelType> all_parallel_types;
  // Since we propagate across exprs, either all tvs are fusion inputs or none
  // are.
  if (tv->isFusionInput()) {
    all_parallel_types = {ParallelType::Stream};
  } else {
    all_parallel_types = deviceAndStreamParallelTypes();
  }

  // Collect any DID or stream parallel types seen on the loop domain of the
  // given tvs. For DID, only non-reduction dimensions are considered.
  // For e.g., if tv is i0, r{DIDx(d)}, r{i1/d}], then we allow parallelizing
  // i0 with DIDx. This generates the reduce-scatter communication (`DIDx(d),
  // i0/d, r{DIDx(d)}, r{i1/d}`). where i0 is scattered and i1 is reduced.
  std::unordered_set<ParallelType> existing_parallel_types;
  for (IterDomain* id : tv->getLoopDomain()) {
    if (id->isStream() || (id->isDeviceDim() && !id->isReduction())) {
      existing_parallel_types.insert(id->getParallelType());
    }
  }
  std::unordered_set<ParallelType> selected_parallel_types;
  for (ParallelType pt : all_parallel_types) {
    if (!existing_parallel_types.count(pt)) {
      selected_parallel_types.insert(pt);
    }
  }
  return selected_parallel_types;
}
} // namespace

// Propagates device / stream splits and parallelizations from user annotated
// tensorviews to other tensorviews in the fusion. User annotated tensorviews
// (tvs with device mesh) are not modified in this pass. The only exception is
// fusion inputs which are allowed to be stream parallelized. This does not
// have any effect on actual computation (since fusion inputs do not have a
// producer) but may allow for easier analysis.
//
// The pass has two phases:
// 1. Forward propagating across each expression from inputs to outputs that
// don't have a mesh.
// 2. Back propagating from outputs to inputs. The pass attempts to propagate
// any device or stream parallelization if not already present. For fusion
// inputs, only stream parallelization is propagated.
// (See `MultiDevicePresegPassesTest.ResidualAdd`).

void PropagateShardingsPass::runPass(Fusion* fusion) {
  // Any tensorview with a device mesh is considered scheduled by user and not
  // modified in this pass.
  auto user_sharded_tvs = [&]() {
    const auto all_tvs = fusion->allTvs();
    auto filtered = fusion->allTvs() |
        std::views::filter(std::mem_fn(&TensorView::hasDeviceMesh));
    return std::unordered_set<TensorView*>(filtered.begin(), filtered.end());
  }();

  const std::vector<Expr*>& exprs = fusion->exprs();

  for (Expr* expr : exprs) {
    const auto& reference_inputs = getOrderedReferenceInputs(expr);
    // Propagate shardings from reference inputs in order.
    for (auto* ref_input : reference_inputs) {
      NVF_ERROR(ref_input != nullptr);

      // Consider out [M, N] = linear (inp [M, K], weight (N,
      // K)) with inp sharded on M ([DIDx(d), M/d, K]) and weight sharded on N
      // ([DIDy(d), N/d, K]). We propagate from weights first, so the output
      // will be [M, DIDx(d), N/d]. When we propagate from inp next, we should
      // not propagate DIDx parallel type to the output. Otherwise, the output
      // will have multiple DIDx shardings which is invalid.
      for (TensorView* target :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        if (user_sharded_tvs.count(target)) {
          continue;
        }
        std::unordered_set<ParallelType> selected_parallel_types =
            getParallelTypesToPropagate(target);
        shardLoopLike(
            /*ref=*/ref_input,
            /*tv=*/target,
            selected_parallel_types,
            PropagateDirection::kForward);
      }
    }
  }

  // Back-propagate device meshes. This makes sure all TensorViews have a mesh
  // if any of them has one. This is needed in addition to the forward
  // propagation for ops that don't take any TensorView operands, e.g.,
  // `uniform` used in dropout. See MultiDeviceTest.BackpropMeshes for an
  // example. For non-fusion inputs, we also propagate shardings from outputs to
  // inputs. See MultiDevicePresegPassesTest.ResidualAdd for an example.
  for (Expr* expr : exprs | std::views::reverse) {
    const auto& outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    if (outputs.empty()) {
      continue;
    }
    // All outputs of an expression (Welford, SDPA) should be uniformly sharded.
    // We pick the most parallel output as the reference.
    // This is to avoid picking seed/offset tvs in SDPA.
    std::vector<TensorView*> sorted_outputs = sortTvsByParallelDims(outputs);
    TensorView* ref_output = sorted_outputs.front();

    for (auto* target : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // Allow inputs to be stream parallelized for easier analysis.
      if (user_sharded_tvs.count(target) && !target->isFusionInput()) {
        continue;
      }
      std::unordered_set<ParallelType> selected_parallel_types =
          getParallelTypesToPropagate(target);
      shardLoopLike(
          /*ref=*/ref_output,
          /*tv=*/target,
          selected_parallel_types,
          PropagateDirection::kBackward);
    }
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << std::endl
            << "Fusion Transforms after " << name() << ":" << std::endl;
    fusion->printTransforms();
    debug() << std::endl;
  }
}

} // namespace nvfuser::preseg_passes

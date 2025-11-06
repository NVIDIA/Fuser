// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/finalize_multidevice_domains.h>

#include <fusion.h>
#include <ir/allocation_utils.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <linked_hash_map.h>
#include <multidevice/allocation_utils.h>
#include <multidevice/utils.h>
#include <scheduler/utils.h>
#include <type.h>

namespace nvfuser::preseg_passes {

namespace {

// Validates meshes (i.e. all TensorViews have a device mesh or none) and
// returns true if any TensorView has a device mesh.
bool validateMeshes(Fusion* fusion) {
  // Validate that meshes are assigned to all TensorViews or none.
  bool tv_with_mesh_found = false;
  bool tv_without_mesh_found = false;

  for (auto tv : fusion->allTvs()) {
    if (tv->isCpuScalar()) {
      continue;
    }
    tv->hasDeviceMesh() ? tv_with_mesh_found = true
                        : tv_without_mesh_found = true;
  }
  NVF_CHECK(
      !(tv_with_mesh_found && tv_without_mesh_found),
      "Cannot have some TensorViews with device mesh and some without.");
  return tv_with_mesh_found;
}

bool shouldParallelizeAllocationOnStream(TensorView* tv) {
  if (tv->isFusionInput() || tv->isFusionOutput()) {
    return false;
  }
  for (Expr* use_of_tv : tv->uses()) {
    for (TensorView* output :
         ir_utils::filterByType<TensorView>(use_of_tv->outputs())) {
      if (haveDifferentShardings(tv, output, {ParallelType::Stream})) {
        return false;
      }
    }
  }
  return true;
}

bool isLoopStreamParallelized(const TensorView* tv) {
  return std::any_of(
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      [](IterDomain* id) { return id->isStream(); });
}

// Split the allocation domain of a TensorView when it has device or stream
// parallelization. Device parallelization always propagates to the allocation
// domain. Stream parallelization propagates only if the tensor is allocated
// inside a for loop.
void shardAllocation(TensorView* tv) {
  if (!isLoopStreamParallelized(tv) && !tv->hasDeviceMesh()) {
    // This is required for tests such as `LayoutOpTest.SchedulerKernel` The
    // tensorview has allocation domain disjoint from logical domain. This will
    // currently cause errors when setting allocation domain.
    return;
  }

  std::unordered_set<ParallelType> parallel_types(
      kParallelTypeDIDs.begin(), kParallelTypeDIDs.end());
  if (shouldParallelizeAllocationOnStream(tv)) {
    parallel_types.insert(ParallelType::Stream);
  }
  shardAllocationAsLoop(tv, parallel_types);
}

} // namespace

void FinalizeMultideviceDomainsPass::runPass(Fusion* fusion) {
  validateMeshes(fusion);

  for (Expr* expr : fusion->exprs()) {
    auto inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    auto outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    for (auto* tv : inputs) {
      // Only set loop and allocation domain for fusion inputs.
      // Other tvs would already have been processed as outputs of their
      // definitions. This avoids processing the same tv multiple times.
      if (tv->isFusionInput()) {
        shardAllocation(tv);
        reorderParallelizedToFront(tv);
      }
    }
    for (auto* tv : outputs) {
      shardAllocation(tv);
      reorderParallelizedToFront(tv);
    }

    if (isResharding(expr)) {
      auto check_contiguity = [&](const auto& tvs) {
        return std::all_of(tvs.begin(), tvs.end(), isTvContiguous);
      };
      NVF_CHECK(
          check_contiguity(inputs) && check_contiguity(outputs),
          "Resharding expression must have contiguous inputs and outputs: ",
          expr);
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

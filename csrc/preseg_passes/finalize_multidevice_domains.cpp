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
#include <multidevice/utils.h>
#include <scheduler/utils.h>

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

bool isAllocationParallelized(TensorView* tv, Split* split) {
  NVF_CHECK(
      split->outer()->isDeviceDim() || split->outer()->isStream(),
      "Expected the outer dimension to be a device or stream dimension: ",
      split);
  if (split->outer()->isDeviceDim()) {
    return true;
  }
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

// Splits the allocation domain of a TensorView if it is device or stream
// parallelized Device parallelization is always propagated to the allocation
// domain Stream parallelization is propagated to the allocation domain if it is
// allocated inside a for loop
void setShardedAllocationDomain(TensorView* tv) {
  if (!isStreamParallelized(tv) || !tv->hasDeviceMesh()) {
    // This is required for tests such as
    // `NVFP4QuantizeTest.SwizzledOuputAndWithoutPerTensorAmax` The test has a
    // split allocation domain but no stream/device parallelization. The loop
    // domain is first split to set the allocation domain and then merged back
    // to the logical domain. This will currently fail with the following
    // implementation.
    return;
  }
  LinkedHashMap<IterDomain*, std::optional<bool>> allocation_to_contiguity;
  for (const auto&& [id, contiguity] :
       zip(tv->getMaybeAllocationDomain(), tv->getContiguity())) {
    allocation_to_contiguity.pushBack(id, contiguity);
  }

  // Allocation domain should be a permutation of logical domain at this point.
  std::vector<Expr*> transform_exprs = DependencyCheck::getAllExprsBetween(
      {tv->getMaybeAllocationDomain().begin(),
       tv->getMaybeAllocationDomain().end()},
      {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});

  for (auto* expr : transform_exprs) {
    auto* split = dynamic_cast<Split*>(expr);
    NVF_ERROR(
        split != nullptr,
        "Expected all transform exprs to be a split between allocation and "
        "loop domain during sharding propagation.");

    if (!isAllocationParallelized(tv, split)) {
      continue;
    }
    const auto [contiguity, split_i] =
        allocation_to_contiguity.erase(split->in());
    auto [outer_contiguity, inner_contiguity] = splitContiguity(contiguity);
    allocation_to_contiguity.insert(split_i, split->outer(), outer_contiguity);
    allocation_to_contiguity.insert(split_i, split->inner(), inner_contiguity);
  }

  std::vector<IterDomain*> new_allocation_domain;
  std::vector<std::optional<bool>> new_contiguity;
  new_allocation_domain.reserve(allocation_to_contiguity.size());
  new_contiguity.reserve(allocation_to_contiguity.size());

  for (auto&& [id, contiguity] : allocation_to_contiguity) {
    new_allocation_domain.push_back(id);
    new_contiguity.push_back(contiguity);
  }
  tv->setAllocationDomain(new_allocation_domain, new_contiguity);
}
} // namespace

void FinalizeMultideviceDomainsPass::runPass(Fusion* fusion) {
  validateMeshes(fusion);

  for (Expr* expr : fusion->exprs()) {
    auto inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    auto outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    for (auto tv : inputs) {
      // Only set loop and allocation domain for fusion inputs.
      // Other tvs would already have been processed as outputs of their
      // definitions. This avoids processing the same tv multiple times.
      if (tv->isFusionInput()) {
        setShardedAllocationDomain(tv);
        reorderParallelizedToFront(tv);
      }
    }
    for (auto tv : outputs) {
      setShardedAllocationDomain(tv);
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
}

} // namespace nvfuser::preseg_passes

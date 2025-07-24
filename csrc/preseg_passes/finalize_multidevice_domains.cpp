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

// Reorders the loop domain in the same relative order as the allocation domain.
// Specifically:
// 1. It uses the exprs between logical and loop domain to split the allocation
// domain
// 2. It reorders the loop domain to match the split allocation domain.
// 3. It computes the contiguity of the transformed allocation domain through
// the split exprs.
// 4. Sets the allocation domain to be the same as the loop domain with the
// computed contiguity. This preserves both the sharding and any stride order.
// 5. For non-resharding expressions, it moves the DIDx to the front of the
// loop/allocation domain.
// Note: Ideally, the loop domain can follow the logical domain and the
// allocation domain can follow the stride order specified/inferred. However, we
// currently require loop domain to be the same as allocation domain. This
// behavior will be modified in the future with allocation and loop domain being
// propagated independently.
void setLoopAndAllocationDomain(TensorView* tv, bool is_resharding) {
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
    Split* split = dynamic_cast<Split*>(expr);
    NVF_ERROR(
        split != nullptr,
        "Expected all transform exprs to be a split between allocation and "
        "loop domain during sharding propagation.");
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

  std::optional<std::vector<int64_t>> permutation =
      ir_utils::computePermutation(new_allocation_domain, tv->getLoopDomain());
  NVF_ERROR(
      permutation.has_value(),
      "Failed to find a valid permutation for reordering ",
      tv->getLoopDomain(),
      " as ",
      new_allocation_domain);
  tv->reorder(permutation.value());

  if (is_resharding) {
    // Resharding expressions have specific requirements on position of
    // gathered/scattered dimensions in the allocation domain that is ensured
    // by ReorderShardedAxisPass. So we do not move the DIDx to the front in
    // this case. For example, in reduce-scatter, the scattered axis is the
    // outer-most dimension in communication input and output.
    tv->setAllocationDomain(tv->getLoopDomain(), new_contiguity);
    return;
  }

  // Most schedulers require DIDx to be at the front of the loop domain.
  auto old2new = reorderDIDToFront(tv);
  auto new2old = ir_utils::normalizeOld2New(old2new, tv->nDims());
  std::vector<std::optional<bool>> reordered_contiguity;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::back_inserter(reordered_contiguity),
      [&new_contiguity](int64_t i) -> std::optional<bool> {
        return new_contiguity[i];
      });
  tv->setAllocationDomain(tv->getLoopDomain(), reordered_contiguity);
}

} // namespace

void FinalizeMultideviceDomainsPass::runPass(Fusion* fusion) {
  bool has_mesh = validateMeshes(fusion);
  if (!has_mesh) {
    return;
  }

  for (Expr* expr : fusion->exprs()) {
    auto inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    auto outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    bool is_resharding = isResharding(expr);
    for (auto tv : inputs) {
      // Only set loop and allocation domain for fusion inputs.
      // Other tvs would already have been processed as outputs of their
      // definitions. This avoids processing the same tv multiple times.
      if (tv->isFusionInput()) {
        setLoopAndAllocationDomain(tv, is_resharding);
        ;
      }
    }
    for (auto tv : outputs) {
      setLoopAndAllocationDomain(tv, is_resharding);
    }

    if (is_resharding) {
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

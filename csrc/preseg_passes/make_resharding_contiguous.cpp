// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/make_resharding_contiguous.h>

#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
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

// Transform the maybe allocation domain to the loop domain.
// using exprs between logical and loop and get the permutation required to
// reorder the loop domain in the same relative order as the allocation domain.
// Returns the contiguity of the transformed allocation domain.
std::vector<std::optional<bool>> reorderLoopAsAllocation(TensorView* tv) {
  auto alloc_dom = tv->getMaybeAllocationDomain();
  auto contiguity = tv->getContiguity();

  auto splitContiguity = [](std::optional<bool> contiguity)
      -> std::pair<std::optional<bool>, std::optional<bool>> {
    if (!contiguity.has_value()) {
      return std::make_pair(std::nullopt, std::nullopt);
    }
    if (contiguity.value()) {
      return std::make_pair(true, true);
    }
    return std::make_pair(true, false);
  };

  // Allocation domain should be a permutation of logical domain at this point.
  std::vector<Expr*> transform_exprs = DependencyCheck::getAllExprsBetween(
      {alloc_dom.begin(), alloc_dom.end()},
      {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});

  NVF_ERROR(
      std::all_of(
          transform_exprs.begin(),
          transform_exprs.end(),
          [](Expr* expr) { return expr->isA<Split>(); }),
      "Expected all transform exprs to be a split between logical and loop domain during sharding propagation.");

  for (auto* expr : transform_exprs) {
    Split* split = dynamic_cast<Split*>(expr);
    auto find_it = std::find(alloc_dom.begin(), alloc_dom.end(), split->in());
    NVF_ERROR(
        find_it != alloc_dom.end(),
        "Split input ",
        split->in()->toString(),
        " not found in given ids: ",
        alloc_dom);

    auto pos = std::distance(alloc_dom.begin(), find_it);
    auto [outer_contiguity, inner_contiguity] =
        splitContiguity(contiguity.at(pos));

    alloc_dom[pos] = split->inner();
    alloc_dom.insert(alloc_dom.begin() + pos, split->outer());

    contiguity[pos] = inner_contiguity;
    contiguity.insert(contiguity.begin() + pos, outer_contiguity);
  }

  std::optional<std::vector<int64_t>> permutation =
      ir_utils::computePermutation(alloc_dom, tv->getLoopDomain());
  NVF_ERROR(
      permutation.has_value(),
      "Failed to find a valid permutation for reordering",
      tv->getLoopDomain(),
      " as ",
      alloc_dom);
  tv->reorder(permutation.value());

  return contiguity;
}

bool isTvContiguous(TensorView* tv) {
  return std::all_of(
      tv->getContiguity().begin(),
      tv->getContiguity().end(),
      [](const std::optional<bool>& c) { return c.value_or(true); });
}

template <typename Range>
void setShardedAllocationDomain(Range tvs) {
  for (auto tv : tvs) {
    auto contiguity = reorderLoopAsAllocation(tv);
    tv->setAllocationDomain(tv->getLoopDomain(), contiguity);
  }
}

} // namespace

void MakeReshardingContiguousPass::runPass(Fusion* fusion) {
  bool has_mesh = validateMeshes(fusion);
  if (!has_mesh) {
    return;
  }

  for (Expr* expr : fusion->exprs()) {
    auto inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    auto outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    if (isResharding(expr)) {
      NVF_CHECK(
          std::all_of(
              inputs.begin(),
              inputs.end(),
              [](TensorView* tv) { return isTvContiguous(tv); }),
          "Resharding expression inputs must be contiguous: ",
          expr);
    }

    setShardedAllocationDomain(inputs);
    setShardedAllocationDomain(outputs);
  }
}

} // namespace nvfuser::preseg_passes

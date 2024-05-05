// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <preseg_passes/allocation_order_inference.h>
#include <root_domain_map.h>

namespace nvfuser::preseg_passes {

namespace {

// counting the number of non-broadcast & non-reduction iter domains in tv's
// allocation domain.
size_t countLoopIterDomains(const TensorView* tv) {
  return std::count_if(
      tv->getMaybeAllocationDomain().begin(),
      tv->getMaybeAllocationDomain().end(),
      [&](auto ptr_id) {
        return !ptr_id->isBroadcast() && !ptr_id->isReduction();
      });
}

// Note [ Allocation Order Mapping ]
//
// Map allocation domain from ref to target's rfactor domain to construct a new
// allocation domain for target. The objective is to have target in a similar
// memory format as with ref.
//
// The propagation rule explained in an example, given inputs:
//   ref's allocation domain
//     {iS0[i0], ir1[i1], iS2[i2]}
//   target's rfactor domain
//     {iS3[i3], iS4[i4], ir5[i1], iS6[i5], iS7[i2], ir8[1]}
//
// 1. we project iter domains from targets' rfactor domain which has an exact
// map to ref's allocation domain.
//   mapped_id_vec {ir5[i1], iS7[i2]}
// 2. remove all projected ids and reduction iter domains from target's rfactor
// domain:
//   unmapped_ids_vec {iS3[i3], iS4[i4], iS6[i5], ir8[1]}
// 3. append mapped_id_vec at the end of unmapped_id_vec.
//   target_alloc_domain
//   {iS3[i3], iS4[i4], iS6[i5], ir8[1], ir5[i1], iS7[i2]}
void AllocationOrderMapping(
    const IdModel& id_model,
    TensorView* ref,
    TensorView* target) {
  const DisjointSets<Val*>& val_sets =
      id_model.idGraph(IdMappingMode::EXACT).disjointValSets();

  std::vector<IterDomain*> ref_alloc_domain = ref->getMaybeAllocationDomain();
  const std::vector<IterDomain*>& target_rfactor_domain =
      target->getMaybeRFactorDomain();

  // map target rfactor domain into ref's allocation domain
  std::vector<IterDomain*> mapped_id_vec;
  std::unordered_set<IterDomain*> mapped_id_set;
  for (auto* ref_id : ref_alloc_domain) {
    for (auto* id : target_rfactor_domain) {
      // how do we resolve multiple mapping?
      if (val_sets.permissiveAreMapped(ref_id, id)) {
        mapped_id_vec.push_back(id);
        mapped_id_set.insert(id);
        break;
      }
    }
  }

  // removing mapped ids and reduction ids to create unmapped_ids_vec.
  std::vector<IterDomain*> target_alloc_domain = target_rfactor_domain;
  auto unmapped_ids_vec_end = std::remove_if(
      target_alloc_domain.begin(),
      target_alloc_domain.end(),
      [&mapped_id_set](IterDomain* it) {
        return mapped_id_set.count(it) != 0;
      });
  std::copy(mapped_id_vec.begin(), mapped_id_vec.end(), unmapped_ids_vec_end);

  // skip trivial allocation domain
  if (target_alloc_domain != target_rfactor_domain) {
    target->setAllocationDomain(target_alloc_domain, true);
  }
}

} // namespace

// Note [ Allocation Order Propagation ]
//
// The propagation tries to populate allocation domain from srcs to dsts.
//
// For each TensorView in dsts, it iterate through all TensorView in srcs
// looking for a reference TensorView to propagate its allocation domain.
//   1. It only propagate to TensorView in dsts when it's safe to manipulate its
//   allocation domain:
//     1.1 It doesn't have an allocation domain set;
//     1.2 It is not an aliase to another TensorView;
//     1.3 It does not have self mapping;
//   2. Among all entries in srcs, we pick reference that:
//     2.1 It has a dependency towards dst;
//     2.2 It has the highest count of loop (non-broadcast/non-reduction) iter
//     domains in allocation domain.
//         Note0: The reason to count behind this is that, we could have binary
//         operation on a full-sized tensor with a broadcast vector tensor. In
//         which case, we would want to propagate the layout of the full-sized
//         tensor to the output, even though both candidates have the same rank.
//         Note1: when we have multiple candidates with the same count of loop
//         iter domains, we require there's no ambiguity by checking both
//         candidates having the same iter domain mapping. Otherwise we'll stop
//         the propagation.
//     2.3 It does not have self mapping;
//   3. Propagate memory format from selected reference in `srcs` to its
//   corresponding target in `dsts`.
//
// propagation rule:
//   Given a reference TensorView `ref` and a target TensorView `target`, we try
//   to map iter domain in `ref->getMaybeAllocationDomain()` to
//   `target->getMaybeRFactorDomain()`, which would gives `target` to a similar
//   memory layout as `ref`. For details on the propagation rule see Note [
//   Allocation Order Mapping ]
void inferenceAllocationOrder(
    Fusion* fusion,
    const std::vector<TensorView*>& srcs,
    const std::vector<TensorView*>& dsts) {
  // build IdModel, setting allow_self_mapping to avoid assert
  // even though we do NOT populate allocation order where self_mapping is
  // present
  auto id_model =
      IdModel(fusion, /*build_graphs=*/true, /*allow_self_mapping=*/true);
  const auto& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const auto& val_sets = exact_graph.disjointValSets();

  // populate the number of loop iter domains on srcs
  std::vector<std::pair<TensorView*, size_t>> loop_iter_count;
  for (auto* tv : srcs) {
    // skip entry with self mapping.
    if (!hasSelfMapping(tv, exact_graph).has_value()) {
      loop_iter_count.emplace_back(tv, countLoopIterDomains(tv));
    }
  }

  // propagate new allocation domain on dsts
  for (TensorView* dst : dsts) {
    // safe check when allocation domain on the entry cannot be safely mutated.
    if (dst == nullptr || dst->hasAllocation() ||
        fusion->getOutputAlias(dst).type != AllocationType::New) {
      continue;
    }

    // skip entry with self mapping.
    if (hasSelfMapping(dst, exact_graph).has_value()) {
      continue;
    }

    // find a ref among srcs to be propagated to given dst
    TensorView* ref = nullptr;

    // high water mark for candidate of ref.
    size_t non_bc_high_water_mark = 0;
    for (const auto& iter : loop_iter_count) {
      // discard srcs for propagation which dst has no dependency on.
      if (!DependencyCheck::isDependencyOf(iter.first, dst)) {
        continue;
      }
      // discard srcs with lower iterdomain count than ref
      if (iter.second < non_bc_high_water_mark) {
        // TODO: if loop_iter_count is sorted, we can early return here.
        continue;
      }

      // new candidate found, update ref and high water mark
      if (iter.second > non_bc_high_water_mark) {
        ref = iter.first;
        non_bc_high_water_mark = iter.second;
      }

      // found multiple candidate with the same iterdomain count
      if (iter.second == non_bc_high_water_mark && ref != nullptr) {
        // ensure that there's no ambiguity on permutation mapping from multiple
        // references. we need both ref candidates to have the same mapping on
        // allocation domain
        for (auto i : c10::irange(ref->nDims())) {
          if (!val_sets.permissiveAreMapped(
                  ref->getMaybeAllocationDomain()[i],
                  iter.first->getMaybeAllocationDomain()[i])) {
            // reset ref to nullptr, while keeping the iterdomain count high
            // water mark. No propagatoin will occur unless we found another ref
            // candidate with a higher iterdomain count.
            ref = nullptr;
            break;
          }
        }
        continue;
      }
    }

    // propagate allocation domain if we still have a candidate.
    if (ref) {
      AllocationOrderMapping(id_model, ref, dst);
    }
  }
}

void AllocationDomainPass::runPass(Fusion* fusion) {
  // mark input TensorViews as propagation sources
  auto input_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  std::vector<TensorView*> srcs(input_tvs.begin(), input_tvs.end());
  // mark output TensorViews as propagation destinations
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  std::vector<TensorView*> dsts(output_tvs.begin(), output_tvs.end());
  // propagate allocation domain from sources to destinations
  inferenceAllocationOrder(fusion, srcs, dsts);
}

} // namespace nvfuser::preseg_passes

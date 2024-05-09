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
int64_t countNonTrivialIterDomains(const TensorView* tv) {
  return std::count_if(
      tv->getMaybeAllocationDomain().begin(),
      tv->getMaybeAllocationDomain().end(),
      [&](auto* ptr_id) {
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
// map to ref's allocation domain. (sharp-edge 0: we exclude mapping from
// iteration id on ref to reduction id on target to avoid unnecessary
// re-ordering which exposes #2202).
//   mapped_ids {ir5[i1], iS7[i2]}
// 2. remove all projected ids and reduction iter domains from target's rfactor
// domain:
//   unmapped_ids {iS3[i3], iS4[i4], iS6[i5]}
// 3. iterating through unmodified target's rfactor domain to construct target
// allocation domain:
//   (sharp-edge 1: if target_rfactor_domain[i] is a reduction and is not
//   mapped, we keep the reduction iter domain in the original position.) Push
//   the front of unmapped_id_vec to the end of target allocation domain, if
//   unmapped_id_vec isn't empty yet; Otherwise, push the frnot of mapped_ids at
//   the end of target allocation domain.
//
// Note: we could be using a simplified logic below,
// See issue https://github.com/NVIDIA/Fuser/issues/2202
// 1. we project iter domains from targets' rfactor domain which has an exact
// map to ref's allocation domain.
//   mapped_ids {ir5[i1], iS7[i2]}
// 2. remove all projected iter domains from target's rfactor
// domain:
//   unmapped_ids {iS3[i3], iS4[i4], iS6[i5], ir8[1]}
// 3. append mapped_ids at the end of unmapped_id_vec.
//   target_alloc_domain
//   {iS3[i3], iS4[i4], iS6[i5], ir8[1], ir5[i1], iS7[i2]}
void mapAllocationDomain(
    const IdModel& id_model,
    const TensorView* ref,
    TensorView* target) {
  const DisjointSets<Val*>& val_sets =
      id_model.idGraph(IdMappingMode::EXACT).disjointValSets();

  std::vector<IterDomain*> ref_alloc_domain = ref->getMaybeAllocationDomain();
  const std::vector<IterDomain*>& target_rfactor_domain =
      target->getMaybeRFactorDomain();

  // map target rfactor domain into ref's allocation domain
  nvfuser::VectorOfUniqueEntries<IterDomain*> mapped_ids;

  // logic to preserve reduction iter domain in target to WAR #2202
#if true
  // mapping id between ref's allocation domain to target's rfactor domain
  for (auto* ref_id : ref_alloc_domain) {
    for (auto* id : target_rfactor_domain) {
      // sharp-edges 0
      // avoid mapping a reduced dimension.
      if (!ref_id->isReduction() && id->isReduction()) {
        continue;
      }
      if (val_sets.strictAreMapped(ref_id, id)) {
        mapped_ids.pushBack(id);
        break;
      }
    }
  }

  // removing mapped ids and reduction ids to create unmapped_ids.
  // This means for the rest of ids in target_rfactor_domain that's not in
  // mapped_ids, they are either 1. a reduction domain, or; 2. in
  // [unmapped_ids.begin(), unmapped_ids_vec_end) This ensures that sharp-edges
  // 1's loop would reconstruct a permutation of the target_rfactor_domain,
  // hence a valid allocation domain for target.
  std::vector<IterDomain*> unmapped_ids = target_rfactor_domain;
  auto unmapped_ids_vec_end = std::remove_if(
      unmapped_ids.begin(), unmapped_ids.end(), [&mapped_ids](IterDomain* it) {
        return mapped_ids.has(it) || it->isReduction();
      });

  auto mapped_id_iter = mapped_ids.begin();
  auto unmapped_id_iter = unmapped_ids.begin();
  // initialize new target allocation domain with nullptr
  std::vector<IterDomain*> target_alloc_domain(
      target_rfactor_domain.size(), nullptr);
  for (auto i : c10::irange(target_rfactor_domain.size())) {
    // sharp-edges 1
    // preserves non-mapped reduction id in its original position
    if (target_rfactor_domain[i]->isReduction() &&
        mapped_ids.has(target_rfactor_domain[i])) {
      target_alloc_domain[i] = target_rfactor_domain[i];
      continue;
    }
    // push unmapped ids to outer dimension until it's fully consumed
    if (unmapped_id_iter != unmapped_ids_vec_end) {
      target_alloc_domain[i] = *unmapped_id_iter++;
    } else {
      // push mapped ids to inner dimension
      target_alloc_domain[i] = *mapped_id_iter++;
    }
  }
#else
  // mapping id between ref's allocation domain to target's rfactor domain
  for (auto* ref_id : ref_alloc_domain) {
    for (auto* id : target_rfactor_domain) {
      if (val_sets.permissiveAreMapped(ref_id, id)) {
        mapped_ids.pushBack(id);
        break;
      }
    }
  }
  std::vector<IterDomain*> target_alloc_domain = target_rfactor_domain;
  // removing mapped ids.
  auto unmapped_ids_vec_end = std::remove_if(
      target_alloc_domain.begin(),
      target_alloc_domain.end(),
      [&mapped_ids](IterDomain* it) { return mapped_ids.has(it); });
  // appending mapped ids at the end of target_alloc_domain.
  std::copy(mapped_ids.begin(), mapped_ids.end(), unmapped_ids_vec_end);
#endif

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
//     2.2 It has the highest no. of non-trivial (non-broadcast/non-reduction)
//     iter domains in allocation domain.
//         Note0: The reason to count behind this is that, we could have binary
//         operation on a full-sized tensor with a broadcast vector tensor. In
//         which case, we would want to propagate the layout of the full-sized
//         tensor to the output, even though both candidates have the same rank.
//         Note1: when we have multiple candidates with the same count of
//         non-trivial iter domains, we require there's no ambiguity by
//         checking both candidates having the same iter domain mapping.
//         Otherwise we'll stop the propagation by leaving ref as nullptr.
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
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const DisjointSets<Val*>& val_sets = exact_graph.disjointValSets();

  // populate the number of non-trivial iter domains on srcs
  std::unordered_map<TensorView*, int64_t> non_trivial_iter_count;
  for (auto* tv : srcs) {
    // skip entry with self mapping.
    if (!hasSelfMapping(tv, exact_graph).has_value()) {
      non_trivial_iter_count[tv] = countNonTrivialIterDomains(tv);
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
    int64_t non_bc_high_water_mark = 0;
    for (auto* tv : srcs) {
      // skip when non-trivial iter domain count is missing.
      if (non_trivial_iter_count.count(tv) == 0) {
        continue;
      }
      // discard srcs for propagation which dst has no dependency on.
      if (!DependencyCheck::isDependencyOf(tv, dst)) {
        continue;
      }
      // discard srcs with lower iterdomain count than ref.
      if (non_trivial_iter_count[tv] < non_bc_high_water_mark) {
        continue;
      }
      // new candidate found, update ref and high water mark.
      if (non_trivial_iter_count[tv] > non_bc_high_water_mark) {
        ref = tv;
        non_bc_high_water_mark = non_trivial_iter_count[tv];
        continue;
      }
      // found multiple candidate with the same iterdomain count
      if (non_trivial_iter_count[tv] == non_bc_high_water_mark &&
          ref != nullptr) {
        // ensure that there's no ambiguity on permutation mapping from multiple
        // references. we need both ref candidates to have the same mapping on
        // allocation domain
        for (auto i : c10::irange(ref->nDims())) {
          if (!val_sets.permissiveAreMapped(
                  ref->getMaybeAllocationDomain()[i],
                  tv->getMaybeAllocationDomain()[i])) {
            // reset ref to nullptr, while keeping the iterdomain count high
            // water mark. No propagation will occur unless we found another ref
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
      mapAllocationDomain(id_model, ref, dst);
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

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

// NOTE: [Allocation Order Inference]
//
// AllocationOrderInferencer ctor takes a map of allocation order for inputs as
// `unordered_map<const TensorView*, AllocationOrder>`. It propagates
// AllocationOrder on a fusion and updates the the map with allocation order for
// other TensorView in the fusion.
//
// e.g.
//   std::unordered_map<const TensorView*, AllocationOrder> alloc_order_map;
//   // ... update alloc_order_map with AllocationOrder for tensors
//   //     (i.e. usually inputs)
//
//   // create AllocationOrderInferencer
//   AllocationOrderInferencer infer(alloc_order_map);
//   // propagates AllocationOrder from entries already in alloc_order_map
//   infer.traverse(fusion);
//   // all tensor that's propagated successfully will have their allocation
//   // order in alloc_order_map
//
// The protocol for AllocationOrder in alloc_order_map_ has three states. For
// each `tv`, its corresponding allocation order `alloc_order_map_[tv]`:
// 1. The allocation order has the same size as the `tv`'s rfactor domain;
//    This means it has a preferred allocation order and the entry should
//    participate in propagation.
// 2. The allocation order is an empty array;
//    This means it's a wild card and shouldn't dictate output allocation
//    order. But it marks that propagation is successful for `tv`.
//    i.e. This currently happens for TensorViews that's created by factory
//    methods and its consumers.
// 3. alloc_order_map_ does not have an entry for `tv`.
//    This is the case where propagation has not reach the `tv`, likely due to
//    lack of allocation order on inputs or certain operation not yet supported
//    by propagation rule.
//
// Identify the difference between case 2. and 3. above allows us to better
// handle `resolveAllocationOrder` among multiple candidates.
// i. We do not want to ignore candidates where propagation has failed and
// aggressively propagates allocatoin order through unresolved candidates. So we
// would want to identify case 3. ii. Tensors created by factory methods should
// carry a wild-card and should not actively participate propagation. Because
// those tensors are not going to affect vectorization. Hence we need to
// identify case 2.

// helper function to count the number of non-broadcast & non-reduction
// iterdomains in tv's rfactor domain.
size_t countLoopIterDomains(const TensorView* tv) {
  return std::count_if(
      tv->getMaybeRFactorDomain().begin(),
      tv->getMaybeRFactorDomain().end(),
      [&](auto ptr_id) {
        return !ptr_id->isBroadcast() && !ptr_id->isReduction();
      });
};

// mapping allocation domain from producer to consumer without reduction
//
// e.g.
//   producer rfactor dom [r0', i0', i1', i2'] @ allocation order {0, 1, 3, 2}
//    |       alloc dom [r0', i0', i2', i1']
//    |
//    Operation
//    |
//    v
//   consumer rfactor dom [..., i0, ..., i1, ..., i2, ...]
//
// we construct allocation domain on producer, filtering out reduction, apply
// root domain map from producer to consumer.
//   [r0', i0', i2', i1'] -> [i0', i2', i1'] -> [i0, i2, i1]
// so the function would return [i0, i2, i1]
void replayAllocationDomain(
    const IdModel& id_model,
    TensorView* ref,
    TensorView* target) {
  const DisjointSets<Val*>& val_sets =
      id_model.idGraph(IdMappingMode::EXACT).disjointValSets();

  std::vector<IterDomain*> ref_alloc_domain = ref->getMaybeAllocationDomain();

  std::vector<IterDomain*> mapped_id_vec;
  std::unordered_set<IterDomain*> mapped_id_set;
  for (auto* ref_id : ref_alloc_domain) {
    // maybe not skipping broadcast/reduction domains

    for (auto* id : target->getMaybeRFactorDomain()) {
      // avoid mapping a reduced dimension.
      if (!ref_id->isReduction() && id->isReduction()) {
        // technically we don't need to skip this. But it's giving issues
        continue;
      }
      // skip already map id
      if (mapped_id_set.count(id) != 0) {
        continue;
      }
      // how do we resolve multiple mapping?
      if (val_sets.strictAreMapped(ref_id, id)) {
        mapped_id_vec.push_back(id);
        mapped_id_set.insert(id);
        break;
      }
    }
  }

  // NOTE: preserve reduction iterdomain.
  // we are not mapping rS{} id in outputs to inputs. This causes the pass to
  // aggressively push for permutation on output. Which should be fine since
  // re-ordering reduced id in allocation domain shouldn't matter. But it's
  // hitting failures.
  std::vector<IterDomain*> unmapped_ids_vec = target->getMaybeRFactorDomain();
  // auto iter = std::remove_if(unmapped_ids_vec.begin(),
  // unmapped_ids_vec.end(), [&mapped_id_set](IterDomain* it) {return
  // mapped_id_set.count(it) != 0;}); std::copy(mapped_id_vec.begin(),
  // mapped_id_vec.end(), iter);

  auto iter = std::remove_if(
      unmapped_ids_vec.begin(),
      unmapped_ids_vec.end(),
      [&mapped_id_set](IterDomain* it) {
        return mapped_id_set.count(it) != 0 || it->isReduction();
      });

  auto mapped_id_iter = mapped_id_vec.begin();
  auto unmapped_id_iter = unmapped_ids_vec.begin();
  const std::vector<IterDomain*>& target_rfactor_domain =
      target->getMaybeRFactorDomain();
  std::vector<IterDomain*> target_alloc_domain(
      target_rfactor_domain.size(), nullptr);
  for (auto i : c10::irange(target_rfactor_domain.size())) {
    if (target_rfactor_domain[i]->isReduction() &&
        mapped_id_set.count(target_rfactor_domain[i]) == 0) {
      target_alloc_domain[i] = target_rfactor_domain[i];
      continue;
    }
    if (unmapped_id_iter != iter) {
      target_alloc_domain[i] = *unmapped_id_iter++;
    } else {
      target_alloc_domain[i] = *mapped_id_iter++;
    }
  }

  // skip when it isn't updating.
  if (target_alloc_domain != target_rfactor_domain) {
    target->setAllocationDomain(target_alloc_domain, true);
  }
}

} // namespace

// Note [ Allocation Order Propagation ]
//
// The propagation tries to propagate allocation order from inputs to the entire
// fusion:
//   1. Iterates through all inputs, looking for TensorView with allocation
//   domain that's a permutation of its corresponding rfactor domain and record
//   it as the allocation order of the tensor;
//   2. Traverse the fusion IR, propagate allocation order and record results in
//   alloc_order_map.
void inferenceAllocationOrder(
    Fusion* fusion,
    const std::unordered_set<Val*>& skip_set) {
  // build IdModel, setting allow_self_mapping to avoid assert
  // even though we do NOT populate allocation order where self_mapping is
  // present
  auto id_model =
      IdModel(fusion, /*build_graphs=*/true, /*allow_self_mapping=*/true);
  const auto& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const auto& val_sets = exact_graph.disjointValSets();

  // populate the number of non-broadcast/non-reduction iterdomains on srcs
  std::vector<std::pair<TensorView*, size_t>> loop_iter_count;
  for (auto* tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    // skip entry with self mapping.
    if (!hasSelfMapping(tv, exact_graph).has_value()) {
      loop_iter_count.emplace_back(tv, countLoopIterDomains(tv));
    }
  }

  // propagate new allocation domain on dsts
  for (Val* out_val : fusion->outputs()) {
    if (skip_set.count(out_val) != 0) {
      continue;
    }

    auto* out_tv = dynamic_cast<TensorView*>(out_val);

    // safe check when allocation domain on the entry cannot be safely mutated.
    if (out_tv == nullptr || out_tv->hasAllocation() ||
        fusion->getOutputAlias(out_val).type != AllocationType::New) {
      continue;
    }

    // skip entry with self mapping.
    if (hasSelfMapping(out_tv, exact_graph).has_value()) {
      continue;
    }

    // find a ref among srcs to be propagated to given dst
    TensorView* ref = nullptr;

    // high water mark for candidate of ref.
    size_t non_bc_high_water_mark = 0;
    for (const auto& iter : loop_iter_count) {
      // discard srcs for propagation which dst has no dependency on.
      if (!DependencyCheck::isDependencyOf(iter.first, out_val)) {
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
          if (!val_sets.strictAreMapped(
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
      replayAllocationDomain(id_model, ref, out_tv);
    }
  }
}

void AllocationDomainPass::runPass(Fusion* fusion) {
  inferenceAllocationOrder(fusion);
}

} // namespace nvfuser::preseg_passes

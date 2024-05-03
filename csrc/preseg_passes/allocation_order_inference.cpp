// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <id_model/id_model.h>
#include <iter_visitor.h>
#include <preseg_passes/allocation_order_inference.h>
#include <root_domain_map.h>

namespace nvfuser::preseg_passes {

namespace {

// performs permutation by `alloc_order` on `tv`'s rfactor_domain.
std::vector<IterDomain*> constructAllocationDomain(
    TensorView* tv,
    const AllocationOrder& alloc_order) {
  auto rfactor_dom = tv->getMaybeRFactorDomain();
  auto rank = rfactor_dom.size();

  std::vector<IterDomain*> allocation_domain(rank, nullptr);
  // specify allocation domain with dimension per allocation order.
  for (auto i : c10::irange(rank)) {
    allocation_domain[i] = rfactor_dom.at(alloc_order.at(i));
  }

  return allocation_domain;
}

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




// helper utils to count the number of non broadcast / non reduction
// iterdomain
size_t countLoopIterDomains(const TensorView* tv) {
  return std::count_if(
      tv->getMaybeRFactorDomain().begin(),
      tv->getMaybeRFactorDomain().end(),
      [&](auto ptr_id) {
        return !ptr_id->isBroadcast() && !ptr_id->isReduction();
      });
};

// TODO: update comment
// Returns the candidate operand that dominates the allocation order.
//
// It scans through each candidate to find the first one that:
//   1. is a TensorView
//   2. has the most non_broadcast IterDomains
//
// The function returns a nullptr when it encounters a TensorView that does
// not have an entry in alloc_order_map_, since this means we failed to
// propagate memory format for an entry, we do NOT want to aggressively insert
// output memory format.
//
// The function is used to resolve allocation order propagation for operator
// with multiple operands. The operand with the most number of
// non-broadcast IterDomain will be dominating the output allocation order.
// The motivation behind it to avoid breaking allocation order propagation
// from operands produced by broadcast. e.g. When a binary operator could take
// in a channels_last 4d tensor and an unsqueezed bias vector. We'll want to
// propagate the channels_last allocation order to output.
//
// Pre-condition: `candidates` must be the input operands of the same Expr.
TensorView* findReference(const std::vector<Val*>& candidates) {
}

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
  // // constructing alloc_domain for producer from its root domain, while
  // // filtering out reduction because they won't appear in consumer's domain.
  // std::vector<IterDomain*> alloc_domain = TensorDomain::noReductions(
  //     constructAllocationDomain(producer, alloc_order_map_.at(producer)));
  // // creating producer to consumer root domain map
  // std::unordered_map<IterDomain*, IterDomain*> p2c_map =
  //     PairwiseRootDomainMap(producer, consumer).mapProducerToConsumer();
  // // map alloc_domain to consumer
  // std::transform(
  //     alloc_domain.cbegin(),
  //     alloc_domain.cend(),
  //     alloc_domain.begin(),
  //     [&p2c_map](IterDomain* id) { return p2c_map.at(id); });
  // return alloc_domain;
  const DisjointSets<Val*>& val_sets = id_model.idGraph(IdMappingMode::EXACT).disjointValSets();

  // TODO: I don't think I'm doing it right here.
  std::vector<IterDomain*> ref_alloc_domain = ref->getMaybeAllocationDomain();
  std::vector<IterDomain*> mapped_ids;
  std::unordered_set<IterDomain*> mapped_id;
  for (auto* ref_id : ref_alloc_domain) {
    // maybe not skipping broadcast/reduction domains

    for (auto* id : target->getMaybeRFactorDomain()) {
      // avoid mapping a reduced dimension. 
      if (!ref_id->isReduction() && id->isReduction()) {
        // technically we don't need to skip this. But it's giving issues
        continue;
      }
      // skip already map id
      if (mapped_id.count(id) != 0) {
        continue;
      }
      // how do we resolve multiple mapping?
      if (val_sets.strictAreMapped(ref_id, id)) {
        mapped_ids.push_back(id);
        mapped_id.insert(id);
        break;
      }
    }
  }

  // NOTE: preserve reduction iterdomain.
  // we are not mapping rS{} id in outputs to inputs. This causes the pass to aggressively push for permutation on output. Which should be fine since re-ordering reduced id in allocation domain shouldn't matter. But it's hitting failures.
  std::vector<IterDomain*> target_alloc_domain = target->getMaybeRFactorDomain();
  // auto iter = std::remove_if(target_alloc_domain.begin(), target_alloc_domain.end(), [&mapped_id](IterDomain* it) {return mapped_id.count(it) != 0;});
  // std::copy(mapped_ids.begin(), mapped_ids.end(), iter);

  auto iter = std::remove_if(target_alloc_domain.begin(), target_alloc_domain.end(), [&mapped_id](IterDomain* it) {return mapped_id.count(it) != 0 || it->isReduction();});

  auto mapped_iter = mapped_ids.begin();
  auto unmapped_iter = target_alloc_domain.begin();
  const std::vector<IterDomain*>& alloc_domain = target->getMaybeRFactorDomain();
  std::vector<IterDomain*> new_alloc_domain(alloc_domain.size(), nullptr);
  for (auto i : c10::irange(alloc_domain.size())) {
    if (alloc_domain[i]->isReduction() && mapped_id.count(alloc_domain[i]) == 0) {
      new_alloc_domain[i] = alloc_domain[i];
      continue;
    }
    if (unmapped_iter != iter) {
      new_alloc_domain[i] = *unmapped_iter++;
    } else {
      new_alloc_domain[i] = *mapped_iter++;
    }
  }
  

  // skip when it isn't updating.
  if (new_alloc_domain != target->getMaybeRFactorDomain()) {
    target->setAllocationDomain(new_alloc_domain, true);
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
  // std::unordered_map<const TensorView*, AllocationOrder> alloc_order_map;
  // // Note: we only consider simple permutation of allocation domain to rfactor
  // // domain.
  // for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
  //   std::optional<AllocationOrder> permutation = ir_utils::computePermutation(
  //       TensorDomain::noReductions(tv->getMaybeRFactorDomain()),
  //       TensorDomain::noReductions(tv->getMaybeAllocationDomain()));
  //   if (permutation.has_value()) {
  //     alloc_order_map[tv] = permutation.value();
  //   }
  // }
  //
  // // Initialize AllocationOrderInferencer with allocation order of input tensor
  // // views
  // AllocationOrderInferencer infer(alloc_order_map);
  // infer.traverse(fusion);
  //
  // return the propagated map
  // return alloc_order_map;

  // allow self mapping to avoid assert
  auto id_model = IdModel(fusion, true, true);
  const auto& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const auto& val_sets = exact_graph.disjointValSets();

  // picking a candidate for propagation.
  std::vector<std::pair<TensorView*, size_t>> loop_iter_count;
  for (auto* tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    if (!hasSelfMapping(tv, exact_graph).has_value()) {
      loop_iter_count.emplace_back(tv, countLoopIterDomains(tv));
    }
  }

  // propagating the allocation order through graph
  // option1: a vanilla mapping with `val_sets.strictAreMapped` and only manipulate things that is mapped.
  // option2: wondering if there's something for us to replay a partial map?! i.e. we can replay ref->rfactor --> ref->allocation to tv->rfactor
  for (Val* out_val : fusion->outputs()) {
    if (skip_set.count(out_val) != 0) {
      continue;
    }
    auto* out_tv = dynamic_cast<TensorView*>(out_val);
    if (out_tv == nullptr || out_tv->hasAllocation() ||
        fusion->getOutputAlias(out_val).type != AllocationType::New ||
        hasSelfMapping(out_tv, exact_graph).has_value()) {
      continue;
    }

    TensorView* ref = nullptr;
    // skipping cases where output has iter loop count.
    // size_t non_bc_high_water_mark = countLoopIterDomains(out_tv) - 1;
    size_t non_bc_high_water_mark = 0;
    for (const auto& iter : loop_iter_count) {
      // only consider inputs for propagation when output has dependency on.
      if (DependencyCheck::isDependencyOf(iter.first, out_val)) {
        if (iter.second > non_bc_high_water_mark) {
          // TODO: if loop_iter_count is sorted, we can early return here.
          ref = iter.first;
          non_bc_high_water_mark = iter.second;
	} else if (iter.second == non_bc_high_water_mark && ref != nullptr) {
	  // we need to ensure that there's no ambiguity on permutation mapping from multiple dominating references.
	  for (auto i : c10::irange(ref->nDims())) {
            if (!val_sets.strictAreMapped(ref->getMaybeAllocationDomain()[i], iter.first->getMaybeAllocationDomain()[i])) {
	      ref = nullptr;
	      return;
	    }
	  }
	}
      }
    }
    if (ref) {
      replayAllocationDomain(id_model, ref, out_tv);
    }
  }
}

void AllocationDomainPass::runPass(Fusion* fusion) {
  inferenceAllocationOrder(fusion);
}

} // namespace nvfuser::preseg_passes

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
#include <logical_domain_map.h>
#include <preseg_passes/allocation_order_inference.h>

namespace nvfuser::preseg_passes {

namespace {

// returns non-broadcast & non-reduction iter domains in tv's allocation
// domain.
std::vector<IterDomain*> nonTrivialIterDomains(const TensorView* tv) {
  return TensorDomain::noReductions(
      TensorDomain::noBroadcasts(tv->getMaybeAllocationDomain()));
}

// counts the number of non-broadcast & non-reduction iter domains in tv's
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
// Map allocation domain from ref to target's logical domain to construct a new
// allocation domain for target. The objective is to set target with the similar
// innermost dimensions in storage to facilitate larger vectorization factor.
//
// The propagation rule explained in an example, given inputs:
//   ref's allocation domain
//     {iS9[i5], iS0[i0], iS1[i1], iS2[i2]}
//   target's logical domain
//     {iS3[i3], iS4[i4], iS5[i1], iS6[i5], iS7[i2], ir8[1]}
//
// 1. we project iter domains from refs' allocation domain to targets' logical
// domains, starting from the fast iter domains, until we fail to find an exact
// map for a non trivial iter domain, note: we would skip no mapping for trivial
// iter domains in refs. (sharp-edge 0: we exclude mapping from iteration id on
// ref to reduction id on target to avoid unnecessary re-ordering which exposes
// #2202).
//   we go through iS2[i2] (mapped)
//              -> iS1[i1] (mapped)
//              -> iS0[i0] (break, since there's no mapping)
//   mapped_ids  {iS7[i2], iS5[i1]}
// 2. remove all projected ids and reduction iter domains from target's rfactor
// domain:
//   unmapped_ids {iS3[i3], iS4[i4], iS6[i5]}
// 3. iterating through unmodified target's logical domain to construct target
// allocation domain:
//   (sharp-edge 1: if target_logical_domain[i] is a reduction and is not
//   mapped, we keep the reduction iter domain in the original position.) Push
//   the front of unmapped_id_vec to the end of target allocation domain, if
//   unmapped_id_vec isn't empty yet; Otherwise, push the rfront of mapped_ids
//   at the end of target allocation domain.
//
// Note: we could be using a simplified logic below,
// See issue https://github.com/NVIDIA/Fuser/issues/2202
// 1. we project iter domains from targets' logical domain which has an exact
// map to ref's allocation domain.
//   mapped_ids {iS7[i2], iS5[i1]}
// 2. remove all projected iter domains from target's rfactor
// domain:
//   unmapped_ids {iS3[i3], iS4[i4], iS6[i5], ir8[1]}
// 3. append reversed mapped_ids at the end of unmapped_id_vec.
//   target_alloc_domain
//   {iS3[i3], iS4[i4], iS6[i5], ir8[1], iS5[i1], iS7[i2]}
void mapAllocationDomain(
    const IdModel& id_model,
    const TensorView* ref,
    TensorView* target) {
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

  std::vector<IterDomain*> ref_alloc_domain = ref->getMaybeAllocationDomain();
  // reverse ref_alloc_domain, so a range based loop would iterate through from
  // fast to slow dimensions
  std::reverse(ref_alloc_domain.begin(), ref_alloc_domain.end());
  const std::vector<IterDomain*>& target_logical_domain =
      target->getLogicalDomain();

  // map target logical domain into ref's allocation domain
  nvfuser::VectorOfUniqueEntries<IterDomain*> mapped_ids;

  std::unordered_map<ValGroup, IterDomain*> exact_id_map;
  for (auto* id : target_logical_domain) {
    if (exact_graph.hasGroup(id)) {
      exact_id_map[exact_graph.toGroup(id)] = id;
    }
  }

  // logic to preserve reduction iter domain in target to WAR #2202
#if true
  // mapping id between ref's allocation domain to target's logical domain,
  // iterating from fast to slow loop
  for (auto* ref_id : ref_alloc_domain) {
    // no ValGroup for ref_id to map.
    if (!exact_graph.hasGroup(ref_id)) {
      // no mapping for trivial iter domains is skipped, since it doesn't block
      // vectorization
      if (ref_id->isBroadcast() || ref_id->isReduction()) {
        continue;
      }
      // break when no mapping ValGroup found in target_logical_domain.
      break;
    }
    const ValGroup& vg = exact_graph.toGroup(ref_id);
    // no mapping ValGroup found in target_logical_domain.
    if (exact_id_map.count(vg) == 0) {
      // no mapping for trivial iter domains is skipped, since it doesn't block
      // vectorization
      if (ref_id->isBroadcast() || ref_id->isReduction()) {
        continue;
      }
      // break when no mapping ValGroup found in target_logical_domain.
      break;
    }
    IterDomain* id = exact_id_map[vg];
    // sharp-edges 0
    // avoid mapping a reduced dimension.
    if (!ref_id->isReduction() && id->isReduction()) {
      continue;
    }
    mapped_ids.pushBack(id);
  }
  // Note: empty `mapped_ids` will give us `target_alloc_domain` that's
  // identical to `target_logical_domain`. Hence specifying no allocation domain
  // on target tensor.

  // removing mapped ids and reduction ids to create unmapped_ids.
  // This means for the rest of ids in target_logical_domain that's not in
  // mapped_ids, they are either 1. a reduction domain, or; 2. in
  // [unmapped_ids.begin(), unmapped_ids_vec_end) This ensures that sharp-edges
  // 1's loop would reconstruct a permutation of the target_logical_domain,
  // hence a valid allocation domain for target.
  std::vector<IterDomain*> unmapped_ids = target_logical_domain;
  auto unmapped_ids_vec_end = std::remove_if(
      unmapped_ids.begin(), unmapped_ids.end(), [&mapped_ids](IterDomain* it) {
        return mapped_ids.has(it) || it->isReduction();
      });

  // iterate through reverse order, so the entries iterate from slow to fast
  // dimensions
  auto mapped_id_iter = mapped_ids.rbegin();
  auto unmapped_id_iter = unmapped_ids.begin();
  // initialize new target allocation domain with nullptr
  std::vector<IterDomain*> target_alloc_domain(
      target_logical_domain.size(), nullptr);
  for (auto i : arange(target_logical_domain.size())) {
    // sharp-edges 1
    // preserves non-mapped reduction id in its original position
    if (target_logical_domain[i]->isReduction() &&
        !mapped_ids.has(target_logical_domain[i])) {
      target_alloc_domain[i] = target_logical_domain[i];
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
  // mapping id between ref's allocation domain to target's logical domain,
  // iterating from fast to slow loop
  for (auto* ref_id : ref_alloc_domain) {
    // no ValGroup for ref_id to map.
    if (!exact_graph.hasGroup(ref_id)) {
      // no mapping for trivial iter domains is skipped, since it doesn't block
      // vectorization
      if (ref_id->isBroadcast() || ref_id->isReduction()) {
        continue;
      }
      // break when no mapping ValGroup found in target_logical_domain.
      break;
    }
    const ValGroup& vg = exact_graph.toGroup(ref_id);
    // no mapping ValGroup found in target_logical_domain.
    if (exact_id_map.count(vg) == 0) {
      // no mapping for trivial iter domains is skipped, since it doesn't block
      // vectorization
      if (ref_id->isBroadcast() || ref_id->isReduction()) {
        continue;
      }
      // break when no mapping ValGroup found in target_logical_domain.
      break;
    }
    IterDomain* id = exact_id_map[vg];
    mapped_ids.pushBack(id);
  }
  // Note: empty `mapped_ids` will give us `target_alloc_domain` that's
  // identical to `target_logical_domain`. Hence specifying no allocation domain
  // on target tensor.
  std::vector<IterDomain*> target_alloc_domain = target_logical_domain;
  // removing mapped ids.
  auto unmapped_ids_vec_end = std::remove_if(
      target_alloc_domain.begin(),
      target_alloc_domain.end(),
      [&mapped_ids](IterDomain* it) { return mapped_ids.has(it); });
  // appending reversed mapped ids at the end of target_alloc_domain.
  std::copy(mapped_ids.rbegin(), mapped_ids.rend(), unmapped_ids_vec_end);
#endif

  // skip trivial allocation domain
  if (target_alloc_domain != target_logical_domain) {
    target->setAllocationDomain(target_alloc_domain, true);
  }
}

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
//   `target->getLogicalDomain()`, which would gives `target` similar innermost
//   dimensions as with `ref`. For details on the propagation rule see Note [
//   Allocation Order Mapping ]
void inferAllocationOrder(
    Fusion* fusion,
    const std::vector<TensorView*>& srcs,
    const std::vector<TensorView*>& dsts) {
  // build IdModel, setting allow_self_mapping to avoid assert
  // even though we do NOT populate allocation order where self_mapping is
  // present
  auto id_model =
      IdModel(fusion, /*build_graphs=*/false, /*allow_self_mapping=*/true);
  id_model.buildExactGraph();
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
      // skip when non-trivial iter domains count is missing.
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
        std::vector<IterDomain*> ref_alloc_non_trivial =
            nonTrivialIterDomains(ref);
        std::vector<IterDomain*> tv_alloc_non_trivial =
            nonTrivialIterDomains(tv);
        NVF_ERROR(
            ref_alloc_non_trivial.size() == tv_alloc_non_trivial.size(),
            "candidates of allocation order reference should have identical "
            "non-trivial ID size");
        // ensure that there's no ambiguity on permutation mapping from multiple
        // references. we need both ref candidates to have the same mapping on
        // allocation domain
        for (const auto& [id_ref, id] :
             zip(ref_alloc_non_trivial, tv_alloc_non_trivial)) {
          if (!val_sets.permissiveAreMapped(id_ref, id)) {
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

// Propagate allocation orders from an SDPA's inputs to outputs. This is
// necessary to make an SPDA's allocation domain consistent with the output
// at::Tensor from expression evaluation. Currently, we call ATen to evaluate
// SDPAs so matching their behavior, despite being fragile, is the best
// solution.
class SdpaPropagator : public OptOutConstDispatch {
 public:
  void handle(const SdpaFwdOp* e) override {
    // https://github.com/pytorch/pytorch/blob/0db21a6b23fc6d7ccf6246dfd22f063694996144/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp#L439.
    propagateAllocation(e->query(), e->attn_out());
    // Don't propagate allocation to LSE because it's allocated as [B,H,S]:
    // https://github.com/pytorch/pytorch/blob/0db21a6b23fc6d7ccf6246dfd22f063694996144/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp#L454.
  }
  void handle(const SdpaBwdOp* e) override {
    // https://github.com/pytorch/pytorch/blob/7578a0b26836116fed4daecf2f08ff75a4b2dbea/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp#L904
    propagateAllocation(e->query(), e->grad_query());
    // https://github.com/pytorch/pytorch/blob/7578a0b26836116fed4daecf2f08ff75a4b2dbea/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp#L913
    propagateAllocation(e->key(), e->grad_key());
    // https://github.com/pytorch/pytorch/blob/7578a0b26836116fed4daecf2f08ff75a4b2dbea/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp#L922
    propagateAllocation(e->value(), e->grad_value());
  }

 private:
  // Returns true if propagation succeeded. Nit: the return value is not
  // currently used anywhere. I just tend to use this semantic for functions
  // that may or may not change the IR.  Compared with returning `void`, it is
  // little extra code to maintain and becomes handy when actually needed.
  static bool propagateAllocation(TensorView* in, TensorView* out) {
    if (out->hasAllocation()) {
      return false;
    }

    auto in_order = ir_utils::computePermutation(
        in->getLogicalDomain(), in->getMaybeAllocationDomain());
    if (!in_order.has_value()) {
      return false;
    }

    // It's fragile to unconditionally set contiguity to `true`. In code paths
    // that we care about, ATen allocates outputs using `at::empty_like` which
    // by default produces a *contiguous* tensor of the same stride *order*.
    out->setAllocationDomain(
        ir_utils::applyPermutation(out->getLogicalDomain(), *in_order), true);
    return true;
  }
};

} // namespace

void AllocationDomainPass::runPass(Fusion* fusion) {
  // mark input TensorViews as propagation sources
  auto input_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  std::vector<TensorView*> srcs(input_tvs.begin(), input_tvs.end());
  // mark output TensorViews as propagation destinations
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  std::vector<TensorView*> dsts;
  dsts.reserve(output_tvs.size());
  // TODO: instead of exclusion to propagation, this pass should mark it clear
  // that the propagated allocation order is strictly an optimization hint,
  // rather than a semantic requirement coming from computation definition.
  // Scheduler/segmenter would be able to coordinate and discard optimization
  // hint, but they should respect semantic requirement.
  // see issue: https://github.com/NVIDIA/Fuser/pull/2425
  for (TensorView* output : output_tvs) {
    if (Expr* def = output->definition()) {
      if (def->isOneOf<LinearOp, SdpaFwdOp, SdpaBwdOp, MatmulOp, MmaOp>()) {
        continue;
      }
    }
    dsts.push_back(output);
  }
  // propagate allocation domain from sources to destinations
  inferAllocationOrder(fusion, srcs, dsts);

  SdpaPropagator sdpa_propagator;
  for (Expr* e : fusion->exprs()) {
    sdpa_propagator.dispatch(e);
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << std::endl
            << "Fusion Transforms after " << name() << ":" << std::endl;
    fusion->printTransforms();
  }
}

} // namespace nvfuser::preseg_passes

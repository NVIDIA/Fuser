// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/tools/domain_map.h>
#include <scheduler/utils.h>

namespace nvfuser {
namespace scheduler_tools {

namespace {
// Check if a root ID of a fusion input tensor that is indirectly
// accessed by ops such as torchGather needs to be mapped with
// a reference tensor. Select has a similar effect as squeeze as the
// indexed domain is removed, so the domain does not need to be mapped
// as long as the tensor is a fusion input. Similarly, in index_select
// and torchGather, if the output domain is a broadcast, it does not
// need to be mapped if not resolved.
bool canIgnoreIndexedInputDomainID(
    TensorView* input_tv,
    IterDomain* root_id,
    const ComputeAtMap& ca_map) {
  NVF_ERROR(input_tv->isFusionInput());
  for (auto use : input_tv->uses()) {
    if (auto select = dynamic_cast<SelectOp*>(use)) {
      if (root_id != select->getIndexedID()) {
        return false;
      }
    } else if (auto index_select = dynamic_cast<IndexSelectOp*>(use)) {
      // If the root_id is an indexed ID, and the consumer ID may be a
      // broadcast. In that case, nothing needs to be mapped if the
      // consumer broadcast is not resolved
      if (root_id != index_select->getIndexedID() ||
          !ca_map
               .getConcreteMappedID(
                   index_select->getConsumerOfIndexedID(),
                   IdMappingMode::PERMISSIVE)
               ->isBroadcast()) {
        return false;
      }
    } else if (auto gather = dynamic_cast<GatherOp*>(use)) {
      // TODO: Remove this. Once slice is used for torchGather, this
      // should not be necessary. For now, it is necessary to not
      // break the existing torchGather tests
      if (!gather->exactSizes()) {
        continue;
      }
      // If the root_id is an indexed ID, and the consumer ID may be a
      // broadcast. In that case, nothing needs to be mapped if the
      // consumer broadcast is not resolved
      if (root_id != gather->getIndexedID() ||
          !ca_map
               .getConcreteMappedID(
                   gather->getConsumerOfIndexedID(), IdMappingMode::PERMISSIVE)
               ->isBroadcast()) {
        return false;
      }
    } else if (auto layout = dynamic_cast<PreprocessGroupedMatmulInputSf*>(use)) {
      if (input_tv == layout->inputOffsets() || input_tv == layout->outputOffsets()) {
        continue;
      }
    } else {
      // If the input TV is used by any other ops
      return false;
    }
  }

  return true;
}

// Grab all exact set mappings from consumer to producer domains of
// indexed accesses, e.g., index_select
std::unordered_multimap<
    std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
    std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
getIndexedConsumerToProducerMap(Fusion* fusion, const ComputeAtMap& ca_map) {
  std::unordered_multimap<
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      indexed_id_map;

  for (auto expr : fusion->exprs()) {
    if (auto gather = dynamic_cast<GatherOp*>(expr)) {
      auto p_id = gather->getIndexedID();
      auto c_id = gather->getConsumerOfIndexedID();
      indexed_id_map.emplace(
          ca_map.disjointSetOf(c_id, IdMappingMode::EXACT),
          ca_map.disjointSetOf(p_id, IdMappingMode::EXACT));
    } else if (auto index_select = dynamic_cast<IndexSelectOp*>(expr)) {
      auto p_id = index_select->getIndexedID();
      auto c_id = index_select->getConsumerOfIndexedID();
      indexed_id_map.emplace(
          ca_map.disjointSetOf(c_id, IdMappingMode::EXACT),
          ca_map.disjointSetOf(p_id, IdMappingMode::EXACT));
    } else {
      // Note there's no consumer ID for select. This means we can't
      // just propagate from consumers to indexed producers. It seems
      // it's necessary to schedule producers and consumers separately
      // in those cases.
      continue;
    }
  }

  return indexed_id_map;
}

std::vector<IterDomain*> getSchedulingIds(TensorView* tv) {
  // if (tv->isDefinitionType<PreprocessGroupedMatmulInputSf>() && 
  //     (tv->definition()->as<PreprocessGroupedMatmulInputSf>()->inputOffsets() == tv ||
  //     tv->definition()->as<PreprocessGroupedMatmulInputSf>()->outputOffsets() == tv)) {
  //   return tv->getMaybeRootDomain();
  // }
  return tv->getLogicalDomain();
}

} // namespace

DomainMap::DomainMap(Fusion* fusion) : fusion_(fusion), ca_map_(fusion) {
  tvs_with_rfactor_ = scheduler_utils::getTVsWithNonReductionRFactor(fusion);
}

// Determine if all IterDomains in input are mapped to the given tensor
bool DomainMap::areAllInputIdsMappedTo(TensorView* input_tv, TensorView* tv)
    const {
  // Get concrete IDs for input root or logical domain
  std::unordered_set<IterDomain*> in_concrete_ids;
  for (auto in_id : input_tv->getLogicalDomain()) {
    if (canIgnoreIndexedInputDomainID(input_tv, in_id, ca_map_)) {
      continue;
    }

    // Permissive map is required for the transpose scheduler to support cases
    // like T0[I0, b] + T1[b, I1]
    auto concrete =
        ca_map_.getConcreteMappedID(in_id, IdMappingMode::PERMISSIVE);

    if (!concrete->isBroadcast() && !in_id->isReduction()) {
      in_concrete_ids.insert(concrete);
    }
  }

  // Erase all input concrete IDs mapped to the output domain
  // Ignore unresolved broadcast dimensions
  eraseifInputMappedThroughRootDomainAndIndexing(
      in_concrete_ids, getSchedulingIds(tv));

  return in_concrete_ids.empty();
}

// Note: ideally we would want to check that reference_tv contains all iter
// domains in target_tv, so that transformation applied on reference_tv can be
// propagated to target_tv. But we don't have an easy way to check that. Instead
// of that, this function checks that all source iter domains involved in
// transformation on target_tv is covered by reference_tv. Source iter domains
// of TensorViews are IDs that doesn't have an definition and are producers of
// any IDs on the logical domain of the given TensorView.
//
// ------
//
// e.g 0.
//   T34 [i0, i1]
//   T185 [i0, b2, i1]     = broadcast(T34)
//   T192 [i0, b3(ex), i1] = expand(T185)
//   T198 [i0, b3(ex)*i1]  = reshape(T192)
//   output(T34)
//   output(T198)
//
// if we consider taking T34 as reference_tv. T198 is the target_tv. We can't
// replay T34's transform of merging all the dimensions to T198, since b3(ex)*i1
// can't be reversed. The check in this function would give us T34 with source
// i0, i1; where T198 would have source i0, b3, i1, where b3 isn't contained in
// T34. Hence we'll reject this reference_tv.
//
// ------
//
// e.g 1.
//   T0 [i0, i1]
//   T1 [i2, i0, i1]
//   T2 [i0*i1]      = reshape(T0)
//   T3 [b3, i0, i1] = broadcast(T0)
//   T4 [i2, i0, i1] = add(T1, T3)
//   output(T2)
//   output(T4)
//
// the example above should be able to pick T4 as reference_tv. T2's source i0,
// i1 are both contained by the source of T4, so this example could be scheduled
// as a single fusion.
bool DomainMap::areAllTargetIdsCoveredBy(
    TensorView* target_tv,
    TensorView* reference_tv) const {
  auto get_source_iter_domains = [this](const std::vector<IterDomain*>& ids) {
    // traverse back to collect all disjoint set producer IDs for each ID in the
    // logical domain of tv.
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        all_producer_sets;
    std::for_each(ids.begin(), ids.end(), [&](IterDomain* tv_logical_id) {
      all_producer_sets.pushBack(
          ca_map_.disjointSetOf(tv_logical_id, IdMappingMode::EXACT));
    });
    all_producer_sets.pushBack(
        ca_map_.getAllDisjointSetProducers(all_producer_sets));

    std::vector<IterDomain*> source_ids;
    // filtering all producer IDs with empty definition to get source iter
    // domains
    std::for_each(
        all_producer_sets.vector().begin(),
        all_producer_sets.vector().end(),
        [&source_ids,
         this](const std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>&
                   producer_set_ptr) {
          IterDomain* producer_id = producer_set_ptr->front();
          if (ca_map_.uniqueExactDefinitions(producer_id).empty()) {
            source_ids.push_back(producer_id);
          }
        });
    return source_ids;
  };

  // this contains all source iter domain that's covered by reference_tv, so
  // it's safe for target_tv to have them.
  std::unordered_set<IterDomain*> covered_source_ids;
  for (IterDomain* source_id_ref :
       get_source_iter_domains(getSchedulingIds(reference_tv))) {
    covered_source_ids.insert(source_id_ref);
  }
  // It's safe to have unmapped broadcast IterDomain. There're quite a few tests
  // expecting pointwise scheduler to handle this pattern
  for (IterDomain* id_out : target_tv->getLogicalDomain()) {
    if (id_out->isBroadcast()) {
      NVF_ERROR(
          id_out->definition() == nullptr ||
          id_out->definition()->isA<Resize>());

      // Note that ideally we should also be able to handle merge/split on
      // broadcast IDs, so we should really move this skip inside the loop below
      // `get_source_iter_domains(target_tv->getLogicalDomain())` and skip
      // broadcast source IDs. currently we have the issue that split/merge does
      // not preserve expanded broadcasts, see issue:
      // https://github.com/NVIDIA/Fuser/issues/1126
      covered_source_ids.insert(id_out);
    }
  }
  // Note: there's certain cases where it's safe to have dangling IDs,
  // e.g
  //   T34  [i0, i1]
  //   T185 [i0, b2, i1]     = broadcast(T34)
  //   T192 [i0, b3(ex), i1] = expand(T185)
  // It's safe to propagate T34 to T192, since b3(ex) is not involved in the
  // propagation. But this isn't generally safe. If the above example is changed
  // to e.g
  //   T34  [i0, i1]
  //   T185 [i0, b2, i1]     = broadcast(T34)
  //   T186 [i0, i4, i1]     = ones({i0, i4, i1})
  //   T193 [i0, i4, i1]     = add(T185, T186)
  // It's unsafe to propagate from T34 to T193, see issue
  // https://github.com/NVIDIA/Fuser/issues/3542

  // Check all source iter domain involved in producing target_tv
  for (IterDomain* source_id_out :
       get_source_iter_domains(target_tv->getLogicalDomain())) {
    // NOTE: we use concrete id instead. This allows us to link indirect
    // broadcast. So in the example below:
    //   input T0[
    //   T2[i0, i2*i3] = T0[i0, i2, i3]
    //   T3[i0, i2*i3] = T1[i0, b0] + T2[i0, i2*i3]
    //   T4[i0, i9] = pad(T1[i0, b0])
    // We have i9 in T3
    //     -> source ID b0
    //     -> concrete map to i2*i3
    //     -> source ID from i2*i3 to [i2, i3]
    // So T3 is contained by T2. See test `PointwiseTest.DomainMapPad1`
    auto concrete_id_out =
        ca_map_.getConcreteMappedID(source_id_out, IdMappingMode::PERMISSIVE);

    // After mapping with PERMISSIVE map, `concrete_id_out` might no longer be a
    // source ID. We project to source ID again from concrete_id_out. See test
    // DomainMapBroadcastIssue3653
    // In the example above. `i2*i3` is not a source ID. Hence we needed to go
    // through another projection to source IDs in order to map it to
    // covered_source_ids.
    for (IterDomain* concrete_source_id_out :
         get_source_iter_domains({concrete_id_out})) {
      // if we find any source_id_out that's not contained, it's possible our
      // propagation would fail since transformation involving this iter
      // domain can't be resolved.
      if (!getMappedInputConcreteID(
              covered_source_ids, concrete_source_id_out)) {
        return false;
      }
    }
  }
  return true;
}

// Reference domains must exactly match with the input domains. See
// also PR #661
IterDomain* DomainMap::getMappedInputConcreteID(
    const std::unordered_set<IterDomain*>& in_concrete_ids,
    IterDomain* out_id) const {
  auto in_concrete_id_iter = std::find_if(
      in_concrete_ids.begin(),
      in_concrete_ids.end(),
      [&](IterDomain* in_concrete_id) {
        return ca_map_.areMapped(in_concrete_id, out_id, IdMappingMode::EXACT);
      });
  if (in_concrete_id_iter != in_concrete_ids.end()) {
    return *in_concrete_id_iter;
  } else {
    return nullptr;
  }
}

// Erase input concrete ID if it is mapped to output ID
bool DomainMap::eraseIfMapped(
    std::unordered_set<IterDomain*>& in_concrete_ids,
    IterDomain* out_id) const {
  auto mapped_input_conrete_id =
      getMappedInputConcreteID(in_concrete_ids, out_id);
  if (mapped_input_conrete_id != nullptr) {
    in_concrete_ids.erase(mapped_input_conrete_id);
    return true;
  } else {
    return false;
  }
}

void DomainMap::eraseifInputMappedThroughRootDomainAndIndexing(
    std::unordered_set<IterDomain*>& in_ids,
    const std::vector<IterDomain*>& ids) const {
  // Use ComputeAtMap::getAllDisjointSetProducers to grab all producer
  // IDs through rfactor exprs
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      exact_sets;
  std::for_each(ids.begin(), ids.end(), [&](IterDomain* id) {
    exact_sets.pushBack(ca_map_.disjointSetOf(id, IdMappingMode::EXACT));
  });

  // Traverse through indexed domains.
  const auto indexed_id_multimap =
      getIndexedConsumerToProducerMap(fusion_, ca_map_);

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      all_exact_sets_covered;

  // Back traverses through the exact map and indexed
  // producer-consumer pairs
  for (auto current_sets = exact_sets; !current_sets.empty();) {
    auto producer_sets = ca_map_.getAllDisjointSetProducers(current_sets);
    all_exact_sets_covered.pushBack(producer_sets);

    current_sets.clear();

    // Further traversal if any of the new producer sets is a producer
    // of indexed domains
    for (const auto& producer_set : producer_sets) {
      auto indexed_id_multimap_range =
          indexed_id_multimap.equal_range(producer_set);
      for (auto producer_of_producer_it = indexed_id_multimap_range.first;
           producer_of_producer_it != indexed_id_multimap_range.second;
           ++producer_of_producer_it) {
        if (producer_sets.has(producer_of_producer_it->second)) {
          // Prevent infinite recursion
          continue;
        }
        current_sets.pushBack(producer_of_producer_it->second);
      }
    }
  }

  for (const auto& exact_set_ptr : all_exact_sets_covered) {
    auto exact_concrete_id = ca_map_.getConcreteMappedID(
        exact_set_ptr->front(), IdMappingMode::EXACT);
    eraseIfMapped(in_ids, exact_concrete_id);
  }
}

// Find any id in domain that maps with target id
IterDomain* DomainMap::anyMapped(
    const std::vector<IterDomain*>& domain,
    IterDomain* target) const {
  for (auto id : domain) {
    if (ca_map_.areMapped(id, target, IdMappingMode::EXACT)) {
      return id;
    }
  }
  return nullptr;
}

// Determine if output TensorView is a valid reference tensor for this fusion.
// The reference tensor must map to all the iterDomains in each input and
// output
bool DomainMap::isValidReference(TensorView* tv, bool check_inputs) const {
  if (check_inputs) {
    for (auto input_tv :
         ir_utils::filterByType<TensorView>(fusion_->inputs())) {
      if (input_tv->uses().empty()) {
        continue;
      }
      // TODO: Same backward traversal from tv is done for all input
      // tvs. Consider doing the analysis one for all inputs
      if (!areAllInputIdsMappedTo(input_tv, tv)) {
        return false;
      }
    }
  }
  // The check on outputs are optional, transpose scheduler might propose a
  // secondary reference that only applies to a subset of IO tensors. Ideally
  // we should have a more robust check and consider the IO groups instead of
  // blindly skip outputs.
  for (auto output_tv :
       ir_utils::filterByType<TensorView>(fusion_->outputs())) {
    // no need to check for self.
    if (output_tv == tv) {
      continue;
    }
    if (!areAllTargetIdsCoveredBy(output_tv, tv)) {
      return false;
    }
  }
  return true;
}

TensorView* PointwiseDomainMap::findReferenceTensor(
    int64_t minimum_num_axes) const {
  TensorView* result = nullptr;
  int64_t max_dims = -1;
  for (auto output_tv :
       ir_utils::filterByType<TensorView>(fusion_->outputs())) {
    if (isValidReference(output_tv) &&
        hasMinimumSize(output_tv, minimum_num_axes) &&
        !output_tv->isFusionInput()) {
      int64_t n_dims = scheduler_utils::nLogicalDims(output_tv);
      if (n_dims > max_dims) {
        result = output_tv;
        max_dims = n_dims;
      }
    }
  }
  return result;
}

TensorView* TransposeDomainMap::findReferenceFor(
    const std::vector<TensorView*>& group) const {
  TensorView* result = nullptr;
  int64_t max_dims = -1;
  for (auto tv : group) {
    // since transpose scheduler have different set of reference, we skip IDs
    // coverage check of the reference on outputs of the fusion. Note that
    // this is not ideal, we would want to instead have reference tensor
    // checked against all its target IO tensors.
    // TODO: open an issue for this one. transpose scheduler is not supposed
    // to reuse pointwise_utils::DomainMap::isValidRefrence. This function is
    // too restrictive and doesn't align well with the scheme of transpose
    // scheduler
    if (isValidReference(tv)) {
      int64_t dims = scheduler_utils::nLogicalDims(tv);
      if (dims > max_dims) {
        result = tv;
        max_dims = dims;
      }
    }
  }
  return result;
}

IterDomain* TransposeDomainMap::getMappedAllocDimIn(
    TensorView* tv,
    IterDomain* root_dim) const {
  // Find the id mapped to `Allocation Domain`
  const auto& alloc_dom = tv->getMaybeAllocationDomain();
  IterDomain* mapped_id = nullptr;
  for (auto i : arange(alloc_dom.size())) {
    if (ca_map_.areMapped(alloc_dom[i], root_dim, IdMappingMode::INNERMOST)) {
      mapped_id = alloc_dom[i];
      break;
    }
  }
  return mapped_id;
}

bool TransposeDomainMap::hasAtLeastTwoValidGroups(Fusion* fusion) {
  FusionGuard fg(fusion);
  TransposeDomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  if (grouped_inputs_outputs.size() < 2) {
    return false;
  }
  auto ref1 = domain_map.findReferenceFor(grouped_inputs_outputs[0]);
  auto ref2 = domain_map.findReferenceFor(grouped_inputs_outputs[1]);
  if (ref1 == nullptr || ref2 == nullptr) {
    return false;
  }
  // reference 1 is the global reference, so it must have dim mapped the
  // innermost dim of both groups
  auto innermost2 = scheduler_utils::innerMostAllocDim(ref2);
  return domain_map.getMappedAllocDimIn(ref1, innermost2) != nullptr;
}

int64_t TransposeDomainMap::getInnerLeafDim(
    TensorView* tv,
    IterDomain* root_dim) const {
  // TODO: ideally we should be mapping to loop domain directly here.
  // However, our current compute at map is constructed before loop domain is
  // transformed. So the mapping here would require a new compute at map to be
  // constructed from the updated fusion. We'll revisit this once our id graph
  // refactor is done.
  auto mapped_id = getMappedAllocDimIn(tv, root_dim);
  NVF_ERROR(
      mapped_id != nullptr,
      "Can not find ID mapped to ",
      root_dim,
      " in tensor ",
      tv);
  std::vector<Expr*> replay_exprs = StmtSort::getExprsBetween(
      {mapped_id}, {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
  // Project the root id to loop id. Similar to projectIdToRFactor.
  for (auto* expr : replay_exprs) {
    if (auto* split = dynamic_cast<Split*>(expr)) {
      if (split->in() == mapped_id) {
        if (split->inner()->extent()->isOneInt() &&
            !split->outer()->extent()->isOneInt()) {
          mapped_id = split->outer();
        } else {
          mapped_id = split->inner();
        }
      }
    } else if (auto* merge = dynamic_cast<Merge*>(expr)) {
      // Merge with size-1 dimension is not supposed to be here, reshape would
      // map this to a squeeze. This is a conservative assert, we can relaxed
      // it and support with mapping it to out.
      NVF_ERROR(
          !merge->inner()->extent()->isOneInt(),
          "merge with size-1 dimension is supposed to be translated to squeeze "
          "by reshape");
      if (merge->inner() == mapped_id) {
        mapped_id = merge->out();
      }
    } else if (auto* resize = dynamic_cast<Resize*>(expr)) {
      if (resize->in() == mapped_id) {
        mapped_id = resize->out();
      }
    }
  }

  // Find the position of the loop id
  const auto& dom = tv->getLoopDomain();
  for (auto i : arange(dom.size())) {
    if (dom[i] == mapped_id) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

std::vector<std::vector<TensorView*>> TransposeDomainMap::
    groupInputsOutputsByInnerDim() const {
  std::vector<std::vector<TensorView*>> groups;
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion_->outputs());
  auto input_tvs = ir_utils::filterByType<TensorView>(fusion_->inputs());
  std::unordered_set<TensorView*> grouped;
  std::array<decltype(input_tvs)*, 2> tv_filtered_groups = {
      &output_tvs, &input_tvs};
  for (auto tv_filtered_group : tv_filtered_groups) {
    for (auto tv : *tv_filtered_group) {
      if (tv->isFusionInput() && tv->uses().empty()) {
        continue;
      }
      if (grouped.count(tv) > 0) {
        continue;
      }
      groups.emplace_back(std::vector<TensorView*>{tv});
      grouped.emplace(tv);
      // We only want to grab the inner-most dimension, because we don't want
      // tensors with different inner-most dimension to be put in the same
      // group. For example, if we have:
      //   T2[i1, i3*i2] = relu(view(transpose(T1[i1, i2, i3])))
      // then we don't want T1 and T2 to be in the same group.
      //
      // But we don't want to check contiguity. For example, if we have:
      //   T1[i1, i2, i3] (contiguous) + T2[i1, i2, i3] (discontiguous)
      // Then we still want to T1 and T2 to be grouped together.
      auto group =
          scheduler_utils::getInputsOutputsWithInnerDim(tv, true, false);
      if (group.empty()) {
        // In case that the inner most dim of tv is not found (for example, tv
        // is a fusion input with only reductions), we just return a null
        // result which will tell the scheduler to reject the fusion
        return {};
      }
      for (auto member_tv : group) {
        if (grouped.count(member_tv) == 0) {
          grouped.emplace(member_tv);
          groups.back().emplace_back(member_tv);
        } else if (member_tv != tv) {
          // Ambiguous grouping. This should only happen at `canSchedule`, so
          // we just return a null result which will tell the scheduler to
          // reject the fusion
          return {};
        }
      }
    }
  }
  std::stable_sort(
      groups.begin(),
      groups.end(),
      [](const std::vector<TensorView*>& v1,
         const std::vector<TensorView*>& v2) { return v1.size() > v2.size(); });
  return groups;
}

IterDomain* TransposeDomainMap::getMappedInputConcreteID(
    const std::unordered_set<IterDomain*>& in_concrete_ids,
    IterDomain* out_id) const {
  auto in_concrete_id_iter = std::find_if(
      in_concrete_ids.begin(),
      in_concrete_ids.end(),
      [&](IterDomain* in_concrete_id) {
        return ca_map_.areMapped(
            in_concrete_id, out_id, IdMappingMode::PERMISSIVE);
      });
  if (in_concrete_id_iter != in_concrete_ids.end()) {
    return *in_concrete_id_iter;
  } else {
    return nullptr;
  }
}

} // namespace scheduler_tools
} // namespace nvfuser

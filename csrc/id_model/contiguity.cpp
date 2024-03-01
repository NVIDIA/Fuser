// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/contiguity.h>
#include <id_model/utils.h>

namespace nvfuser {

std::vector<IterDomain*>::const_iterator OrderedIdGroupInformation::
    findActiveId(IterDomain* id) const {
  NVF_ERROR(id != nullptr);
  auto it = std::find_if(
      active_ids_.begin(),
      active_ids_.end(),
      [&](IterDomain* active_id) -> bool {
        return active_id != nullptr && graph_.disjointValSets().strictAreMapped(active_id, id);
      });
  return it;
}

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>::
    const_iterator
    OrderedIdGroupInformation::findAllocIDs(IterDomain* id) const {
  // This is an ugly workaround. id_to_alloc_ids_ is a map. This
  // should be done as an O(1) lookup
  for (auto it = id_to_alloc_ids_.begin(); it != id_to_alloc_ids_.end(); ++it) {
    if (graph_.disjointValSets().strictAreMapped(it->first, id)) {
      return it;
    }
  }
  return id_to_alloc_ids_.end();
}

bool OrderedIdGroupInformation::
isConsistentlyOrdered(IterDomain* id) const {
  NVF_ERROR(id != nullptr);
  auto it = std::find_if(
      consistently_ordered_ids_.begin(),
      consistently_ordered_ids_.end(),
      [&](IterDomain* consistent_id) -> bool {
        return graph_.disjointValSets().strictAreMapped(consistent_id, id);
      });
  return it != consistently_ordered_ids_.end();
}


void OrderedIdGroupInformation::traverse(const ExprPath& path) {
  for (const auto& [eg, direction] : path) {
    if (direction != ExprDirection::Forward) {
      // Not supported
      continue;
    }
    dispatch(eg->front());
  }
}

namespace {

// Adapted from ContigIDs
class ContigIDGroups {
 public:
  ContigIDGroups(
      const std::vector<IterDomain*>& alloc_domains,
      const std::vector<bool>& contiguity,
      const ExprPath& path_from_alloc,
      const ValGraph& graph,
      const ConcretizedBroadcastDomains& concrete_info,
      bool is_predicate_pass)
      : graph_(graph),
        alloc_domains_(alloc_domains),
        alloc_contiguity_(contiguity),
        concrete_info_(concrete_info),
        is_predicate_pass_(is_predicate_pass),
        consistent_transform_info_(
            std::make_unique<const OrderedIdGroupInformation>(
                OrderedIdGroupInformation::get(
                    alloc_domains,
                    path_from_alloc,
                    graph,
                    concrete_info_))) {
    if (alloc_domains_.empty()) {
      return;
    }

    NVF_ERROR(
        alloc_domains_.size() == alloc_contiguity_.size(),
        "Arguments don't match ",
        alloc_domains_.size(),
        " != ",
        alloc_contiguity_.size());

    for (const auto alloc_domain_i : c10::irange(alloc_domains_.size())) {
      IterDomain* alloc_domain_id = alloc_domains_.at(alloc_domain_i);
      if (alloc_domain_id->isBroadcast()) {
        NVF_ERROR(false, "This should be filtered out: ", alloc_domain_id->toString());
        continue;
      }
      alloc_to_contig_ids_[alloc_domain_id] = graph_.toGroup(alloc_domain_id);
      
      // If a allocation domain has halo, can't use merged domain even if
      // both inputs are contiguous. HaloInfo is also initialized for
      // rfactor root domains, which should just return "zero"
      // RootAxisInfo. This should be safe as no rfactor tensor should
      // need halo.
      auto alloc_contiguity = alloc_contiguity_.at(alloc_domain_i);
#if 0
      NVF_ERROR(
          alloc_domain_id->isReduction() != alloc_contiguity.has_value(),
          "Expecting a reduction because contiguity has no value, get ",
          alloc_domain_id->toString());
#endif
      // Index of merged reductions can always be coalesced, so considering
      // reduction as true contiguity.
      if (alloc_contiguity &&
          alloc_domain_id->getIterType() != IterType::GatherScatter) {
        contig_ids_.emplace(graph_.toGroup(alloc_domain_id));
      }
    }

    for (const auto& [eg, direction] : path_from_alloc) {
      auto expr = eg->front();
      if (auto resize = dynamic_cast<Resize*>(expr)) {
        resize_deps_.emplace(graph_.toGroup(resize->out()));
      } else {
        auto inputs = graph_.inputGroups(eg);
        if (std::any_of(
                inputs.begin(), inputs.end(), [&](const ValGroup& inp) {
                  return resize_deps_.count(inp) > 0;
                })) {
          for (const auto& out : graph_.outputGroups(eg)) {
            resize_deps_.insert(out);
          }
        }
      }

      dispatch(eg, direction);
    }
  }

  void dispatch(const ExprGroup& eg, ExprDirection direction) {
    NVF_ERROR(!eg->empty());
    Expr* expr = eg->front();
    
    if (isInputFinal(expr, direction)) {
      return;
    }

    //  Currently not propagating any contiguity information with
    // swizzles as contiguity is generally not preserved after swizzles.
    // But in follow ups we could gradually add back a few special
    // cases, depending on specific swizzle type and axes.
    
    if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge, direction);
    }
  }

  void handle(Merge* merge, ExprDirection direction);
  
  bool isInputFinal(Expr* expr, ExprDirection direction) const {
    const auto& inputs = direction == ExprDirection::Forward ? expr->inputs() : expr->outputs();
    return std::any_of(inputs.begin(), inputs.end(), [this](Val* inp) -> bool {
      return final_id_groups_.find(graph_.toGroup(inp)) !=
          final_id_groups_.end();
    });
  }

  const std::unordered_set<ValGroup>& contigIDs() const {
    return contig_ids_;
  }

  const std::unordered_map<IterDomain*, ValGroup>& allocToContigIDs() const {
    return alloc_to_contig_ids_;
  }

 private:
  const ValGraph& graph_;
  //! Allocation domains to analyze contiguity
  const std::vector<IterDomain*> alloc_domains_;
  //! Contiguity of alloc_domains_
  const std::vector<bool>& alloc_contiguity_;
  const std::unordered_set<ValGroup> final_id_groups_;
  const ConcretizedBroadcastDomains& concrete_info_;
  const bool is_predicate_pass_;
  std::unique_ptr<const OrderedIdGroupInformation> consistent_transform_info_;

  std::unordered_set<ValGroup> contig_ids_;
  std::unordered_map<IterDomain*, ValGroup> alloc_to_contig_ids_;

  std::unordered_set<ValGroup> resize_deps_;
};

void ContigIDGroups::handle(Merge* merge, ExprDirection direction) {
  // Only forward direction is supported for now
  if (direction != ExprDirection::Forward) {
    VERBOSE() << "Backward merge not supported: " << merge->toString();
    return;
  }

  VERBOSE() << "ContigIDGroups::handle: " << merge->toString();

  const bool is_indexing_pass = !is_predicate_pass_;
  const bool ignore_consistent_ordering = is_predicate_pass_;
  
  // If output is not consistently ordered or doesn't solely consume all
  // allocation domains in its dependencies, then it can't be a contiguously
  // indexable iterdomain.
  if (!(ignore_consistent_ordering ||
        consistent_transform_info_->isConsistentlyOrdered(merge->out()))) {
    return;
  }

  if (!consistent_transform_info_->exclusivelyConsumesAllocs(merge->out())) {
    return;
  }

  // Check allocation domains for contiguity
  auto alloc_ids_it =
      consistent_transform_info_->idToAllocIds().find(merge->out());

  NVF_ERROR(
      alloc_ids_it != consistent_transform_info_->idToAllocIds().end(),
      "\nError in contiguous analysis, merge info doesn't exist for:\n",
      merge->toString(),
      "\nId: ",
      merge->out()->toString());

  VectorOfUniqueEntries<IterDomain*> alloc_ids = alloc_ids_it->second;

  IterDomain* last_alloc = nullptr;
  for (auto alloc_id_i : c10::irange(alloc_domains_.size())) {
    auto alloc_id = alloc_domains_[alloc_id_i];
    if (alloc_id->isBroadcast()) {
      NVF_ERROR(false);
      //NVF_ERROR(!alloc_contiguity_.at(alloc_id_i).has_value());
      continue;
    }
    if (alloc_ids.has(alloc_id)) {
      // ID found, remove it
      alloc_ids.erase(alloc_id);
      // If we're indexing:
      // we could still potentially consider this ID linearly indexable, as we
      // could multiple the index by the last allocation's stride.
      //
      // If we're computing predicates (ignore_consistent_ordering_==true),
      // then we don't have this same constraint, we can just ignore
      // contiguity of the allocations all together.
      auto alloc_contiguity = alloc_contiguity_.at(alloc_id_i);
#if 0
      NVF_ERROR(
          alloc_id->isReduction() != alloc_contiguity.has_value(),
          "Expecting a reduction because contiguity has no value, get ",
          alloc_id->toString());
#endif
      // Index of merged reductions can always be coalesced, so considering
      // reduction as true contiguity.
      if (!alloc_contiguity && is_indexing_pass) {
        if (!alloc_ids.empty()) {
          return;
        }
      }
      last_alloc = alloc_id;
    }
  }

  // TODO
#if 0
  // If there's a non_divisible split in the history of merge->out then it can't
  // be contiguously indexable.
  if (non_divisible_id_info_.dependsOnNonDivisibleSplit(merge->out())) {
    return;
  }
#endif  

  // Don't allow contig indexing after resize as we need traverse back
  // at least to direct outputs of resize ops
  if (resize_deps_.count(graph_.toGroup(merge->out()))) {
    return;
  }

  // All broadcasting
  if (last_alloc == nullptr) {
    return;
  }

  // Now we know merge->out is a contiguously indexable ID

  // Reset alloc_ids
  alloc_ids = alloc_ids_it->second;
  for (auto alloc_id : alloc_ids) {
    alloc_to_contig_ids_[alloc_id] = graph_.toGroup(merge->out());
  }

  // Still necessary?
#if 0
  auto all_within_vals = DependencyCheck::getAllValsBetween(
      {alloc_domain_.begin(), alloc_domain_.end()}, {merge->out()});
  auto all_within_ids = ir_utils::filterByType<IterDomain>(all_within_vals);

  std::unordered_set<IterDomain*> within_id_set(
      all_within_ids.begin(), all_within_ids.end());

  within_id_set.erase(merge->out());
  within_contig_ids_[merge->out()] = within_id_set;
  for (auto id : all_within_ids) {
    contig_ids_.erase(id);
  }
#endif

  VERBOSE() << "Contig merge ouput: " << merge->out()->toString() << std::endl;

  contig_ids_.emplace(graph_.toGroup(merge->out()));
}

} // namespace

std::unordered_map<IterDomain*, ValGroup> getContigDomains(
    const std::vector<IterDomain*>& alloc_domains,
    const std::vector<bool>& contiguity,
    const ExprPath& path_from_alloc,
    const ValGraph& graph,
    const ConcretizedBroadcastDomains& concrete_info,
    bool is_predicate_pass) {

  ContigIDGroups contig_finder(
      alloc_domains, contiguity, path_from_alloc, graph, concrete_info, is_predicate_pass);

  return contig_finder.allocToContigIDs();
}

} // namespace nvfuser

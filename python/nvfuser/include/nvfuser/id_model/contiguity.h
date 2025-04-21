// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <contiguity.h>
#include <id_model/id_model.h>

namespace nvfuser {

// Minimal adaptation of OrderedIdInformation for IdModel. Note that
// the analysis only propagates forward for now.
class OrderedIdGroupInformation : public OrderedIdInformation {
 public:
  // Run the ordering analysis from given allocation domains through
  // a given traversal path
  static OrderedIdGroupInformation get(
      const std::vector<IterDomain*>& alloc_domain,
      const ExprPath<ExprGroup>& path_from_alloc,
      const ValGraph& graph) {
    OrderedIdGroupInformation info(alloc_domain, graph);
    info.traverse(path_from_alloc);
    return info;
  }

  // Traversal is based on the AlmostExact graph, so matching of iter
  // domains also needs to be done with the same graph
  bool isConsistentlyOrdered(IterDomain* id) const override {
    return std::find_if(
               consistently_ordered_ids_.begin(),
               consistently_ordered_ids_.end(),
               [&](IterDomain* consistent_id) -> bool {
                 return graph_.disjointValSets().strictAreMapped(
                     consistent_id, id);
               }) != consistently_ordered_ids_.end();
  }

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>::
      const_iterator
      findAllocIDs(IterDomain* id) const override {
    // This is a little ugly workaround. id_to_alloc_ids_ is a map of
    // iter domains. If it were a map from ValGroup, this lookup
    // should have been O(1)
    return std::find_if(
        id_to_alloc_ids_.begin(),
        id_to_alloc_ids_.end(),
        [&](const auto& kv) -> bool {
          return graph_.disjointValSets().strictAreMapped(kv.first, id);
        });
  }

 protected:
  OrderedIdGroupInformation(
      const std::vector<IterDomain*>& alloc_domain,
      const ValGraph& graph)
      : OrderedIdInformation(alloc_domain), graph_(graph) {
    using_id_graph_ = true;
  }

  // Currently only forward propagation is supported
  void traverse(const ExprPath<ExprGroup>& path_from_alloc) {
    for (const auto& [eg, direction] : path_from_alloc) {
      if (direction == Direction::Backward) {
        // TODO: support Backward prop
        continue;
      }
      dispatch(eg->front());
    }
  }

  std::vector<IterDomain*>::const_iterator findActiveId(
      IterDomain* id) const override {
    NVF_ERROR(id != nullptr);
    auto it = std::find_if(
        active_ids_.begin(),
        active_ids_.end(),
        [&](IterDomain* active_id) -> bool {
          return active_id != nullptr &&
              graph_.disjointValSets().strictAreMapped(active_id, id);
        });
    return it;
  }

 private:
  const ValGraph& graph_;
};

// Adapted from ContigIDs
class ContigIDGroups {
 public:
  ContigIDGroups(
      const std::vector<IterDomain*>& alloc_domains,
      std::vector<bool> contiguity,
      const ExprPath<ExprGroup>& path_from_alloc,
      const ValGraph& graph,
      bool is_predicate_pass);

  void dispatch(const ExprGroup& eg, Direction direction) {
    NVF_ERROR(!eg->empty());
    Expr* expr = eg->front();

    // Currently not propagating any contiguity information with
    // swizzles as contiguity is generally not preserved after swizzles.
    // But in follow ups we could gradually add back a few special
    // cases, depending on specific swizzle type and axes.

    if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge, direction);
    } else if (auto split = dynamic_cast<Split*>(expr)) {
      handle(split, direction);
    } else if (auto resize = dynamic_cast<Resize*>(expr)) {
      handle(resize, direction);
    }
  }

  void handle(Merge* merge, Direction direction);

  void handle(Split* split, Direction direction);

  void handle(Resize* resize, Direction direction);

  const std::unordered_set<ValGroup>& contigIDs() const {
    return contig_ids_;
  }

  const std::unordered_map<IterDomain*, ValGroup>& allocToContigIDs() const {
    return alloc_to_contig_ids_;
  }

 private:
  // Indexing traversal graph.
  const ValGraph& graph_;
  // Domains to analyze contiguity. They are typically allocation
  // domains but if this is a predicate indexing pass, they are
  // likely logical domains.
  const std::vector<IterDomain*> alloc_domains_;
  // Contiguity of alloc_domains_
  const std::vector<bool> alloc_contiguity_;
  const bool is_predicate_pass_;
  std::unique_ptr<const OrderedIdGroupInformation> consistent_transform_info_;

  // Contig domain groups
  std::unordered_set<ValGroup> contig_ids_;
  // Mapping of allocation domains to contig groups
  std::unordered_map<IterDomain*, ValGroup> alloc_to_contig_ids_;
  // All domains that have dependencies with resize ops
  std::unordered_set<ValGroup> resize_deps_;
  // All domains that have dependencies with non-divisible split ops
  std::unordered_set<ValGroup> non_divisible_deps_;
};

// Get a contiguous indexing domain for a given allocation domain. If
// no such domain is found, just the allocation domain itself is
// returned.
std::unordered_map<IterDomain*, ValGroup> getContigDomains(
    const std::vector<IterDomain*>& alloc_domains,
    const std::vector<bool>& alloc_contiguity,
    const ExprPath<ExprGroup>& path_from_alloc,
    const ValGraph& graph,
    bool is_predicate_pass);

} // namespace nvfuser

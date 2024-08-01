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
#include <val_graph_visitor.h>

namespace nvfuser {

// Minimal adaptation of OrderedIdInformation for IdModel. Note that
// the analysis only propagates forward for now.
class OrderedIdGroupInformation : public OrderedIdInformation {
 public:
  static OrderedIdGroupInformation get(
      const std::vector<IterDomain*>& alloc_domain,
      const ExprPath<ExprGroup>& path,
      const ValGraph& graph) {
    OrderedIdGroupInformation info(alloc_domain, graph);
    info.traverse(path);
    return info;
  }

  bool isConsistentlyOrdered(IterDomain* id) const override {
    return std::find_if(
               consistently_ordered_ids_.begin(),
               consistently_ordered_ids_.end(),
               [&](IterDomain* consistent_id) -> bool {
                 return graph_.disjointValSets().strictAreMapped(
                     consistent_id, id);
               }) != consistently_ordered_ids_.end();
  }

 protected:
  OrderedIdGroupInformation(
      const std::vector<IterDomain*>& alloc_domain,
      const ValGraph& graph)
      : OrderedIdInformation(alloc_domain), graph_(graph) {
    using_id_graph_ = true;
  }

  // Currently only forward propagation is supported
  void traverse(const ExprPath<ExprGroup>& path) {
    for (const auto& [eg, direction] : path) {
      if (direction == Direction::Backward) {
        // TODO: support Backward prop
        continue;
      }
      dispatch(eg->front());
    }
  }

  // Traversal is based on the AlmostExact graph, so matching of iter
  // domains also needs to be done with the same graph
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

 private:
  const ValGraph& graph_;
};

std::unordered_map<IterDomain*, ValGroup> getContigDomains(
    const std::vector<IterDomain*>& index_domains,
    const std::vector<bool>& contiguity,
    const ExprPath<ExprGroup>& path_from_index_domains,
    const ValGraph& graph,
    bool is_predicate_pass);

} // namespace nvfuser

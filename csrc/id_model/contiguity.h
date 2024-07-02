// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <contiguity.h>
#include <val_graph_visitor.h>
#include <id_model/id_model.h>

namespace nvfuser {

class OrderedIdGroupInformation : public OrderedIdInformation {
 public:
  static OrderedIdGroupInformation get(
      const std::vector<IterDomain*>& alloc_domain,
      const ExprPath& path,
      const ValGraph& graph,
      const ConcretizedBroadcastDomains& concrete_info) {
    OrderedIdGroupInformation info(alloc_domain, graph, concrete_info);
    info.traverse(path);
    return info;
  }

  bool isConsistentlyOrdered(IterDomain* id) const override;
  
 protected:
  OrderedIdGroupInformation(
      const std::vector<IterDomain*>& alloc_domain,
      const ValGraph& graph,
      const ConcretizedBroadcastDomains& concrete_info)
      : OrderedIdInformation(alloc_domain, concrete_info), graph_(graph) {
    using_id_graph_ = true;
  }

  virtual void traverse(const ExprPath& path);
  
 protected:
  std::vector<IterDomain*>::const_iterator findActiveId(
      IterDomain* id) const override;

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>::
  const_iterator
  findAllocIDs(IterDomain* id) const override;

 private:
  const ValGraph& graph_;
};

std::unordered_map<IterDomain*, ValGroup> getContigDomains(
    const std::vector<IterDomain*>& alloc_domains,
    const std::vector<bool>& contiguity,
    const ExprPath& path_from_alloc,
    const ValGraph& graph,
    const ConcretizedBroadcastDomains& concrete_info,
    bool is_predicate_pass);

} // namespace nvfuser

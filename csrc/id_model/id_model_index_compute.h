// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dispatch.h>
#include <id_model/id_model.h>
#include <val_graph_visitor.h>

namespace nvfuser {

// Similar to IndexCompute but adapted for the graph-based indexing
class IdGraphIndexCompute : public OptOutDispatch {
 public:
  IdGraphIndexCompute(
      const ValGraph& traversal_graph,
      std::unordered_map<ValGroup, Val*> initial_index_map,
      const std::unordered_set<ValGroup>& max_path_domains)
      : traversal_graph_(traversal_graph),
        index_map_(std::move(initial_index_map)),
        max_path_domains_(max_path_domains) {}

  // Propagate the index map through a given expr of a specified
  // direction.
  void propagate(const ExprGroup& expr_group, Direction direction) {
    NVF_ERROR(!expr_group->empty());
    // Propagate max path domains
    propagateMaxPathDomains(expr_group, direction);
    // This looks a little ugly but the dispatch interface doesn't
    // have a way to pass arguments
    current_direction_ = direction;
    dispatch(expr_group->front());
    current_direction_ = Direction::Undefined;
  }

  const std::unordered_map<ValGroup, Val*>& indexMap() const {
    return index_map_;
  }

 private:
  using OptOutDispatch::handle;

  void handle(Split* split) override;

  void handle(Merge* merge) override;

  void handle(Swizzle* swizzle) override;

  void handle(Resize* resize) override;

  bool isForward(Expr* expr) const {
    return current_direction_ == Direction::Forward;
  }

  bool hasIndex(IterDomain* id) const {
    // If it's a broadcast, its index is always zero.
    if (id->isBroadcast()) {
      return true;
    }
    return indexMap().find(toGroup(id)) != indexMap().end();
  }

  Val* getIndex(IterDomain* id) const {
    // If it's a broadcast, its index is always zero.
    if (id->isBroadcast()) {
      return id->fusion()->zeroVal();
    }
    auto it = index_map_.find(toGroup(id));
    NVF_ERROR(it != index_map_.end(), "Index not found: ", id->toString());
    return it->second;
  }

  void setIndex(IterDomain* id, Val* idx) {
    index_map_.emplace(toGroup(id), idx);
  }

  const ValGroup& toGroup(IterDomain* id) const {
    return traversal_graph_.toGroup(id);
  }

  bool isInMaxPath(IterDomain* id) const {
    const auto& id_group = traversal_graph_.toGroup(id);
    return max_path_domains_.find(id_group) != max_path_domains_.end();
  }

  void propagateMaxPathDomains(
      const ExprGroup& expr_group,
      Direction direction) {
    const auto inputs = direction == Direction::Forward
        ? traversal_graph_.inputGroups(expr_group)
        : traversal_graph_.outputGroups(expr_group);

    if (std::any_of(
            inputs.begin(), inputs.end(), [&](const ValGroup& input) -> bool {
              return max_path_domains_.find(input) != max_path_domains_.end();
            })) {
      const auto outputs = direction == Direction::Forward
          ? traversal_graph_.outputGroups(expr_group)
          : traversal_graph_.inputGroups(expr_group);
      max_path_domains_.insert(outputs.begin(), outputs.end());
    }
  }

 private:
  const ValGraph& traversal_graph_;
  std::unordered_map<ValGroup, Val*> index_map_;
  Direction current_direction_ = Direction::Undefined;
  std::unordered_set<ValGroup> max_path_domains_;
};

} // namespace nvfuser

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
#include <id_model/to_string.h>

namespace nvfuser {

// Similar to IndexCompute but adapted for the graph-based indexing
class IdGraphIndexCompute : public OptOutDispatch {
 public:
  IdGraphIndexCompute(
      const ValGraph& traversal_graph,
      std::unordered_map<ValGroup, Val*> initial_index_map)
      : traversal_graph_(traversal_graph),
        index_map_(std::move(initial_index_map)) {}

  // Propagate the index map through a given expr of a specified
  // direction.
  void propagate(const ExprGroup& expr_group, Direction direction) {
    NVF_ERROR(!expr_group->empty());
    // This looks a little ugly but the dispatch interface doesn't
    // have a way to pass arguments
    current_direction_ = direction;
    dispatch(expr_group->front());
    current_direction_ = Direction::Undefined;
  }

  const std::unordered_map<ValGroup, Val*> indexMap() const {
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
    return indexMap().find(toGroup(id)) != indexMap().end();
  }

  Val* getIndex(IterDomain* id) const {
    auto it = index_map_.find(toGroup(id));
    NVF_ERROR(it != index_map_.end(), "Index not found: ", id->toString());
    return it->second;
  }

  void setIndex(IterDomain* id, Val* idx) {
    const auto& group = toGroup(id);
    if (auto it = index_map_.find(group); it != index_map_.end()) {
      // This can happen when back propagating a merge with a
      // broadcast domain. For example, merge 1, 8 -> 8. Here, the
      // index of the output ID is valid also for the inner input ID
      // since it's almost exact. However, to get the index of the the
      // outer broadcast input ID, the propgation would generate a new
      // index for the inner input ID that would look like x%8, which
      // may not be what we need for predication.
      // I don't think we would need to do any update here. The
      // shortest way to get an index should be the most preferred
      // option.
      std::cerr << "Do not updating index for " << id->toString() << " ("
                << nvfuser::toString(group) << "): " << idx->toInlineString()
                << "\n";
      return;
      it->second = idx;
      NVF_ERROR(index_map_.at(group) == idx);
    } else {
      index_map_.emplace(toGroup(id), idx);
    }
  }

  const ValGroup& toGroup(IterDomain* id) const {
    return traversal_graph_.toGroup(id);
  }

 private:
  const ValGraph& traversal_graph_;
  std::unordered_map<ValGroup, Val*> index_map_;
  Direction current_direction_ = Direction::Undefined;
};

} // namespace nvfuser

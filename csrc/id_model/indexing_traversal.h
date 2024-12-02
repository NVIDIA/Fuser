// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {

// BFS traversal for indexing. The only difference with the default
// ValGraphBFS is that for indexing there must be a special care taken
// when resize is involved since there can be multiple paths and
// there's only one correct path. Specifically, any resize expr group
// node must appear in the root-logical path of the consumer
// tensor. Otherwise, resize nodes should be ignored. See
// IndexingTest.ResizePath for a concrete example.
class IndexingTraversal : public ValGraphBFS {
 public:
  IndexingTraversal(
      const Expr* expr,
      const ValGraph& graph,
      std::vector<NodeType> from_groups,
      std::vector<NodeType> to_groups,
      bool require_all_to_visited = true);

  ~IndexingTraversal() override = default;

  static ExprPath getExprsBetween(
      const Expr* expr,
      const ValGraph& graph,
      const std::vector<IterDomain*>& from_domains,
      const std::vector<IterDomain*>& to_domains);

  static std::optional<ExprPath> getExprsBetweenForResize(
      const Expr* expr,
      const ValGraph& graph,
      const std::vector<IterDomain*>& from_domains,
      const std::vector<IterDomain*>& to_domains);

  using ValGraphBFS::isVisited;

  bool excludeFromTraversal(const NodeType& group) const override {
    if (const ExprGroup* eg = std::get_if<ExprGroup>(&group)) {
      if ((*eg)->empty()) {
        return false;
      }
      auto resize = dynamic_cast<Resize*>((*eg)->front());
      if (resize == nullptr) {
        return false;
      }
      if (std::none_of((*eg)->begin(), (*eg)->end(), [&](Expr* expr) -> bool {
            return resize_paths_.find(expr->as<Resize>()) !=
                resize_paths_.end();
          })) {
        // This resize node should never be traversed for indexing of
        // the given expr
        return true;
      }
    }
    return false;
  }

 private:
  std::unordered_set<Resize*> resize_paths_;
};

} // namespace nvfuser

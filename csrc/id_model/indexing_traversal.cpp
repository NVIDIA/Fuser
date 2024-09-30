// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/indexing_traversal.h>

namespace nvfuser {

IndexingTraversal::IndexingTraversal(
    const Expr* expr,
    const ValGraph& graph,
    std::vector<NodeType> from_groups,
    std::vector<NodeType> to_groups)
    : ValGraphBFS(graph, from_groups, to_groups) {
  auto consumer_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(consumer_tv != nullptr);
  // Remember the resize exprs appearing in the consumer
  // tensor. These resize exprs are the only ones that should be
  // valid to visit when indexing the inputs and outputs of the
  // expr.
  //
  // This is a WAR for cases like
  // ResizeTest.SliceScheduledLikeProducer. Alternatively, we could
  // have a separate graph for indexing that does not map producer and
  // consumers in non-unary ops. See PR #2897.
  auto all_ids = consumer_tv->domain()->allIDs();
  std::unordered_set<IterDomain*> all_id_set(all_ids.begin(), all_ids.end());
  for (auto id : all_ids) {
    auto resize = dynamic_cast<Resize*>(id->definition());
    if (resize == nullptr) {
      continue;
    }
    auto resize_in = resize->in();
    if (all_id_set.find(resize_in) == all_id_set.end()) {
      // ths resize must not be part of the exprs involved for
      // the domains of consumer_tv
      continue;
    }
    resize_paths_.insert(resize);
  }
}

bool IndexingTraversal::excludeFromTraversal(const NodeType& group) const {
  const ExprGroup* eg = std::get_if<ExprGroup>(&group);
  if (eg == nullptr || (*eg)->empty()) {
    return false;
  }

  auto resize = dynamic_cast<Resize*>((*eg)->front());
  if (resize == nullptr) {
    return false;
  }

  auto is_included_resize = [&](const ExprGroup& eg) -> bool {
    auto resize = dynamic_cast<Resize*>(eg->front());
    if (resize == nullptr) {
      return false;
    }

    return std::any_of(eg->begin(), eg->end(), [&](Expr* expr) -> bool {
      return resize_paths_.find(expr->as<Resize>()) != resize_paths_.end();
    });
  };

  if (is_included_resize(*eg)) {
    return false;
  }

  bool is_forward = isVisited(inputs_(*eg).at(0));

  ValGroup inp = is_forward ? inputs_(*eg).at(0) : outputs_(*eg).at(0);

  const ExprGroups& uses_of_inp = uses_(inp);
  const ExprGroups& defs_of_inp = definition_(inp);

  // If there's any other resize op that's in the resize path, exclude
  // this resize
#if 0
  std::cerr << "Checking if any other op is included. is_forward: "
            << is_forward << "\n"
            << resize->toString();
#endif
  for (const auto& use_or_def : {uses_of_inp, defs_of_inp}) {
    for (const ExprGroup& expr_g : use_or_def) {
      if (expr_g == *eg) {
        continue;
      }
      // Don't care if already visited
      if (isVisited(expr_g)) {
        continue;
      }
      auto resize = dynamic_cast<Resize*>((expr_g)->front());
      if (resize != nullptr) {
        // std::cerr << "Other resize: " << resize->toString();
        if (is_included_resize(expr_g)) {
          // std::cerr << "Other resize included\n";
          return true;
        }
        // std::cerr << "Other resize not included\n";
      }
    }
  }

  // std::cerr << "Not exluding: " << resize->toString();
  return false;
}

} // namespace nvfuser

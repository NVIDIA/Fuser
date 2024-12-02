// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
#include <id_model/indexing_traversal.h>
#include <ir/utils.h>

namespace nvfuser {

IndexingTraversal::IndexingTraversal(
    const Expr* expr,
    const ValGraph& graph,
    std::vector<NodeType> from_groups,
    std::vector<NodeType> to_groups,
    bool require_all_to_visited)
    : ValGraphBFS(graph, from_groups, to_groups, require_all_to_visited) {
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

std::optional<IndexingTraversal::ExprPath> IndexingTraversal::
    getExprsBetweenForResize(
        const Expr* expr,
        const ValGraph& graph,
        const std::vector<IterDomain*>& from_ids,
        const std::vector<IterDomain*>& to_ids) {
  auto consumer_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(consumer_tv != nullptr);

  IdModel local_model(
      std::vector<Expr*>{consumer_tv->definition()},
      /*additional_tvs=*/{},
      /*build_graphs=*/false);

  // If there's no resize in the producer and consumer tensors of this
  // expr, it should not need this WAR.
  if (std::none_of(
          local_model.idUses().begin(),
          local_model.idUses().end(),
          [](const auto& kv) {
            const VectorOfUniqueEntries<Expr*>& exprs = kv.second;
            return !exprs.empty() && exprs.at(0)->isA<Resize>();
          })) {
    return std::nullopt;
  }

  const auto& local_graph = local_model.buildAlmostExactGraph();

  // from_ids are loop domains, which are representative
  // domains of loop groups and not necessarily domains of any
  // of the producer and the consumer.  In that case, find an ID out
  // of the global group that is mapped in the local graph.
  ValGroups from_groups;
  for (const auto i : c10::irange(from_ids.size())) {
    auto from_id = from_ids.at(i);
    if (local_graph.hasGroup(from_id)) {
      from_groups.pushBack(local_graph.toGroup(from_id));
      continue;
    }
    bool found = false;
    const auto& global_group = graph.toGroup(from_id);
    for (const auto& vg : local_graph.disjointValSets().disjointSets()) {
      if (global_group->has(vg->front())) {
        from_groups.pushBack(vg);
        found = true;
        break;
      }
    }
    // If not found, it should mean it's promoted to some IDs of
    // further consumer tensors. This WAR does not work then. We could
    // simply fall back to the default ValGraph-based path, but that
    // might hit the resize indexing issue (#3455). For now, this is
    // considered an error.
    NVF_ERROR(
        found, "Indexing path for resize not found: ", from_id->toString());
  }

  // Similarly, to_ids may not be IDs found in any of the producer and
  // consumer tensors of this expr. For example, if it's an allocation
  // ID, it may be a loop promotion ID.
  ValGroups to_groups;
  for (auto to_id : to_ids) {
    if (local_graph.hasGroup(to_id)) {
      to_groups.pushBack(local_graph.toGroup(to_id));
      continue;
    }
    // to_id is not found in the producer or consumer tensors of the
    // expr. Look for a mapped ID in the ID group of the global graph.
    bool found = false;
    const auto& global_group = graph.toGroup(to_id);
    for (const auto& vg : local_graph.disjointValSets().disjointSets()) {
      if (global_group->has(vg->front())) {
        to_groups.pushBack(vg);
        found = true;
        break;
      }
    }
    NVF_ERROR(found, "Indexing path for resize not found: ", to_id->toString());
  }

  IndexingTraversal traversal(
      expr,
      local_graph,
      {from_groups.vector().begin(), from_groups.vector().end()},
      {to_groups.vector().begin(), to_groups.vector().end()},
      /*require_all_to_visited=*/true);
  traversal.traverse();
  auto [path, all_visited] = traversal.getShortestExprPath();

  for (const auto& [g, d] : path) {
    if (g->front()->isA<Resize>()) {
      return path;
    }
  }

  // If resize doesn't appear, the default path should work fine.
  return std::nullopt;
}

IndexingTraversal::ExprPath IndexingTraversal::getExprsBetween(
    const Expr* expr,
    const ValGraph& graph,
    const std::vector<IterDomain*>& from_domains,
    const std::vector<IterDomain*>& to_domains) {
  // Take the path if found by the war for resize indexing
  if (auto path =
          getExprsBetweenForResize(expr, graph, from_domains, to_domains);
      path.has_value()) {
    return *path;
  }

  auto from_groups = graph.toGroups(from_domains);
  auto to_groups = graph.toGroups(to_domains);

  IndexingTraversal traversal(
      expr,
      graph,
      {from_groups.vector().begin(), from_groups.vector().end()},
      {to_groups.vector().begin(), to_groups.vector().end()});
  traversal.traverse();
  return traversal.getShortestExprPath().first;
}

} // namespace nvfuser

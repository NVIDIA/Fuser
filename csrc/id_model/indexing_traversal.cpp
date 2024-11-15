// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
#include <id_model/indexing_traversal.h>

#include <fstream>

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

IndexingTraversal::ExprPath IndexingTraversal::getExprsBetween(
    const Expr* expr,
    const ValGraph& graph,
    const ValGroups& from_groups,
    const ValGroups& to_groups) {
  // First traversal. Allows unreachable nodes as they may be resolved
  // by the second traversal
  IndexingTraversal first_traversal(
      expr,
      graph,
      {from_groups.vector().begin(), from_groups.vector().end()},
      {to_groups.vector().begin(), to_groups.vector().end()},
      /*require_all_to_visited=*/false);
  first_traversal.traverse();
  auto first_path = first_traversal.getShortestExprPath();
  auto unreachable_to_groups = ValGraphBFS::getUnreachableValsFrom(
      graph, from_groups, to_groups, first_path);
  if (unreachable_to_groups.empty()) {
    return first_path;
  }

  ValGroups from_groups_with_broadcat = from_groups;
  for (const auto& vg : graph.disjointValSets().disjointSets()) {
    if (vg->front()->as<IterDomain>()->isBroadcast()) {
      from_groups_with_broadcat.pushBack(vg);
    }
  }

  // Second trial. Ignore dependencies with broadcast groups since
  // unreachable broadcast groups should be indexable with just
  // 0. Note that that may not be true if a broadcast group is
  // reachable from from_groups, so this relaxation should not be used
  // in the first traversal.
  IndexingTraversal second_traversal(
      expr,
      graph,
      {from_groups_with_broadcat.vector().begin(),
       from_groups_with_broadcat.vector().end()},
      {unreachable_to_groups.begin(), unreachable_to_groups.end()},
      /*require_all_to_visited=*/true);
  second_traversal.traverse();
  auto second_path = second_traversal.getShortestExprPath();
  first_path.insert(first_path.end(), second_path.begin(), second_path.end());

  return first_path;
}

std::optional<IndexingTraversal::ExprPath> IndexingTraversal::
    getExprsBetweenForResize(
        const Expr* expr,
        const ValGraph& graph,
        const std::vector<IterDomain*>& from_domains,
        const std::vector<IterDomain*>& to_domains) {
  auto consumer_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(consumer_tv != nullptr);

  std::cerr << "IndexingTraversal: trying resize war: "
            << consumer_tv->toString() << "\n";
  IdModel local_model(
      std::vector<Expr*>{consumer_tv->definition()},
      /*additional_tvs=*/{},
      /*build_graphs=*/false);
  const auto& local_graph = local_model.buildAlmostExactGraph();

  // from_domains are loop domains, which are representative
  // domains of loop groups and not necessarily domains of any
  // of the producer and the consumer
  std::vector<IterDomain*> consumer_from_domains(from_domains.size());
  for (const auto i : c10::irange(from_domains.size())) {
    auto consumer_id = consumer_tv->getLoopDomain().at(i);
    auto from_id = from_domains.at(i);
    NVF_ERROR(
        graph.disjointValSets().strictAreMapped(consumer_id, from_id),
        "Expected strict mapping: ",
        consumer_id->toString(),
        ", ",
        from_id->toString());
    consumer_from_domains.at(i) = consumer_id;
  }

  auto from_groups = local_graph.toGroups(consumer_from_domains);

  // to_domains may not point to IDs of this tensor. If it's an
  // allocation ID, it may be a loop promotion ID.
  // auto to_groups = local_graph.toGroups(to_domains);
  ValGroups to_groups;
  for (auto id : to_domains) {
    if (local_graph.hasGroup(id)) {
      to_groups.pushBack(local_graph.toGroup(id));
      continue;
    }
    bool found = false;
    const auto& global_group = graph.toGroup(id);
    for (const auto& vg : local_graph.disjointValSets().disjointSets()) {
      if (global_group->has(vg->front())) {
        to_groups.pushBack(vg);
        found = true;
        break;
      }
    }
    NVF_ERROR(found);
  }

  IndexingTraversal traversal(
      expr,
      local_graph,
      {from_groups.vector().begin(), from_groups.vector().end()},
      {to_groups.vector().begin(), to_groups.vector().end()});
  // Local graph may not have all info necessary for
  // indexing. If traversal fails, just giving up taking this
  // WAR. This isn't ideal, but the WAR itself isn't ideal either...
  traversal.require_all_to_visited_ = false;
  traversal.traverse();
  auto path = traversal.getShortestExprPath();
  auto path_outputs = getOutputsOfExprPath(local_graph, path);
  if (std::any_of(
          to_groups.begin(), to_groups.end(), [&](const ValGroup& to_group) {
            return !path_outputs.has(to_group) && !from_groups.has(to_group);
          })) {
    // Some to_group wasn't reached. Not a valid path.
    return std::nullopt;
  }

  bool resize_used = false;
  for (const auto& [g, d] : path) {
    if (g->front()->isA<Resize>()) {
      // found
      resize_used = true;
      break;
    }
  }

  if (resize_used) {
    std::cerr << "getExprsBetween resize WAR: " << expr->toString();
    std::cerr << "From: " << toDelimitedString(from_domains) << "\n";
    std::cerr << "To: " << toDelimitedString(to_domains) << "\n";
    std::cerr << "Resize indexing path\n";
    for (const auto& [g, d] : path) {
      std::cerr << "\t" << d << g->front()->toString();
    }
    return path;
  }

  return std::nullopt;
}

IndexingTraversal::ExprPath IndexingTraversal::getExprsBetween(
    const Expr* expr,
    const ValGraph& graph,
    const std::vector<IterDomain*>& from_domains,
    const std::vector<IterDomain*>& to_domains) {
  if (auto path =
          getExprsBetweenForResize(expr, graph, from_domains, to_domains);
      path.has_value()) {
    return *path;
  }

  std::cerr << "Not using resize war\n";

  auto from_groups = graph.toGroups(from_domains);
  auto to_groups = graph.toGroups(to_domains);

  return IndexingTraversal::getExprsBetween(
      expr, graph, from_groups, to_groups);
}

} // namespace nvfuser

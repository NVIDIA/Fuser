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

  // A unique expr path should be always allowed
  for (const auto& expr_g : graph.disjointExprSets().disjointSets()) {
    auto resize = dynamic_cast<Resize*>(expr_g->front());
    if (resize == nullptr) {
      continue;
    }

    if (graph.inputGroups(expr_g).size() == 1 &&
        graph.outputGroups(expr_g).size() == 1) {
      resize_paths_.insert(resize);
    }
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

  std::cerr << "IndexingTraversal: trying resize war: "
            << consumer_tv->toString() << "\n";
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
  // of the producer and the consumer
  std::vector<IterDomain*> consumer_from_domains(from_ids.size());
  for (const auto i : c10::irange(from_ids.size())) {
    auto consumer_id = consumer_tv->getLoopDomain().at(i);
    auto from_id = from_ids.at(i);
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

  std::cerr << "Resize war from: " << nvfuser::toString(from_groups) << " -> "
            << nvfuser::toString(to_groups) << "\n";
  std::cerr << expr->toString();

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
  auto path = traversal.getShortestExprPath().first;
  auto path_outputs = getOutputsOfExprPath(local_graph, path);
  if (std::any_of(
          to_groups.begin(), to_groups.end(), [&](const ValGroup& to_group) {
            return !path_outputs.has(to_group) && !from_groups.has(to_group);
          })) {
    // Some to_group wasn't reached. Not a valid path.
    std::cerr << "Unreachable node detected\n";
    return std::nullopt;
  }

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

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
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

IndexingTraversal::ExprPath IndexingTraversal::getExprsBetween(
    const Expr* expr,
    const ValGraph& graph,
    const ValGroups& from_groups,
    const ValGroups& to_groups) {
  IndexingTraversal traversal(
      expr,
      graph,
      {from_groups.vector().begin(), from_groups.vector().end()},
      {to_groups.vector().begin(), to_groups.vector().end()});
  traversal.traverse();
  return traversal.getShortestExprPath();
}

IndexingTraversal::ExprPath IndexingTraversal::getExprsBetween(
    const Expr* expr,
    const ValGraph& graph,
    const std::vector<IterDomain*>& from_domains,
    const std::vector<IterDomain*>& to_domains) {
  auto consumer_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(consumer_tv != nullptr);
  std::cerr << "DEBUG***\n";
  // WAR
#if 0
  {
    if (consumer_tv->hasRoot()) {
      auto root_to_logical_exprs = DependencyCheck::getAllExprsBetween(
          {consumer_tv->getRootDomain().begin(),
           consumer_tv->getRootDomain().end()},
          {consumer_tv->getLogicalDomain().begin(),
           consumer_tv->getLogicalDomain().end()});

      if (std::any_of(
              root_to_logical_exprs.begin(),
              root_to_logical_exprs.end(),
              [](Expr* expr) { return expr->isA<Resize>(); })) {
#if 1
        std::cerr << "getExprsBetween resize WAR: " << expr->toString();
        std::cerr << "From: " << toDelimitedString(from_domains) << "\n";
        std::cerr << "To: " << toDelimitedString(to_domains) << "\n";
#endif
        IdModel local_model(
            std::vector<Expr*>{consumer_tv->definition()},
            /*additional_tvs=*/{},
            /*build_graphs=*/false);
        const auto& local_graph = local_model.buildAlmostExactGraph();
        // std::cerr << "Local graph: " << local_graph.toString() <<
        // "\n";

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
        auto to_groups = local_graph.toGroups(to_domains);

        IndexingTraversal traversal(
            expr,
            local_graph,
            {from_groups.vector().begin(), from_groups.vector().end()},
            {to_groups.vector().begin(), to_groups.vector().end()});
        traversal.traverse();
        auto p = traversal.getShortestExprPath();
        std::cerr << "Resize indexing path\n";
        for (const auto& [g, d] : p) {
          std::cerr << "\t" << d << g->front()->toString();
        }
        return p;
      }
    }
  }
#else
  {
    std::cerr << "IndexingTraversal: trying resize war: "
              << consumer_tv->toString() << "\n";
    IdModel local_model(
        std::vector<Expr*>{consumer_tv->definition()},
        /*additional_tvs=*/{},
        /*build_graphs=*/false);
    const auto& local_graph = local_model.buildAlmostExactGraph();
    // std::cerr << "Local graph: " << local_graph.toString() <<
    // "\n";

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
    auto to_groups = local_graph.toGroups(to_domains);

    IndexingTraversal traversal(
        expr,
        local_graph,
        {from_groups.vector().begin(), from_groups.vector().end()},
        {to_groups.vector().begin(), to_groups.vector().end()});
    traversal.traverse();
    auto path = traversal.getShortestExprPath();
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
  }
#endif

  auto from_groups = graph.toGroups(from_domains);
  auto to_groups = graph.toGroups(to_domains);

  IndexingTraversal traversal(
      expr,
      graph,
      {from_groups.vector().begin(), from_groups.vector().end()},
      {to_groups.vector().begin(), to_groups.vector().end()});
  traversal.traverse();
  return traversal.getShortestExprPath();
}

} // namespace nvfuser

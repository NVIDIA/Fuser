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
  auto register_valid_resizes = [&](TensorView* tv) {
    auto all_ids = tv->domain()->allIDs();
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
  };

  for (auto inp_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
    register_valid_resizes(inp_tv);
  }
  for (auto out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
    register_valid_resizes(out_tv);
  }

  // A unique expr path should be always allowed
  for (const auto& expr_g : graph.disjointExprSets().disjointSets()) {
    auto resize = dynamic_cast<Resize*>(expr_g->front());
    if (resize == nullptr) {
      continue;
    }

    auto input_groups = graph.inputGroups(expr_g);
    auto output_groups = graph.outputGroups(expr_g);
    NVF_ERROR(input_groups.size() == 1);
    NVF_ERROR(output_groups.size() == 1);

    if (graph.getUses(input_groups[0]).size() != 1 ||
        graph.getDefinitions(output_groups[0]).size() != 1) {
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

  // First, try to limit the use of this WAR as much as possible since
  // the WAR itself has a limitation that it assumes the loop domain
  // is not promoted.

  IdModel local_model(
      std::vector<Expr*>{consumer_tv->definition()},
      /*additional_tvs=*/{},
      /*build_graphs=*/false);

  // Gather all resize exprs for each of the inputs and outputs
  std::unordered_map<TensorView*, std::vector<Resize*>> tv_resize_map;
  for (auto inp : expr->inputs()) {
    auto inp_tv = ir_utils::getTv(inp);
    if (inp_tv == nullptr) {
      continue;
    }
    for (auto expr : inp_tv->domain()->allExprs()) {
      if (auto resize = dynamic_cast<Resize*>(expr)) {
        tv_resize_map[inp_tv].push_back(resize);
      }
    }
  }
  for (auto out : expr->outputs()) {
    auto out_tv = ir_utils::getTv(out);
    if (out_tv == nullptr) {
      continue;
    }
    for (auto expr : out_tv->domain()->allExprs()) {
      if (auto resize = dynamic_cast<Resize*>(expr)) {
        tv_resize_map[out_tv].push_back(resize);
      }
    }
  }

  // If there's no resize in the producer and consumer tensors of this
  // expr, it should not need this WAR.
  if (tv_resize_map.empty()) {
    return std::nullopt;
  }

  // The indexing issue with resize may happen when a single iter
  // domain is resized multiple times between a producer and a
  // consumer. In other words, there must be at least two connected
  // resize exprs. If not, this WAR is not necessary.
  //
  // Note that the actual indexing is done from the loop IDs, which
  // might be promoted to IDs outside of this particular expr. Thus,
  // to get the true indexing path, the global IdModel may need to be
  // used rather than the local model. Here, since we just need to
  // know if there are multiple dependent resize exprs, and loop
  // promotion should not further add resize exprs, it is sufficient
  // to analyze only the IDs of this expr.

  const auto& local_graph = local_model.buildAlmostExactGraph();

  // The below analysis is done for each producer-consumer pair, so it
  // can be a rather expensive analysis, but in practice most
  // cases should just bail out at the first if condition

  auto isSingleIdResizedMultipleTimes = [&](TensorView* inp,
                                            TensorView* out) -> bool {
    auto num_resizes = tv_resize_map[inp].size() + tv_resize_map[out].size();
    if (num_resizes < 2) {
      return false;
    }

    std::vector<Resize*> resize_exprs;
    resize_exprs.reserve(num_resizes);
    resize_exprs.insert(
        resize_exprs.end(),
        tv_resize_map[inp].begin(),
        tv_resize_map[inp].end());
    resize_exprs.insert(
        resize_exprs.end(),
        tv_resize_map[out].begin(),
        tv_resize_map[out].end());

    // See if these resize expr groups are connected. Note that in the
    // current default scheduling method, any tensor ops using resize
    // should only show up with a fusion input as its input, so there
    // must be no chained resize ops. The default scheduling, this
    // function should not move beyond this point. In the case of the
    // new resize scheduler that is currently under development will
    // have multiple chained resize ops, but the scheduler should
    // explicitly set the loop domain such that no promotion would
    // happen, thus avoiding hitting the assertion down below.
    ExprGroups resize_groups = local_graph.toGroups(resize_exprs);
    for (const auto i : arange(resize_groups.size() - 1)) {
      const auto resize_i = resize_groups.at(i);
      std::vector<ValGraphBFS::NodeType> other_resizes{
          resize_groups.begin() + i + 1, resize_groups.end()};
      auto reachable_nodes = getReachableNodesFrom<ValGraphBFS>(
          {resize_i}, other_resizes, Direction::Undefined, local_graph);
      if (!reachable_nodes.empty()) {
        return true;
      }
    }

    return false;
  };

  bool single_id_resized_multiple_times = false;
  for (auto out : expr->outputs()) {
    auto out_tv = ir_utils::getTv(out);
    if (out_tv == nullptr) {
      continue;
    }
    for (auto inp : expr->inputs()) {
      auto inp_tv = ir_utils::getTv(inp);
      if (inp_tv == nullptr) {
        continue;
      }
      if (isSingleIdResizedMultipleTimes(inp_tv, out_tv)) {
        single_id_resized_multiple_times = true;
        break;
      }
    }
    if (single_id_resized_multiple_times) {
      break;
    }
  }

  // No connection between the resize exprs is found, which they are
  // all independent and there's no need to use this WAR
  if (!single_id_resized_multiple_times) {
    return std::nullopt;
  }

  // from_ids are loop domains, which are representative
  // domains of loop groups and not necessarily domains of any
  // of the producer and the consumer.  In that case, find an ID out
  // of the global group that is mapped in the local graph.
  ValGroups from_groups;
  for (const auto i : arange(from_ids.size())) {
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

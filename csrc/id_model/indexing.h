// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <id_model/id_model.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <type.h>

// Just for RootPredicateInfo. Should be moved to its own header file
#include <index_compute.h>

#include <unordered_map>

namespace nvfuser {

struct IndexingInfo {
  ExprGroups traversal_path;
  std::unordered_map<ValGroup, Val*> index_map;
};

class TensorIndexer {
 public:
  TensorIndexer(const IdModel& id_model);

  // The actual ForLoop's are required to support double buffering as
  // that affects indexing. If the loops parameter is empty, it's
  // simply ignored. That may be useful if (preliminary) indeices are
  // needed before the double buffering pass
  Val* getIndex(
      TensorView* tv,
      const Expr* expr,
      const std::optional<std::vector<kir::ForLoop*>>& loops);

  std::vector<RootPredicateInfo> getPredicates(
      TensorView* tv,
      const Expr* expr,
      const std::optional<std::vector<kir::ForLoop*>>& loops);

  IndexingInfo getIndex(
      TensorView* tv,
      const Expr* expr,
      const std::optional<std::vector<kir::ForLoop*>>& loops,
      const std::vector<IterDomain*>& index_domains,
      const ValGraph& traversal_graph,
      bool predicate) const;

  static bool isSupported(Fusion* fusion);

 private:
  void buildLoopIndexMap();

  Val* adjustProducerLoopIndexForDoubleBuffering(
      TensorView* producer_tv,
      TensorView* consumer_tv,
      const kir::ForLoop* for_loop,
      Val* loop_index) const;

  Val* adjustIndexToSwitchBuffer(
      TensorView* tv,
      bool as_consumer,
      const std::vector<kir::ForLoop*>& for_loops,
      Val* idx) const;

  Val* getLoopIndex(IterDomain* id) const;

  std::unordered_map<ValGroup, Val*> getInitialIndexMap(
      TensorView* tv,
      const Expr* expr,
      const std::optional<std::vector<kir::ForLoop*>>& for_loops,
      const std::vector<IterDomain*>& loop_domains,
      const ValGraph& traversal_graph,
      bool predicate) const;

 private:
  const IdModel& id_model_;
  std::unordered_map<ValGroup, Val*> loop_index_map_;
};

} // namespace nvfuser

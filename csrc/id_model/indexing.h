// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/analysis/trivial_broadcast.h>
#include <id_model/id_model.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <type.h>
#include <val_graph_visitor.h>

// Just for RootPredicateInfo. Should be moved to its own header file
#include <index_compute.h>

#include <unordered_map>

namespace nvfuser {

struct IndexingInfo {
  ExprPath traversal_path;
  std::unordered_map<ValGroup, Val*> index_map;
};

// The basic algorithm of indexing is:
//
// 1. Find the loop domains
// 2. Find the allocation domains
// 3. Find the path from the loop domains to the allocation domains
// 4. Set the initial index vals
// 5. Propagate the initial indices of the loop domains to the allocation
// domains
class TensorIndexer {
 public:
  TensorIndexer(const IdModel& id_model);

  // The actual ForLoop's are required to support double buffering as
  // that affects indexing. If the loops parameter is empty, it's
  // simply ignored. That may be useful if (preliminary) indeices are
  // needed before the double buffering pass
  Val* getTensorIndex(
      TensorView* tv,
      const Expr* expr,
      const std::optional<std::vector<kir::ForLoop*>>& loops);

 private:
  const ValGraph& traversalGraph() const {
    return id_model_.idGraph(IdMappingMode::ALMOSTEXACT);
  }

  // Build the map of loop groups to their index Vals.
  void buildLoopIndexMap();

  // Get the index of a loop domain.
  Val* getLoopIndex(IterDomain* loop_id) const;

  //
  std::unordered_map<ValGroup, Val*> getInitialIndexMap(
      const Expr* expr,
      const std::vector<IterDomain*>& loop_domains) const;

  IndexingInfo getIndex(
      const Expr* expr,
      const std::vector<IterDomain*>& index_domains) const;

 private:
  const IdModel& id_model_;

  // Mappings from loop groups to their indices. Serial loops will
  // be mapped a unique loop index Val. Parallel loops will be mapped
  // to NamedScalar such as "threadIdx.x". This map needs to be built
  // once and can be reused for different tensors.
  std::unordered_map<ValGroup, Val*> loop_index_map_;
};

} // namespace nvfuser

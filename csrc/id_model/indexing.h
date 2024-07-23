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
#include <id_model/indexing_traversal.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <type.h>
#include <val_graph_visitor.h>

// Just for RootPredicateInfo. Should be moved to its own header file
#include <index_compute.h>

#include <unordered_map>

namespace nvfuser {

struct IndexingInfo {
  std::vector<IterDomain*> loop_domains;
  // Indexing traversal path from loop domains
  ExprPath<ExprGroup> traversal_path;
  // Index mappings of ID groups along the traversal path
  std::unordered_map<ValGroup, Val*> index_map;
};

struct IndexingAllocationInfo {
  std::vector<IterDomain*> domains;
  std::vector<Val*> strides;
};

// The basic algorithm of indexing is:
//
// 1. Find the loop domains
// 2. Find the allocation domains
// 3. Find the path from the loop domains to the allocation domains
// 4. Set the initial index vals for the loop domains
// 5. Propagate the initial indices of the loop domains to the allocation
// domains
//
// The indexing traversal is done on the AlmostExact graph augmented
// with the loop promotion map since both the loop and allocations
// domains may be promoted.
class TensorIndexer {
 public:
  // Using non-const references of IdModel because traversalGraph() returns a
  // non-const reference
  TensorIndexer(IdModel& id_model);

  // Get a linear index of a given tensor appearing in a given expr, either
  // as a consumer or a producer. The predicate indexing will have a
  // separate interface.
  //
  // The actual for-loops are required for handling circular buffering
  Val* getLinearIndex(
      TensorView* tv,
      const Expr* expr,
      const std::vector<ForLoop*>& loops) const;

  // Get the index of a loop domain. Intended to be used only for testing.
  Val* getLoopIndex(IterDomain* loop_id) const;

  // Get the index of the given ID groups
  std::vector<Val*> getIndexFor(
      const Expr* expr,
      bool as_consumer,
      const ValGroups& index_groups,
      const std::vector<ForLoop*>& loops) const;

  // The AlmostExact graph is used since size-1 splits and merges
  // should not affect actual index exprs.
  // Returns non-const reference because indexing may create new domains and
  // need to update the graph.
  ValGraph& traversalGraph() const {
    return id_model_.idGraph(IdMappingMode::ALMOSTEXACT);
  }

  // Traverse exprs and set allocation info for each tensor
  void setupAllocationDomains(const std::vector<Expr*>& exprs);

  // Get the list of predicates of a given tensor appearing in a given
  // expr as a consumer. Each predicate corresponds to a domain of the
  // tensor, which is by default one of the logical domains but can be
  // an intermediate domain with contiguous indexing.
  //
  // An optional ForLoop parameter specifies a loop that is either
  // unswitched/unrolled or vectorized, both of which are handled by
  // UnswitchPredicate. For normal inline predicates, the parameter
  // should be nullptr.
  std::vector<PredicateInfo> getPredicates(
      TensorView* tv,
      const Expr* expr,
      const std::vector<ForLoop*>& for_loops,
      ForLoop* unswitched_loop = nullptr) const;

 private:
  // Build a map of loop groups to their index Vals. See the comment
  // on loop_index_map_.
  void buildLoopIndexMap();

  const IndexingAllocationInfo& getIndexingAllocationInfo(
      TensorView* tv) const {
    auto it = alloc_info_.find(tv);
    NVF_ERROR(
        it != alloc_info_.end(),
        "No allocation info found for ",
        tv->toString());
    return it->second;
  }

  // Returns the index map as well as its traversal path of given
  // index domains appearing in a given expr. Used by
  // getIndexFor.
  IndexingInfo computeIndex(
      const Expr* expr,
      const ValGroups& index_groups,
      const std::vector<ForLoop*>& for_loops) const;

  // Propagate the loop indices of a given list of loop domains to the
  // traversal graph (i.e., the AlmostExact graph). Uses the loop
  // index map, which is built for the Loop graph.
  std::unordered_map<ValGroup, Val*> getInitialIndexMap(
      const std::vector<IterDomain*>& loop_domains,
      const std::vector<ForLoop*>& for_loops) const;

  // Get the loop domains of a given expr. Currently, they're always
  // the loop domains of a consumer tensor, but in the future this
  // function may return the loop domains of a producer for
  // producer-based indexing.
  std::vector<IterDomain*> getLoopDomains(const Expr* expr) const;

  // Check if the loop index of a loop group should be always
  // just zero. For example, a loop group with an extent of one, i.e.,
  // a broadcast-only loop group, should just use zero.
  bool shouldUseZeroIndex(const ValGroup& loop_group) const;

  // Get a replace map for tensor indexing. Examples include replacing
  // an index of a vectorized loop with zero.
  //
  // This replacement map is used to replace a tensor index after an
  // index map is generated. Since replacment is only done for loop
  // domains, this could be done as part of getInitialIndexMap. One
  // reason that we might want to first generate an index and do some
  // replacements, rather than using final index vals to build the
  // index map, is that one index map could be used for multiple
  // indices. For normal tensor indexing, this may not matter, but for
  // predicate indexing, it needs to generate both start and stop
  // predicates, and one index map would be sufficient for both
  // indices by using different replacement maps.
  std::unordered_map<Val*, Val*> getIndexReplacementMap(
      const Expr* expr,
      bool as_consumer,
      const std::vector<IterDomain*>& loop_domains,
      const std::vector<ForLoop*>& for_loops,
      const std::unordered_map<ValGroup, Val*>& index_map) const;

 private:
  // Using non-const references of IdModel because traversalGraph() returns a
  // non-const reference
  IdModel& id_model_;

  // Mappings from loop groups to their indices. Serial loops will
  // be mapped a unique loop index Val. Parallel loops will be mapped
  // to NamedScalar such as "threadIdx.x". This map needs to be built
  // once and can be reused for different tensors.
  std::unordered_map<ValGroup, Val*> loop_index_map_;

  // Allocation info for each tensor. Must be filled before computing
  // the index of each tensor
  std::unordered_map<TensorView*, IndexingAllocationInfo> alloc_info_;
};

} // namespace nvfuser

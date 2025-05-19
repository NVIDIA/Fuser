// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/analysis/trivial_broadcast.h>
#include <device_lower/pass/allocation.h>
#include <device_lower/utils.h>
#include <id_model/id_model.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <options.h>
#include <type.h>

// Just for PredicateInfo. Should be moved to its own header file
#include <index_compute.h>

#include <unordered_map>

namespace nvfuser {

struct IndexingInfo {
  std::vector<IterDomain*> loop_ids;
  std::vector<IterDomain*> index_ids;
  // Indexing traversal path from loop domains
  ExprPath<ExprGroup> traversal_path;
  // Index mappings of ID groups along the traversal path
  std::unordered_map<ValGroup, Val*> index_map;
  // Mappings from ID groups to dependent loop groups
  std::unordered_map<ValGroup, ValGroups> loop_group_dependencies;
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

  bool isContigIndexingEnabled() const {
    return !isOptionDisabled(DisableOption::ContigIndexing);
  }

  // Get a linear index of a given tensor appearing in a given expr, either
  // as a consumer or a producer. The predicate indexing will have a
  // separate interface.
  //
  // The actual for-loops are required for handling circular
  // buffering.
  //
  // The override_index parameter is used to enable indirect indexing
  // such as index_select. When the parameter is given, the index of a
  // ValGroup corresponding to a given iter domain is given by its
  // mapped index Val.
  Val* getLinearIndex(
      TensorView* tv,
      const Expr* expr,
      const std::vector<ForLoop*>& loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {}) const;

  // Get the index of a loop domain.
  Val* getLoopIndex(IterDomain* loop_id, const std::vector<ForLoop*>& for_loops)
      const;

  // Get the index of the given ID groups
  std::vector<Val*> getIndexFor(
      const Expr* expr,
      bool as_consumer,
      const std::vector<IterDomain*>& index_ids,
      const std::vector<ForLoop*>& loops,
      bool use_magic_zero = false) const;

  // Get the contig indices of the given ID groups with their strides
  std::pair<std::vector<Val*>, std::vector<Val*>> getContigIndexFor(
      TensorView* tv,
      const Expr* expr,
      bool as_consumer,
      const AllocationDomainInfo& alloc_info,
      const std::vector<ForLoop*>& loops,
      const std::unordered_map<IterDomain*, Val*>& override_index) const;

  // Grab all for-loops whose indices are actually used in the given
  // index vals. Note that IndexingInfo.loop_group_dependencies can be
  // used to find loop IDs that are connected to the index IDs, but
  // that doesn't always mean corresponding loop indices are actually
  // used in an index Val. For example, unswitch predicates replace loop indices
  // with (N - 1), where N is the extent of an unswitched ID. This
  // function only grabs for-loops whose indices are indeed used.
  std::vector<ForLoop*> getUsedForLoopsOf(
      const std::vector<Val*>& indices,
      const std::vector<ForLoop*>& for_loops) const;

  // Add "pragma unroll" to for-loops whose loop indices are used for
  // the given indexing. This is meant to be used for register tensors.
  void ensureStaticIndexing(const std::vector<ForLoop*>& loops, Val* index)
      const;

  // The AlmostExact graph is used since size-1 splits and merges
  // should not affect actual index exprs.
  // Returns non-const reference because indexing may create new domains and
  // need to update the graph.

  static IdMappingMode traversalGraphType() {
    return IdMappingMode::ALMOSTEXACT;
  }

  ValGraph& traversalGraph() const {
    return id_model_.idGraph(traversalGraphType());
  }

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

  // Get the indexing traversal path for indexing a given list of IDs
  // for a given expr
  ExprPath<ExprGroup> getIndexingPath(
      const Expr* expr,
      const std::vector<IterDomain*>& index_ids) const;

  ExprPath<ExprGroup> getPredicateIndexingPath(TensorView* tv, const Expr* expr)
      const;

  // Protect the index of the innermost loop with magic zero.
  //
  // NOTE: This just follows how the original indexer adds magic zero
  // to indices.
  //
  // TODO: Revisit if this is still necessary.
  std::vector<Val*> protectIndicesWithMagicZero(
      const std::vector<Val*>& indices,
      const std::vector<ForLoop*>& for_loops) const;

  // Check if a given fusion can be indexed with
  // TensorIndexer. Returns fals if the fusion uses features that have
  // only been implemented for the old indexer.
  static bool isSupported(Fusion* fusion);

 private:
  // Build a map of loop groups to their index Vals. See the comment
  // on loop_index_map_.
  void buildLoopIndexMap();

  const AllocationDomainInfo& getIndexAllocationInfo(TensorView* tv) const;

  // Returns the index map as well as its traversal path of given
  // index domains appearing in a given expr. Used by
  // getIndexFor.
  IndexingInfo computeIndex(
      const Expr* expr,
      const std::vector<IterDomain*>& index_ids,
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

  // For a given indexng traversal path toward allocation_domains,
  // return the contiguous domains and their strides that can provide
  // equivalent indexing results.
  //
  // Currently, only backward traversal is supported.
  std::pair<std::vector<ValGroup>, std::vector<Val*>> getContigDomainsAndStrides(
      const AllocationDomainInfo& alloc_info,
      const ExprPath<ExprGroup>& traversal_path) const;

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
  std::unordered_map<TensorView*, AllocationDomainInfo> alloc_info_;
};

} // namespace nvfuser

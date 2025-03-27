// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

/*
 * Index compute takes in a list of indices typically generated from the
 * surrounding for loop nest. The number of indicies are intended to match the
 * number of dimensions of the incomming TensorView which may have less or more
 * dimensions than its allocation domain due to split/merge operations.
 * Split/merge operations are then replayed backwards produce resulting
 * indices (based on input indices) that match the allocation dimension.
 *
 * For example with GLOBAL tensor:
 * TV[I, K]
 * TV[Io, Ii{4}, K] = TV.split(I, factor=4)
 * ALLOC: NONE
 * INDEX: indexCompute {i, j, k} -> {i * 4 + j, k}
 * FLATTENED_INDEX: {i * 4 + j, k} -> {(i * 4 + j) * K + k}
 * PREDICATE: {i * 4 + j, k} -> i * 4 + j < I
 *
 *
 * For example with SHARED tensor:
 *
 * global_TV[I, K]
 * global_TV[Io, Ii{4}, K] = global_TV.split(I, factor=4)
 * smem_TV.compute_at(global_TV, 1)
 * global_TV.parallelize(1, threadIDx.x)
 *
 * ALLOC: alloc(smem_TV, 4 x K)
 * INDEX: indexCompute(smem_TV, {threadIdx.x, k}) -> {threadIdx.x, k}
 * FLATTENED_INDEX: {threadIdx.x * 4 + j, k} -> {(threadIdx.x * 4 + j) * K + k}
 * PREDICATE: {threadIdx.x * 4 + j, k} -> threadIdx.x * 4 + j < I // Same as if
 * global
 *
 *
 * For example with LOCAL tensor:
 * global_TV[I, K, L]
 * global_TV[Io, Ii{4}, K, L] = global_TV.split(I, factor=4)
 * reg_TV.compute_at(global_TV, 2)
 * global_TV.parallelize(1, threadIDx.x)
 * global_TV{i, j, k, l} -> { i * 4 + j, k, l }
 * global_TV{ i * 4 + j, k, l } -> { (i * 4 + j) * K * L  +  k * L  +  l}
 *
 * ALLOC: alloc(reg_TV, K x L)
 * INDEX: {k, l} -> {k, l}
 * FLATTENED_INDEX: {k, l} -> {k * L + l}
 * PREDICATE: i * 4 + j < I && k < K && l < L ->  // Same as if global
 *
 * These indices can then be flattened later based on strides.
 */

namespace nvfuser {

class ContigIDs;
class LoopIndexing;
struct IndexFromIdGraph;
class TensorIndexer;

class IndexCompute : public BackwardVisitor {
 protected:
  using BackwardVisitor::handle;

  void dispatch(Expr*) override;

  void handle(Split*) override;
  void handle(Merge*) override;
  void handle(Swizzle*) override;
  void handle(Swizzle2D*) override;
  void handle(Resize*) override;

  // return extent_map_[id] if exists, else return id->extent()
  Val* getExtent(IterDomain* id) const;

  //! True if a domain is not used to index
  bool isZero(IterDomain* id) const;
  //! True if any dependent of a domain is not used to index
  bool hasZeroMerged(IterDomain* id) const;

  //! Returns the concrete ID from the compute at EXACT mode map if
  //! concrete_id_pass == true, otherwise returns id passed in.
  //! Helps unify the expr handling logic in reference domain and concrete id
  //! based traversal.
  IterDomain* maybeGetExactMapConcreteID(IterDomain* id) const;

  //! (Concrete indexing pass only)
  //!  Collect permissive index binding from the given expression.
  //! See also permissive_map_ and LoopIndexing::getBackwardOutOfLineExprList.
  void collectIndexIntoPermissiveMap(const LoopIndexing& loop_indexing);

  //! (Concrete indexing pass only)
  //!  Iterate through id_expr's input and pull index vals from permissive
  //! map, when both of the following are true:
  //!    1. the output id is missing in index_map_.
  //!    2. the output id is found in permissive map.
  void updateIndexMapFromPermissiveMap(const Expr* id_expr);

  //! Initialize unswitched_domain_map_ from the loop unswitched
  //! domains
  void initializeUnswitchDomainMap();

  //! Propagate unswitched map info from expr outputs to inputs
  void updateUnswitchedDomains(Expr* expr);

  //! Query if an IterDomain has a dependent unswitched domain
  bool hasUnswitchedDependentDomains(IterDomain* id) const;

  //! Query if the usual modulo propagation may be invalid for a merge
  //! inner path
  bool isModuloInvalidUnswitchedIndex(
      IterDomain* out_concrete_id,
      Val* out_ind,
      Val* inner_extent) const;

  // Tensor domain we're mapping back to allocation
  const TensorDomain* td_; // NOLINT

  // Map we update as we propagate backward, containing all IDs in the
  // propagation. Initial indices are mapped with this map at tv->domain()
  // and are back propagated to tv->getMaybeAllocationDomain(). This index_map_
  // keeps the indices at intermediate IterDomain's in that back propagation.
  std::unordered_map<IterDomain*, Val*> index_map_; // NOLINT

  // Map from IterDomain to their broadcasted extent. If a TV has I0*I1 but its
  // producer has B0*I1 this map will contain a mapping from the ID{B0*I1} to
  // the extent I0*I1. Also contains updated extents if we merge in a 0 index.
  // See zero_merged_in_.
  std::unordered_map<IterDomain*, Val*> extent_map_; // NOLINT

  // Keeps track of domains that do not contribute to indexing
  std::unordered_set<IterDomain*> zero_domains_; // NOLINT

  // This set keeps track of IterDomain's that have had a zero index merged into
  // them. This happens if we do something like tv->axis(0)->split(4) then
  // tv->computeAt(1, ...) if this tensor is in smem or lmem the backward
  // indexing would be (0, i) then when we do the backward computation that zero
  // and i would attempt to be merged together. We handle indices like these
  // specially.
  std::unordered_set<IterDomain*> zero_merged_in_;

  // IDs that are a result of contiguous merges
  std::unordered_set<IterDomain*> contig_ids_;

  // Mentions if we should propagate an index down a particular IterDomain path
  // if there's an option
  std::unordered_set<IterDomain*> preferred_paths_;

  // Temporary flag which tells IndexCompute to use concrete id's from the exact
  // map rather than the actual IDs used in the ID expressions.
  bool concrete_id_pass_ = false;

  // Mode of swizzle that are activated in this index compute
  //  instance. Will treat swizzles of different mode as no-op.
  // Currently data mode swizzles are handled same as before in IndexSwizzle
  //  pass, while loop mode swizzles are handled early on in concrete indexing
  //  pass. See also [Note on swizzle mode]
  SwizzleMode swizzle_mode_ = SwizzleMode::NoSwizzle;

  // (Concrete id pass only)
  // Contains the indexing math that could be resolved with only the
  //  iterdomains on the right of the consumer_tv's ca axis, i.e. the
  //  ones that corresponding to the loops that consumer_tv would not
  //  share with any of its consumers.
  // These indexing vals should be kept separate from index_map_ and
  //  should only be used when the indexing traversal follows the
  //  order defined in LoopIndexingAnalysis::traverseFromDomainVals.
  std::unordered_map<IterDomain*, Val*> permissive_index_map_;

  //! Leaf domains that have maximum index values for unswitch
  //! predicates. These domains need extra adjustments when going
  //! through module operations for merge inner domains as module does
  //! not always guarantee to preserve the maximum-ness property
  std::unordered_set<IterDomain*> unswitched_loop_domains_;

  //! Mapppings from unswitched IterDomains to their unswitched
  //! domains and their inner domains. Used to figure out if a module
  //! could invalidate the maximum-ness property of an unswitched index.
  //!
  //! Mappings are created in a bottom-up fashion from loop to root
  //! such that fine-grained domain mappings are kept as much as
  //! possible for making the modulo analysis most precise.
  //!
  //! Specifically, for the loop domains, this just maps unswitched
  //! domains, i.e., those included in unswitched_loop_domains_, to
  //! themselves. There'll be no mapping for those loop domains that
  //! are not included in unswitched_loop_domains_. The mappings of
  //! all other domains are defined based on their consumer
  //! domains. By default, they are also just mapped
  //! to themselves if any of the consumers are also mapped. However,
  //! when a domain is the input to a split, the mappings of the split output
  //! domains are tracked separately and the split input will be
  //! mapped to two sets of unswitched domains, one from the inner
  //! output and another from the outer output. The mapping info from
  //! the inner output is propagated as is, whereas the mapping info
  //! from the outer output is prepended with the inner output
  //! domain so that the unswitched domain list includes its inner
  //! domain. Note that the semantic of inner domains is defined based
  //! on split operations since they define propagated index math.
  //!
  //! The reason of tracking the information from split outer domains
  //! separately is to avoid adjusting the unswitched predicate index
  //! as much as possible. For example, here's a common transpose
  //! scheduling pattern:
  //!
  //! // Initial 2D tensor
  //! [i0, i1]
  //! // Create a square tile of 32x32
  //! -> [i0 / 32, 32, i1 / 32, 32]
  //! -> [i0 / 32 * i1 / 32, 32 * 32]
  //! // Factor out a small domain (commonly vectorized)
  //! -> [i0 / 32 * i1 / 32, 32 * 32 / 4, 4]
  //! // Factor out another domain (commonly parallelized by TIDx)
  //! -> [i0 / 32 * i1 / 32, 32 * 32 / 4 / 128, 128, 4]
  //!
  //! Notice that the merge of "32 * 32" is not contiguous, so we need
  //! to predicate its input domains by propagating index exprs
  //! through the merge inner path with "% 32". If any of the final
  //! loop domains are unswitched, we need to make sure the index expr
  //! sent through "% 32" is the maximum for the domain of extent
  //! "32". Conservatively, this can just be 31, however, that isn't
  //! always strictly required. For example, suppose the innermost
  //! domain of extent 4 is unswitched. Its initial index is
  //! 3. Propagating it through the merge inner path as usual is
  //! guaranteed to be correct. More generally, it's always the case
  //! when the inner extent of a merge is divisible by the extent of
  //! an unswitched output and its domains. Suppose also the third
  //! innermost domain is also unswitched, its initial index is 1. Its
  //! contribution through the merge inner path is zero as the initial
  //! index is multiplied by the extents of its inner domains, i.e.,
  //! 128 and 4, and they are divisible by the extent of the merge
  //! inner domain. Again, more generally, if the stride of an
  //! unswitched domain is a multiple of the inner extent of the merge
  //! operation producing the unswitched domain, there's no
  //! contribution from the unswitched domain, so it doesn't matter if
  //! it's maximum or not.
  //!
  //! In the above pattern, the second innermost domain is commonly
  //! parallelized with TIDx. Suppose it's also unswitched. Notice
  //! that there's no concern for that domain of invalding the
  //! maximum-ness property as threadIdx.x is the only valid initial
  //! index value for each thread. However, this is the reason we keep track
  //! of the split output contributions separately. More specifically,
  //! the intermediate domain of (32 * 32 / 4) will have an index of
  //! (1 * 128 + threadIdx.x), and the domain of (32 * 32) will have
  //! (1 * 128 * 4 + threadIdx.x * 4 + 3). As discussed above, we can
  //! reason about that the first and third components of this
  //! unswitched expression is safe with respect to the propagation
  //! with modulo by 32. The second component is also safe as that's
  //! the only valid index for the domain. If not separately tracked,
  //! all we could know would be that the extent of (32 * 32) is
  //! 1024. Since part of the dependent domains are parallelized the
  //! propagated index is not guaranteed to be 1023, so we would need
  //! to make a conservative decision to send 1023 to the merge inner
  //! path.
  std::unordered_map<IterDomain*, std::vector<std::deque<IterDomain*>>>
      unswitched_domain_map_;

 public:
  const std::unordered_map<IterDomain*, Val*>& indexMap() const {
    return index_map_;
  }

  const std::unordered_map<IterDomain*, Val*>& extentMap() const {
    return extent_map_;
  }

  const std::unordered_set<IterDomain*>& zeroDomains() const {
    return zero_domains_;
  }

  const std::unordered_set<IterDomain*>& zeroMergedIn() const {
    return zero_merged_in_;
  }

  // Propagate back from _td using initial_index_map
  IndexCompute(
      const TensorDomain* _td,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> _extent_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> _zero_merged_in,
      std::unordered_set<IterDomain*> preferred_paths = {});

  IndexCompute(
      const TensorDomain* _td,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> _extent_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> _zero_merged_in,
      const ContigIDs& contig_finder,
      std::unordered_set<IterDomain*> preferred_paths = {},
      std::unordered_set<IterDomain*> unswitched_domains = {});

  // Entry point used for using concrete id based traversal. This traversal is
  // assumed to start at loop IDs provided by initial_index_map.
  IndexCompute(
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> preferred_paths,
      std::unordered_set<IterDomain*> unswitched_domains = {});

  // Updates index_map, extent_map, and zero_merged_in based on id_map and
  // returns a new IndexCompute ready to be used.
  IndexCompute updateIndexCompute(
      const TensorDomain* new_td,
      const std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>&
          id_map,
      const ContigIDs& contig_finder) const;

  // Interface to run index traversal through loop indexing analysis result to
  // be used with the entry point for concrete id based traversal.
  void run(const LoopIndexing& loop_indexing);

  virtual void run();
};

//! Apply swizzle and update allocation indices accordingly
class IndexSwizzle : public IndexCompute {
 public:
  IndexSwizzle(
      const TensorView* tv,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> extent_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> zero_merged_in);

  IndexSwizzle(
      const TensorView* tv,
      const TensorDomain* domain,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> extent_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> zero_merged_in);

  void run() override;

 protected:
  using IndexCompute::handle;

  void dispatch(Expr* e) override;

  void handle(Swizzle2D* swizzle_2d) override;

 private:
  const TensorView* tv_ = nullptr;
  std::unordered_set<IterDomain*> swizzled_ids_;
};

//! Information about a predicate. By default, it corresponds to a
//! single logical domain but may cover multiple logial domains due to
//! contigous indexing.
class PredicateInfo {
  friend class Index;
  friend class TensorIndexer;

 public:
  const auto& startPredicate() const {
    return start_predicate_;
  }

  auto& startPredicate() {
    return start_predicate_;
  }

  const auto& startOffset() const {
    return start_offset_;
  }

  const auto& stopPredicate() const {
    return stop_predicate_;
  }

  auto& stopPredicate() {
    return stop_predicate_;
  }

  const auto& stopOffset() const {
    return stop_offset_;
  }

  const auto& predicatedDomains() const {
    return predicated_domains_;
  }

  const auto& loopDomains() const {
    return loop_domains_;
  }

  CircularBufferLoopStage loopStage() const {
    return loop_stage_;
  }

  //! Return a false RootPredicateInfo, i.e., both start and stop
  //! predicates are false.
  static PredicateInfo getFalseInfo();

 private:
  // prdicate for lower end
  Val* start_predicate_ = nullptr;
  // prdicate for upper end
  Val* stop_predicate_ = nullptr;
  // Offset of the start predicate
  Val* start_offset_ = nullptr;
  // Offset of the stop predicate
  Val* stop_offset_ = nullptr;
  // Track which domains are covered by the generated predicates
  std::unordered_set<IterDomain*> predicated_domains_;
  // Loops domains used for the predicate domains
  std::unordered_set<IterDomain*> loop_domains_;
  // Circular buffer loop stage if applicable
  CircularBufferLoopStage loop_stage_ = CircularBufferLoopStage::NotApplicable;
};

// Simple interface for IndexCompute
// If getComputeAtAxis and more generally TensorView const model is fixed, we
// can make the below tensorviews const.
class Index {
 private:
  // Producer indexing if it's in shared or local memory
  static std::vector<Val*> getNonGlobalProducerStridedIndices(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {});

  // Consumer indexing if it's in shared or local memory
  static std::vector<Val*> getNonGlobalConsumerStridedIndices(
      const TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {});

  // get the strides of a tensor used for the index lowering
  static std::vector<Val*> getStrides(TensorView* tv);

  // get the allocation indices of a consumer tensor
  static std::vector<Val*> getConsumerAllocationIndices(
      const TensorView* tv,
      const std::vector<ForLoop*>& loops,
      const IndexFromIdGraph& index_from_id_graph);

  // get the allocation indices of a producer tensor
  static std::vector<Val*> getProducerAllocationIndices(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {});

 public:
  // Producer if it's in global memory
  static std::vector<Val*> getGlobalProducerStridedIndices(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {});

  // Consumer indexing if it's in global memory
  static std::vector<Val*> getGlobalConsumerStridedIndices(
      TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<int, Val*>& override_index = {});

  // Indexing functions
  // Consumer = Producer
  // i.e. T0 = T1... -> T0 is the consumer, T1 is the producer
  // Producer indexing dispatch
  // The argument `generate_pointer` specifies whether to generate pointer for
  // the tensor. If global tensor, then generate T1.data. If shared memory
  // tensor, then use `cvta` ptx to convert shared memory address to unsigned
  // int for indexing. Search `toSmem` in the codebase for additional
  // information. This argument is effective only if the indexed tensor is a
  // shared memory or global tensor. On other memory type, this argument will
  // cause an error.
  static kir::TensorIndex* getProducerIndex(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {},
      bool generate_pointer = false,
      DataType as_type = DataType::Null);

  // Consumer index dispatch
  static kir::TensorIndex* getConsumerIndex(
      TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<int, Val*>& override_index = {},
      bool generate_pointer = false,
      DataType as_type = DataType::Null);

  //! Returns a vector of strided indices mapped onto the
  //! allocation domain of a producer tensor. The size of the returned
  //! vector is guaranteed to be equal to the number of axes of the
  //! indexing allocation domain.
  static Val* getProducerStridedIndices(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {},
      bool generate_pointer = false);

  //! Returns a vector of strided indices mapped onto the
  //! allocation domain of a consumer tensor. The size of the returned
  //! vector is guaranteed to be equal to the number of axes of the
  //! indexing allocation domain.
  static Val* getConsumerStridedIndices(
      TensorView* consumer,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<int, Val*>& override_index = {},
      bool generate_pointer = false);

  //! Returns the logical index linearized from a multi-dimension address into a
  //! linear memory address a consumer tensor. The returned index is intended to
  //! be used for the computation of some tensor factories, such as: iota and
  //! rand (for Philox pseudo random sequences)
  static Val* getLinearLogicalIndex(
      TensorView* consumer_tv,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops);

  //! Returns a vector of logical indices mapped onto the logical
  //! domain of a consumer tensor. The returned index is intended
  //! to be used for the computation of some tensor factories, such as:
  //! eye
  static std::vector<Val*> getConsumerPerDimLogicalIndex(
      TensorView* consumer_tv,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops);

  //! Returns a vector of logical indices mapped onto the logical
  //! domain of a producer tensor.
  static std::vector<Val*> getProducerPerDimLogicalIndex(
      TensorView* producer_tv,
      const TensorView* consumer_tv,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {});

  //! Take a consumer tensorview and loop nest and generates predicates
  //! associated with the concrete roots of the loop nest. Returns a list of
  //! predicates, and a list of concrete roots they're associated with. It
  //! is assumed that no predicate is required if index[i] is an index
  //! directly from a for loop. This will not catch all cases if we actually
  //! have static size information for example:
  //!
  //! TV[I].split(4)
  //! would produce the code:
  //! for(i : I/4)
  //!   for(j : 4)
  //!     if( i * 4 + j < TV.size(0))
  //!       TV[i * 4 + j]...
  //!
  //! However if we had TV.size[0] = 16 at "compile time" then we wouldn't
  //! need the predicate. This will be caught by canOmitPredicate in the
  //! predicate lowering
  //!
  //! unswitch_or_vec_loop is the for loop to start the unswitch like
  //! predicate, this is not a bool value as if we have an unswitch loop
  //! with a vectorized loop inside, we only want to base the "unswitch"
  //! like predicate on the vectorized loop.
  static std::vector<PredicateInfo> getReferenceRootPredicates(
      TensorView* consumer_tv,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      ForLoop* unswitch_or_vec_loop);

  //! Compute the result for iota
  static Val* iota(
      TensorView* consumer_tv,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      Val* start,
      Val* step,
      DataType dtype);

  //! Compute the result for eye
  static Val* eye(
      TensorView* consumer_tv,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops,
      DataType dtype);

  //! Compute the global index and the expected bytes for complete_tx mechanism
  //! for CpAsyncBulk.
  static std::pair<Val*, Val*> getCpAsyncBulkGmemIndex(
      const LoadStoreOp* ldst,
      Val* mbarrier,
      const std::vector<ForLoop*>& loops,
      const std::unordered_set<ForLoop*>& rotated_loops);
};

// Used for local and shared index mapping. Returns a map from loops
// to loop indices as well as a set of loops that do not contribute to
// indexing.
// TODO: could be cleaned up further.
std::pair<std::unordered_map<ForLoop*, Val*>, std::unordered_set<ForLoop*>>
indexMapFromTV(
    const TensorView* tv,
    const std::vector<ForLoop*>& loops,
    const std::unordered_set<ForLoop*>& rotated_loops,
    ForLoop* alloc_loop,
    bool as_consumer,
    ForLoop* circular_buffer_loop = nullptr);

//! Set "pragma unroll" required for loops that indexing of Local
//! tensors depends on.
//!
//! \param tv Indexed tensor
//! \param alloc_loop Allocation loop of tv
//! \param loops The current loop structure
//! \param id_map Producer-to-consumer map in case of indexing as producer
void ensureStaticIndexing(
    const TensorView* tv,
    ForLoop* alloc_loop,
    const std::vector<ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map = {});

struct PredicateDomainInfo {
 public:
  // Iteration domain to predicate
  IterDomain* id = nullptr;
  // The set of iteration domains that make up the id. If this is for
  // a non-divisible split, the set only contains the id itself. This
  // set is used to remove redundant predicates when gathering
  // unswitch predicates.
  std::unordered_set<IterDomain*> covered_ids;
  // True if this predicate is for an intermediate domain. Examples
  // include domains with non-divisible split and resized domains.
  bool is_intermediate_domain = false;
};

// Get all domains that need to be predicated due to non-divisible splits
std::vector<PredicateDomainInfo> getNonDivisibleConsumerDomainsToPredicate(
    TensorView* consumer_tv);

} // namespace nvfuser

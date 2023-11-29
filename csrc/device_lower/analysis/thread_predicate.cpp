// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/thread_predicate.h>

#include <debug.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/arith.h>

#include <c10/util/irange.h>
#include <algorithm>
#include <numeric>
namespace nvfuser {

namespace {

Val* getPredicatePerParallelType(
    ParallelType pt,
    const ThreadPredicateMap::PredicateInfo& pred_info) {
  auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);

  // If pt is not used or is proven to be one, no need to predicate.
  if (pt_dim == nullptr || pt_dim->isOneInt()) {
    return GpuLower::current()->kernel()->trueVal();
  }
  // When BID needs to be predicated, that means it's an output of a grid
  // reduction and only the last block index in that dimension has the right
  // value from the grid reduce.
  if (isParallelTypeBlockDim(pt) && pred_info.limited_types.get(pt)) {
    return SimplifyingIrBuilder::eqExpr(
        NamedScalar::getParallelIndex(pt),
        SimplifyingIrBuilder::subExpr(
            NamedScalar::getParallelDim(pt),
            GpuLower::current()->kernel()->oneVal()));
  }

  const auto& broadcast_rd_indices_map = pred_info.broadcast_rd_indices_map;
  if (auto it = broadcast_rd_indices_map.find(pt);
      it != broadcast_rd_indices_map.end()) {
    // skip concretized broadcast root domains
    const auto& broadcast_rd_indices = it->second;
    Val* zero = GpuLower::current()->kernel()->zeroVal();
    Val* pred = GpuLower::current()->kernel()->trueVal();
    for (auto broadcast_rd_index : broadcast_rd_indices) {
      pred = SimplifyingIrBuilder::logicalAndExpr(
          pred, SimplifyingIrBuilder::eqExpr(broadcast_rd_index, zero));
    }
    return pred;
  }

  return SimplifyingIrBuilder::eqExpr(
      NamedScalar::getParallelIndex(pt),
      GpuLower::current()->kernel()->zeroVal());
}

} // namespace

Val* ThreadPredicateMap::getPredicateFromPredicateInfo(
    const ThreadPredicateMap::PredicateInfo& pred_info,
    const ParallelTypeBitmap& mask) {
  const auto pred_types =
      (pred_info.limited_types | pred_info.redundant_types) & mask;

  if (pred_types.none()) {
    return GpuLower::current()->kernel()->trueVal();
  }

  Val* pred = nullptr;
  for (const auto pt : pred_types) {
    const auto tp = getPredicatePerParallelType(pt, pred_info);
    pred = SimplifyingIrBuilder::logicalAndExpr(pred, tp);
  }
  NVF_ERROR(pred != nullptr);

  return pred;
}

namespace {

// Build redundant predicate flags. Will be stored as
// PredicateInfo.redundant_types for the given tensor.
ParallelTypeBitmap avoidRedundantWrites(const TensorView* out_tv) {
  // If the memory type is Local, it's fine to write into it always as
  // it's thread local. If it's Global, it's also fine to let each
  // thread do its own write, unless out_tv is an output of a
  // reduction. Standard reductions (forget gridReduce for the sake of this
  // argument) directly into global memory buffers accumulate into the global
  // memory buffer. If this is done redundantly then it could lead to incorrect
  // results. Correctness issues here can come from smem aliasing, smem
  // reductions or gmem reductions because the reduction itself performs an
  // update to a value, not just a set. For performance it's safe to ommit the
  // redundant writes to gmem or smem, this comment is just specifying it's not
  // always just a performance optimization, but can also be a correctness
  // requirement.
  //
  // For now this is enabled for shared memory buffers, global memory buffers
  // undergoing a reduction, and global memory buffers with terminating outputs.
  // This could be extended to all global memory buffer transactions, but in the
  // test Indexing11 there's a case where an intermediate global buffer is set
  // and used to perform a broadcast. At the moment a grid sync is not being
  // inserted here, and it's generally safe since it's just a set. We could
  // enable this more generally for global memory buffers, but would have to
  // insert a sync or a grid broadcast in that example. For now the approach is
  // to only do this on a grid buffer (not undergoing a reduction) if there are
  // no other uses in the kernel.
  //
  // TODO: Revisit if something like Indexing11 could be happening at the same
  // time of a global reduction in a way that could produce an incorrect result.
  const bool is_reduction = ir_utils::isReductionOp(out_tv->definition());
  if (!(out_tv->getMemoryType() == MemoryType::Shared ||
        (out_tv->getMemoryType() == MemoryType::Global && is_reduction) ||
        (out_tv->getMemoryType() == MemoryType::Global &&
         out_tv->uses().empty()))) {
    return ParallelTypeBitmap();
  }

  ParallelTypeBitmap pred;
  // Track which TID types are not used to find redundant parallel
  // types. Only TID types are checked if the tensor is on shared
  // memory otherwise on global memory all TID and BID types are checked.
  ParallelTypeBitmap unused_types;
  // Initially all types are conservatively assumed to not be used.
  unused_types = ~unused_types;
  for (auto out_tv_id : out_tv->getLeafDomain()) {
    auto pt = out_tv_id->getParallelType();
    if (!isParallelTypeThread(pt)) {
      continue;
    }
    // If the axis is a broadcast domain and is parallelized by TID,
    // it is sufficient to use just one thread since the tensor is on
    // shared memory.
    if ((out_tv->getMemoryType() == MemoryType::Shared &&
         out_tv_id->isBroadcast() && isParallelTypeThreadDim(pt)) ||
        // Protect against global memory and is_reduction as we don't want to
        // predicate grid dimensions as codegen will complain predication on
        // block dimensions is not allowed in grid reductions. The old
        // grid reduction runtime kernel does not differentiate
        // non-reduction and predicated parallel types, so the sync
        // integer buffer would need to be expanded even for
        // predicated parallel types, which is not what
        // getGridSyncBufferSize does. The right thing here is either:
        // retire the old grid reduction kernel, or update the kernel
        // to propertly ignore predicated types. The new kernel is
        // significantly complex and has not been tested, so the
        // latter option seems more reasonable for now. See #1671.
        (!is_reduction && out_tv->getMemoryType() == MemoryType::Global &&
         out_tv_id->isBroadcast() && isParallelTypeThread(pt))) {
      pred.set(pt);
    }
    unused_types.clear(pt);
  }

  const auto& par_dim_map = GpuLower::current()->parallelDimensionMap();

  for (const auto pt : unused_types) {
    // For shared memory tensors, unused BID isn't redundant
    if (isParallelTypeBlockDim(pt) &&
        out_tv->getMemoryType() == MemoryType::Shared) {
      continue;
    }
    // If the pt is not used or is proven to be one, it is not
    // really redundant.
    auto pt_dim = par_dim_map.get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    pred.set(pt);
  }

  return pred;
}

// If tv is an output of a reduction with unused parallel types, those
// unused parallel types need to be predicated if the tensor is on
// global memory.
ParallelTypeBitmap getReductionPredicateForUnusedParallelTypes(
    const TensorView* tv,
    const ThreadPredicateMap::PredicateInfo& pred_info) {
  auto tv_def = tv->definition();
  if (!(tv_def && ir_utils::isReductionOp(tv_def) &&
        tv->getMemoryType() == MemoryType::Global)) {
    return {};
  }

  // Unused types are set as redundant types of tv
  return pred_info.redundant_types;
}

} // namespace

// Update the reduction_deps bitset based on provided Expr
void ThreadPredicateMap::updateBitSet(const Expr* expr) {
  FUSER_PERF_SCOPE("GpuLower::Lower::ThreadPredicateMap::updateBitSet");

  auto tv_out = ir_utils::getTvOutput(expr);
  if (tv_out == nullptr) {
    return;
  }

  // If all of the inputs are not updated and all of the outputs have
  // already mappings, don't do anything
  if (std::all_of(
          ir_utils::filterByType<TensorView>(expr->inputs()).begin(),
          ir_utils::filterByType<TensorView>(expr->inputs()).end(),
          [this](TensorView* tv) {
            return updated_tvs_.find(tv) == updated_tvs_.end();
          }) &&
      std::all_of(
          ir_utils::filterByType<TensorView>(expr->outputs()).begin(),
          ir_utils::filterByType<TensorView>(expr->outputs()).end(),
          [this](TensorView* tv) { return find(tv) != end(); })) {
    return;
  }

  // Which predicates were set for the inputs
  ParallelTypeBitmap input_preds;

  // Which dims are reductions in inputs
  ParallelTypeBitmap input_reductions;

  // Parallel types used by the output tensor
  ParallelTypeBitmap output_ptypes;
  std::for_each(
      ir_utils::getTvOutput(expr)->getLeafDomain().begin(),
      ir_utils::getTvOutput(expr)->getLeafDomain().end(),
      [&](auto out_tv_id) {
        if (out_tv_id->isThread()) {
          output_ptypes.set(out_tv_id->getParallelType());
        }
      });

  // Run through inputs and update bitsets
  for (const auto* inp : expr->inputs()) {
    if (!ir_utils::isTV(inp))
      continue;

    auto tv_inp = inp->as<TensorView>();

    // If tv_inp was an output of a multi-output expression, just change it to a
    // consistent sibling to use a single predicate name.
    if (auto tv_def = tv_inp->definition()) {
      if (tv_def->outputs().size() > 1) {
        tv_inp = ir_utils::getTvOutput(tv_def);
      }
    }

    NVF_ERROR(
        thread_predicates_.find(tv_inp) != thread_predicates_.end(),
        "Thread predicate map was not initialized, couldn't find ",
        inp->toString());

    const auto& pred_info = at(tv_inp);

    ParallelTypeBitmap id_reductions;
    ParallelTypeBitmap id_bcasts;
    ParallelTypeBitmap id_ptypes;

    for (auto id : tv_inp->getLeafDomain()) {
      if (id->isThread()) {
        id_ptypes.set(id->getParallelType());
        if (id->isReduction() &&
            !GpuLower::current()->fusedReductionInfo().isAllreduce(id)) {
          id_reductions.set(id->getParallelType());
        }
        if (id->isBroadcast() &&
            GpuLower::current()->concretizedBroadcastDomains()->isConcretized(
                id)) {
          id_bcasts.set(id->getParallelType());
        }
      }
    }

    // Validate the combination of ptypes, reductions, bcasts
    for (const auto i : c10::irange(ParallelTypeBitmap::kNumParallelTypes)) {
      if (input_reductions[i]) {
        if (id_ptypes[i]) {
          NVF_ERROR(
              id_reductions[i],
              "Mismatched parallelized reductions found on inputs of epxr: ",
              expr);
          NVF_CHECK(
              !id_bcasts[i],
              "Invalid broadcast and reduction combination, tried to parallelize both with the same thread dim: ",
              inp);
        }
      }
    }

    // Figure out which dims bcast wants to reset
    auto this_input_preds = pred_info.limited_types;
    const auto bcast_reset_mask = ~(this_input_preds & id_bcasts);
    this_input_preds &= bcast_reset_mask;

    // If the input is on shared or global memory and is predicated,
    // and the output is parallelized by the predicate type, a RAW
    // sync is automatically inserted, so the predication can be
    // cleared.
    if (tv_inp->getMemoryType() == MemoryType::Shared ||
        tv_inp->getMemoryType() == MemoryType::Global) {
      auto raw_sync_reset_mask = (this_input_preds & output_ptypes);
      // If it's on shared memory, only TID predicates can be
      // cleared.
      if (tv_inp->getMemoryType() == MemoryType::Shared) {
        raw_sync_reset_mask &= ParallelTypeBitmap().setAllTID();
      }
      this_input_preds &= ~raw_sync_reset_mask;
    }

    input_preds |= this_input_preds;

    id_reductions |=
        getReductionPredicateForUnusedParallelTypes(tv_inp, at(tv_inp));

    // Accumulate
    input_reductions |= id_reductions;
  }

  // Update map for this tv, before accumulating to other inputs
  // Add any reductions this id has to any input predicates
  auto output_preds = input_preds | input_reductions;

  // Run through outputs and set bitset predicates
  for (auto* out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
    auto redundant_types = avoidRedundantWrites(out_tv);
    update(out_tv, output_preds, redundant_types);
  }
}

namespace {

//! A simple backward data flow pass:
//!  This pass propagates information backward to annotate "redundant use
//!  chain"'s.
//! The reason this is needed is that, say for example, if we have a chain
//! of register-to-register ops that begins with a redundant shared mem write
//! and ends with an op that non-redundantly uses the result, we'd need to
//! insert a sync at the begining of the register-to-register chain.
//!
//! The same mechanism also applies in the case of a register/sharedmem chain
//! that starts and ends with global memory read/write.
//!
//! The propagation rule is summarized as follows:
//!
//!   Shared TV val:
//!      Reset all block redundant info to its own redundant write info
//!      Backpropagate grid redundant info
//!   Global TV val:
//!      Reset all redundant info to its own redundant write info
//!   Local Tv val:
//!      Backpropagate all redundant info
//!   Exprs:
//!      Propagate redundant info backwards from outputs to inputs:
//!        For each parallel type,
//!          The parallel type is redundantly used in the expr input
//!          only if all of the outputs redundantly use the same type.
class RedundantUseAnalysis : BackwardVisitor {
 public:
  RedundantUseAnalysis(Fusion* fusion, const ThreadPredicateMap& pred_map)
      : fusion_(fusion), pred_map_(pred_map) {
    traverseTo(fusion->terminatingMathVals());
  }

  //! Returns a bit map signifying the parallel dimensions
  //!  on which the given tv is redundantly used. On these
  //!  dimensions not all threads/blocks are required to
  //!  hold valid value for their dependent computations.
  ParallelTypeBitmap getRedundantUseBitMap(const TensorView* tv) {
    // Since all tv's consumers are visited at this point, we
    //  can aggregate the final redundant use info for this tv.
    bool not_used_by_tensor_op = true;
    for (auto expr : fusion_->unordered_uses(tv)) {
      // There are ops, especially GetMetaData, that takes TensorView as input
      // and output a non-tensor object. These ops should be treated as scalars,
      // and we do not need to worry about thread predicate.
      if (ir_utils::isTvOp(expr)) {
        not_used_by_tensor_op = false;
        break;
      }
    }
    if (not_used_by_tensor_op) {
      // Base case, un-used is also not redundantly used
      return ParallelTypeBitmap();
    } else {
      // Aggregate redundant use as a conjunction of all
      //  consumer's redundant consumer info propagated
      //  backward from their consumer chains.
      ParallelTypeBitmap redundant_use;
      redundant_use.setAllBID();
      redundant_use.setAllTID();
      for (auto expr : fusion_->unordered_uses(tv)) {
        if (!ir_utils::isTvOp(expr)) {
          // For non-TV op that takes a tensor as input, such as, GetMetaData
          // we should not consider it for predication.
          continue;
        }
        redundant_use &= redundant_expr_use_map_.at(expr);
      }

      return redundant_use;
    }
  }

 private:
  using BackwardVisitor::handle;

  void handle(TensorView* tv) final {
    auto redundant_tv_map = pred_map_.getPredicateInfo(tv).redundant_types;

    // Setup the info to propagate backward for the producer tv's and
    //  expressions.
    ParallelTypeBitmap& redundant_consumer_map =
        redundant_consumer_parallel_type_map_[tv];

    // Initialize the use map to the redundant pred result
    redundant_consumer_map = redundant_tv_map;

    if (tv->getMemoryType() == MemoryType::Shared) {
      backPropagateRedundantUse(
          redundant_consumer_map,
          tv,
          false, // no propagate TID redundant use for shared tv
          true //  propagate BID redundant use
      );

    } else if (tv->getMemoryType() == MemoryType::Local) {
      backPropagateRedundantUse(
          redundant_consumer_map,
          tv,
          true, // propagate TID redundant use
          true // propagate BID redundant use
      );
    }
  }

  void backPropagateRedundantUse(
      ParallelTypeBitmap& use_map,
      TensorView* tv,
      bool propagate_tid,
      bool propagate_bid) {
    // Clear the propagated part of the original result
    if (propagate_bid) {
      use_map.setAllBID();
    }
    if (propagate_tid) {
      use_map.setAllTID();
    }

    for (auto expr : fusion_->unordered_uses(tv)) {
      // Assuming all consumer expressions have been
      //  visited at this point since we are traversing
      //  backward.
      auto expr_use_map = redundant_expr_use_map_.at(expr);
      // Clear the part of expression use map that does not
      //  need to be propagated.
      if (!propagate_bid) {
        expr_use_map.setAllBID();
      }
      if (!propagate_tid) {
        expr_use_map.setAllTID();
      }

      // Accumulate expression redundant usage
      //  This implements the `only if all` part in
      //   the discussion above.
      use_map &= expr_use_map;
    }
  }

  void dispatch(Expr* expr) final {
    if (ir_utils::isTvOp(expr)) {
      // Initialize redundant info for current expr
      std::optional<ParallelTypeBitmap> maybe_expr_pred_map;

      for (auto consumer_tv :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto tv_redundant_bitmap =
            redundant_consumer_parallel_type_map_.at(consumer_tv);

        if (maybe_expr_pred_map.has_value()) {
          // Accumulate redundant info of this tv output.
          maybe_expr_pred_map.value() &= tv_redundant_bitmap;
        } else {
          // Copy the tv's redundant info as the first valid case.
          maybe_expr_pred_map = tv_redundant_bitmap;
        }
      }

      NVF_ERROR(
          maybe_expr_pred_map.has_value(), "TV op not having a tv output");
      redundant_expr_use_map_[expr] = maybe_expr_pred_map.value();
    }
  }

 private:
  // Populated redundant use information on the used tv's
  //  This map provides information on if the given tv does not require
  // valid data from its producer on any parallel dimensions.
  // For example:
  //  T1_local = T0_shared[...]
  //  if(tid.x == 0)
  //    T2_shared[...] = T1_local[...]
  // Then tidx would be redundant consumer parallel type
  //  for T1, as T1 is local tensor, and only threads satisfying
  //  tidx == 0 would need to provide a valid data.
  // In this case, not all threads would need to read correct data
  //  from T0_shared, which would help remove some sync's.
  std::unordered_map<const TensorView*, ParallelTypeBitmap>
      redundant_consumer_parallel_type_map_;

  // Populated redundant use information on the used tv expressions.
  std::unordered_map<const Expr*, ParallelTypeBitmap> redundant_expr_use_map_;

  // Short cut to the owning fusion of this analysis.
  Fusion* fusion_ = nullptr;

  // Short cut to the active pred map analysis this pass is running as part of.
  const ThreadPredicateMap& pred_map_;
};

} // namespace

namespace {
// This class removes the redundant write to gmem when an output tensor has a
// leaf domain merged from concretized broadcast root domain and parallelized by
// thread/block id. issue https://github.com/csarofeen/pytorch/issues/2125
class ConcretizedBroadcastRedundantWriteRemover {
 public:
  // interface to run the check
  ConcretizedBroadcastRedundantWriteRemover(const TensorView* out_tv)
      : tv_(out_tv), root_domain_(out_tv->getMaybeRFactorDomain()) {
    setCandidateLeafDomains();
    if (candidate_leaf_domains_.empty()) {
      return;
    }
    setConcretizedBroadcastRootDomain();
    if (concretized_broadcast_root_domains_.empty()) {
      return;
    }
    for (auto ld : candidate_leaf_domains_) {
      // find all root domains that are merged to this leaf domain.
      const std::vector<IterDomain*>& merged_root_domains =
          getRootDomainsMergedToLeaf(ld);
      if (!merged_root_domains.empty()) {
        const ParallelType& pt = ld->getParallelType();
        const std::vector<Val*>& broadcast_root_indices =
            getIndexOfBroadcastRootDomains(merged_root_domains, pt);
        write_index_map_[pt] = broadcast_root_indices;
      }
    }
  }
  // interface to get results
  const std::unordered_map<ParallelType, std::vector<Val*>>& getWriteIndexMap() {
    return write_index_map_;
  }

 private:
  const TensorView* tv_;
  const std::vector<IterDomain*>& root_domain_;
  // leaf domains that are merged from root domains and parallelized by thread
  // blocks
  std::vector<IterDomain*> candidate_leaf_domains_;
  // map from root domain to its concretized domain
  std::unordered_map<IterDomain*, IterDomain*>
      concretized_broadcast_root_domains_;
  // map from parallel type to its write index
  std::unordered_map<ParallelType, std::vector<Val*>> write_index_map_;

  void setCandidateLeafDomains() {
    for (auto ld : tv_->domain()->leaf()) {
      const ParallelType& pt = ld->getParallelType();
      auto merge = dynamic_cast<Merge*>(ld->definition());
      if (isParallelTypeThread(pt) && merge) {
        candidate_leaf_domains_.push_back(ld);
      }
    }
  }

  void setConcretizedBroadcastRootDomain() {
    std::shared_ptr<const ComputeAtMap> caMap = GpuLower::current()->caMap();
    for (auto leaf_id : candidate_leaf_domains_) {
      auto loop_concrete_id =
          caMap->getConcreteMappedID(leaf_id, IdMappingMode::LOOP);
      auto concrete_root_vals = IterVisitor::getInputsTo({loop_concrete_id});
      auto concrete_root_ids =
          ir_utils::filterByType<IterDomain>(concrete_root_vals);

      // get concretized root domains
      for (auto rd : root_domain_) {
        if (!rd->isBroadcast()) {
          continue;
        }
        auto it = std::find_if(
            concrete_root_ids.begin(),
            concrete_root_ids.end(),
            [&caMap, &rd](auto concrete_root_id) {
              return caMap->areMapped(
                  rd, concrete_root_id, IdMappingMode::PERMISSIVE);
            });
        if (it == concrete_root_ids.end()) {
          // Failed to find the concrete ID. This could happen in complex
          // broadcast and computeAt patterns. Not addressed for now
          continue;
        }
        auto concrete_root_id = *it;
        NVF_ERROR(
            concretized_broadcast_root_domains_.emplace(rd, concrete_root_id)
                .second);
      }
    }
  }

  // Find all the root domains that are merged to the leaf domain.
  // e.g. Root: [I1,B2,B3] -> Leaf: [I1*B2*B3]
  std::vector<IterDomain*> getRootDomainsMergedToLeaf(IterDomain* id) {
    std::vector<IterDomain*> merged_root_domains;
    std::vector<int> index_root_domain;
    std::vector<IterDomain*> intermediate_domains = root_domain_;
    auto all_exp = StmtSort::getExprsBetween(
        {root_domain_.begin(), root_domain_.end()}, {id});
    for (Expr* expr : all_exp) {
      if (auto* merge = dynamic_cast<Merge*>(expr)) {
        auto outer_iter =
            std::find(root_domain_.begin(), root_domain_.end(), merge->outer());
        auto inner_iter =
            std::find(root_domain_.begin(), root_domain_.end(), merge->inner());

        if (outer_iter != root_domain_.end()) {
          merged_root_domains.emplace_back(*outer_iter);
          index_root_domain.emplace_back(
              std::distance(root_domain_.begin(), outer_iter));
        }
        if (inner_iter != root_domain_.end()) {
          merged_root_domains.emplace_back(*inner_iter);
          index_root_domain.emplace_back(
              std::distance(root_domain_.begin(), inner_iter));
        }
      } else {
        // current analysis of predication is only valid if all the exprs
        // between this lead domain and root domains are merge
        return std::vector<IterDomain*>();
      }
    }
    // The following sort is added because in NVFuserTest.FusionIssue2076_CUDA
    // the order is [I3, I1, B2] while the correct order should be [I1, B2, I3]
    size_t n_elements = merged_root_domains.size();
    NVF_ERROR(n_elements, "The number of merged root domains should > 0");
    std::vector<int> indices(n_elements);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return index_root_domain.at(a) < index_root_domain.at(b);
    });
    std::vector<IterDomain*> merged_root_domains_sorted(n_elements);
    for (size_t i = 0; i < n_elements; ++i) {
      merged_root_domains_sorted.at(i) = merged_root_domains.at(indices.at(i));
    }

    return merged_root_domains_sorted;
  }

  // Get the index of the leaf domain if we skip the broadcasted root domains
  std::vector<Val*> getIndexOfBroadcastRootDomains(
      const std::vector<IterDomain*>& merged_root_domains,
      ParallelType pt) {
    const int ndim = (int)merged_root_domains.size();
    // get the stride if we index the leaf domain using its root domains
    std::vector<Val*> root_stride(ndim);
    root_stride.at(ndim - 1) = GpuLower::current()->kernel()->oneVal();
    for (int i = ndim - 2; i >= 0; i--) {
      auto pre_crd = merged_root_domains.at(i + 1);
      Val* pre_extent = pre_crd->isBroadcast()
          ? concretized_broadcast_root_domains_.at(pre_crd)->extent()
          : pre_crd->extent();
      root_stride.at(i) = IrBuilder::mulExpr(root_stride.at(i + 1), pre_extent);
    }
    // convert the linear index of the leaf domain to the indices of the root
    // domains
    Val* remaining_index = NamedScalar::getParallelIndex(pt);
    std::vector<Val*> index_broadcast_root_domains;
    index_broadcast_root_domains.reserve(ndim);
    for (int i = 0; i < ndim; i++) {
      Val* root_index_at_i =
          IrBuilder::divExpr(remaining_index, root_stride.at(i));
      remaining_index = IrBuilder::modExpr(remaining_index, root_stride.at(i));
      if (merged_root_domains.at(i)->isBroadcast()) {
        index_broadcast_root_domains.emplace_back(root_index_at_i);
      }
    }
    return index_broadcast_root_domains;
  }
};
} // namespace

// This function is to avoid redundant writes to global memory
// when the tensor has a leaf domain merged from concretized
// broadcast domains and parallelized by thread/block id.
// Only do the write when the index of the leaf domain equals
// write_index_map_.at(pt) where pt is the parallel type.
void ThreadPredicateMap::avoidConcretizedBroadcastRedundantWrite(
    const TensorView* out_tv) {
  ConcretizedBroadcastRedundantWriteRemover redundant_write_remover(out_tv);
  const auto& broadcast_rd_indices_map =
      redundant_write_remover.getWriteIndexMap();
  if (!broadcast_rd_indices_map.empty()) {
    thread_predicates_[out_tv].broadcast_rd_indices_map =
        broadcast_rd_indices_map;
    for (const auto& iter : broadcast_rd_indices_map) {
      thread_predicates_[out_tv].redundant_types.set(iter.first);
    }
  }
}

void ThreadPredicateMap::build(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::ThreadPredicateMap");

  // Initialize mapping for input tensors
  for (auto inp : fusion->inputs()) {
    if (auto tv = dynamic_cast<const TensorView*>(inp)) {
      update(tv, ParallelTypeBitmap(), ParallelTypeBitmap());
    }
  }
  for (auto expr : fusion->exprs()) {
    updateBitSet(expr);
  }

  for (auto tv : ir_utils::allTvs(fusion)) {
    if (tv->getMemoryType() == MemoryType::Global) {
      avoidConcretizedBroadcastRedundantWrite(tv);
    }
  }

  updated_tvs_.clear();
  populateRedundantUseMap(fusion);
}

void ThreadPredicateMap::populateRedundantUseMap(Fusion* fusion) {
  RedundantUseAnalysis redundant_use(fusion, *this);
  for (auto& it : thread_predicates_) {
    it.second.redundant_use_types =
        redundant_use.getRedundantUseBitMap(it.first);
  }
}

ThreadPredicateMap::const_iterator ThreadPredicateMap::find(
    const TensorView* tv) const {
  return thread_predicates_.find(tv);
}

ThreadPredicateMap::const_iterator ThreadPredicateMap::end() const {
  return thread_predicates_.end();
}

const ThreadPredicateMap::PredicateInfo& ThreadPredicateMap::at(
    const TensorView* tv) const {
  return thread_predicates_.at(tv);
}

ThreadPredicateMap::PredicateInfo& ThreadPredicateMap::at(
    const TensorView* tv) {
  return thread_predicates_.at(tv);
}

ThreadPredicateMap::PredicateInfo ThreadPredicateMap::getPredicateInfo(
    const TensorView* tv) const {
  auto pred_info = thread_predicates_.at(tv);
  // Do not predicate a paralell type if it is a parallel bcast domain
  if (dynamic_cast<BroadcastOp*>(tv->definition())) {
    auto parallel_bcast = getParallelBroadcastDomains(tv);
    pred_info.limited_types ^= parallel_bcast;
  }
  return pred_info;
}

ParallelTypeBitmap ThreadPredicateMap::getPredicatedParallelTypes(
    const TensorView* tv) const {
  auto pred_info = getPredicateInfo(tv);
  return pred_info.limited_types | pred_info.redundant_types;
}

bool ThreadPredicateMap::update(
    const TensorView* tv,
    const ParallelTypeBitmap& limited_types,
    const ParallelTypeBitmap& redundant_types) {
  return update(tv, {limited_types, redundant_types});
}

bool ThreadPredicateMap::update(
    const TensorView* tv,
    const PredicateInfo& pred_info) {
  auto existing_mapping_it = thread_predicates_.find(tv);
  if (existing_mapping_it != end()) {
    PredicateInfo& existing_info = existing_mapping_it->second;
    if (existing_info == pred_info) {
      return false;
    } else {
      existing_info = pred_info;
      markAsUpdated(tv);
      return true;
    }
  } else {
    thread_predicates_.insert({tv, pred_info});
    markAsUpdated(tv);
    return true;
  }
}

Val* ThreadPredicateMap::getPredicate(
    const TensorView* tv,
    ParallelTypeBitmap mask) const {
  NVF_ERROR(find(tv) != end(), "Couldn't find ", tv);
  auto pred_info = getPredicateInfo(tv);
  return getPredicateFromPredicateInfo(pred_info, mask);
}

ParallelTypeBitmap ThreadPredicateMap::getParallelBroadcastDomains(
    const TensorView* tv) const {
  // If no pred is found for tv, no predicate is necessary
  if (find(tv) == end()) {
    return ParallelTypeBitmap();
  }

  ParallelTypeBitmap parallel_broadcast;

  const auto& iter_domains = tv->getLeafDomain();

  // If the output is on shared memory, assume that all subsequent
  // reads from all threads in its CTA can be done with no parallel
  // broadcast. Only one thread will write to shared memory followed
  // by a proper _syncthreads.
  const bool output_smem = tv->getMemoryType() == MemoryType::Shared;

  for (auto id : iter_domains) {
    if (!id->isBroadcast() ||
        !GpuLower::current()->concretizedBroadcastDomains()->isConcretized(
            id)) {
      continue;
    }
    if (id->isBlockDim() || (!output_smem && id->isThreadDim())) {
      parallel_broadcast.set(id->getParallelType());
    }
  }

  return parallel_broadcast & at(tv).limited_types;
}

ParallelTypeBitmap ThreadPredicateMap::getRedundantConsumerType(
    Expr* expr) const {
  std::optional<ParallelTypeBitmap> result;
  for (auto out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
    auto out_tv_redundant_map = getPredicateInfo(out_tv).redundant_use_types;
    if (!result.has_value()) {
      result = out_tv_redundant_map;
    } else {
      result.value() &= out_tv_redundant_map;
    }
  }

  NVF_ERROR(result.has_value(), "ThreadPredicateMap : TV op assumed");
  return result.value();
}

void ThreadPredicateMap::markAsUpdated(const TensorView* tv) {
  updated_tvs_.insert(tv);
}

void ThreadPredicateMap::print() const {
  debug() << "\nThreadPredicateMap\n";
  debug() << "--------------------------------\n";
  for (const auto& kv : thread_predicates_) {
    debug() << "T" << kv.first->name();
    debug() << " {" << kv.second.limited_types.toString() << "}\n";
    debug() << "{" << kv.second.redundant_types.toString() << "}\n";
    debug() << "{" << kv.second.redundant_use_types.toString() << "}\n";
  }
  debug() << "--------------------------------\n\n";
}

} // namespace nvfuser

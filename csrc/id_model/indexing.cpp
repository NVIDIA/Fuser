// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <expr_simplifier.h>
#include <id_model/circular_buffer_indexing.h>
#include <id_model/contiguity.h>
#include <id_model/id_model_index_compute.h>
#include <id_model/indexing.h>
#include <id_model/indexing_traversal.h>
#include <id_model/indexing_utils.h>
#include <id_model/predicate_indexing.h>
#include <id_model/to_string.h>
#include <id_model/utils.h>
#include <index_compute.h>
#include <ir/builder.h>
#include <ir/graphviz.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>
#include <swizzle.h>
#include <val_graph_visitor.h>

#include <algorithm>
#include <fstream>

namespace nvfuser {

namespace {

// True if a given domain is a loop domain of a given tensor and its
// loop is partitioned with respect to the memory type of the tensor
bool isPartitionedLoop(const TensorView* tv, IterDomain* id) {
  // False if id is not a loop ID
  if (std::find(tv->getLoopDomain().begin(), tv->getLoopDomain().end(), id) ==
      tv->getLoopDomain().end()) {
    return false;
  }

  // If the memory of this domain is partitioned with respect to the
  // parallel type of the domain, there's no allocation for the domain
  return ir_utils::isMemoryPartitionedAcross(
      tv->getMemoryType(), id->getParallelType());
}

bool isSizeOneDomain(IterDomain* id) {
  return id->isBroadcast() || id->extent()->isOneInt();
}

// True if a given domain of a tensor *may* require allocation
bool mayRequireAllocation(const TensorView* tv, IterDomain* id) {
  // Conditions to consider:
  // - Fully partitioned
  // - Size one: Allocation is done based on the promotion ID, but as
  // long as the original ID has size one, its allocation should
  // remain size one.
  // - Reduction: Check the original ID, not the promotion, which may
  //   be a reduction ID even though the original ID is not a reduction
  return !isPartitionedLoop(tv, id) && !isSizeOneDomain(id) &&
      !id->isReduction();
}

// Get the allocation stride of a given allocation domain
Val* getStrideOfGlobalMemoryTensor(TensorView* tv, int64_t alloc_dim) {
  NVF_ERROR(tv->getMemoryType() == MemoryType::Global);

  // Allocation domains can include reduction domains, but
  // alloc_stride arrays do not.
  const auto& alloc_dom = tv->getMaybeAllocationDomain();
  int64_t stride_dim = -1;
  for (const auto i : c10::irange(alloc_dim + 1)) {
    if (alloc_dom.at(i)->isReduction()) {
      continue;
    }
    ++stride_dim;
  }

  NVF_ERROR(stride_dim != -1);

  return IrBuilder::getItemExpr(
      IrBuilder::getAttrExpr(IrBuilder::metadataExpr(tv), "alloc_stride"),
      stride_dim);
}

// Preparing allocation info for indexing. Because of broadcasting,
// just looking at the loop groups of a tensor may not be enough to
// determine the allocation of the tensor. For example, this happens
// when a tensor is broadcast and inlined, where the original
// pre-broadcast tensor may not have corresponding domains. If that
// missing domain is annotated with ParallelType::Unroll, which
// affects all inner loops, just looking at the inlined tensor itself
// would miss the unrolling. Since unrolling changes allocation
// shapes, missing unroll can result in incorrect allocations.
//
// TODO: Refactor this and the allocation lowering pass
class AllocationDomainSetup : private kir::IrVisitor {
 public:
  using IrVisitor::dispatch;

  // Set allocation domain info for all tensors
  void setup(const std::vector<Expr*>& exprs) {
    // Find out correct allocation domains for all consumer
    // tensors. Input tensors are handled after this
    for (auto expr : exprs) {
      dispatch(expr);
    }

    // Make sure all tensors have allocation domains
    for (TensorView* producer_tv : used_as_producer) {
      auto it = tv_alloc_info_map.find(producer_tv);
      if (it != tv_alloc_info_map.end()) {
        continue;
      }

      // Not yet set. This must be an input tensor.
      NVF_ERROR(
          producer_tv->isFusionInput(),
          "Expected a fusion input: ",
          producer_tv->toString());

      // For fusion input, we can just use getMaybeAllocationDomain.

      auto alloc_info = getIndexingAllocationInfo(
          producer_tv,
          producer_tv->getMaybeAllocationDomain(),
          producer_tv->domain()->contiguity());

      tv_alloc_info_map.emplace(producer_tv, alloc_info);
    }
  }

  void dispatch(Expr* expr) override {
    if (ir_utils::isTvOp(expr)) {
      for (auto out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        // Note that since we are dealing with a Kernel IR, a single
        // tensor may show up as consumers multiple times, e.g.,
        // zero initialization and actual definition. Using the last
        // expr should give us correct allocation info. See
        // IndexingTest.InlinedUnroll for a concrete
        // example. Specifically, the initization expression of t2
        // doesn't have an unrolling loop, so the allocation info
        // obtained from that expression would fail to give the
        // correct allocation domains.
        auto [alloc_domains, contiguity] =
            getAllocationDomainsAndContiguity(out_tv, for_loops_);
        auto alloc_info =
            getIndexingAllocationInfo(out_tv, alloc_domains, contiguity);
        tv_alloc_info_map[out_tv] = alloc_info;
      }
      for (auto in_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
        used_as_producer.insert(in_tv);
      }
    } else {
      IrVisitor::dispatch(expr);
    }
  }

  // Get the allocation domains and contiguity of a given tensor
  //
  // TODO: Ideally, all tensors should have their correct allocation
  // domains, but that isn't always the case at this moment. The logic
  // here is duplicated in multiple locations and should be cleaned up.
  std::pair<std::vector<IterDomain*>, std::vector<std::optional<bool>>>
  getAllocationDomainsAndContiguity(
      TensorView* tv,
      const std::vector<ForLoop*>& for_loops) {
    std::vector<IterDomain*> allocation_domains;
    std::vector<std::optional<bool>> contiguity;

    // In general, if the tensor has an allocation domain set, it
    // should be used with no change. However, set allocation domains
    // are not always right allocation domains. For example,
    // AliasTest.NotAllOutputAlias_Reduction has a tensor, tv6, that
    // is a Local tensor with CA position of 4 but has an allocation
    // domain that's just a permutation of its logical domain. Such
    // invalid allocations need to be ignored. If there doesn't seem
    // to be any clear condition when the set domain can be used, so
    // it needs to be inferred. Here's what seems to be working
    // reasonably well.
    bool use_set_allocation_domain = false;
    if (tv->hasAllocation()) {
      // Honor the allocation domain if the tensor is global or Hopper MMA's
      // output
      if (tv->getMemoryType() == MemoryType::Global ||
          (tv->definition()->isA<MmaOp>() &&
           isHopper(tv->definition()->as<MmaOp>()->macro()))) {
        use_set_allocation_domain = true;
      } else if (tv->getMemoryType() == MemoryType::Shared) {
        // If it's a shared memory tensor, the set domain is likely
        // valid if Swizzle or Bulk is used. Also, if the allocation
        // domain is just a permutation of the loop domain, use the
        // set allocation domain. This seems to happen only with
        // AllocationDomainTest.TransposedIntermediate.
        if (std::any_of(
                tv->getAllocationDomain().begin(),
                tv->getAllocationDomain().end(),
                [](IterDomain* allocation_domain) {
                  return dynamic_cast<Swizzle*>(
                             allocation_domain->definition()) != nullptr ||
                      allocation_domain->getParallelType() ==
                      ParallelType::Bulk;
                }) ||
            std::is_permutation(
                tv->getLoopDomain().begin(),
                tv->getLoopDomain().end(),
                tv->getAllocationDomain().begin())) {
          use_set_allocation_domain = true;
        }
      }
    }

    if (use_set_allocation_domain) {
      if (tv->getMemoryType() == MemoryType::Global) {
        // For global memory tensors we always allocate the entire tensor
        // TODO: do we really want to treat global memory tensors differently?
        // need to think about this more.
        allocation_domains = tv->getAllocationDomain();
        contiguity = tv->domain()->contiguity();
      } else {
        std::unordered_set<IterDomain*> exclude_ca_ids;
        for (auto i : c10::irange(tv->getComputeAtPosition())) {
          auto ca_id = tv->axis(i);
          if (!ir_utils::isMemorySharedAcross(
                  tv->getMemoryType(), ca_id->getParallelType())) {
            exclude_ca_ids.insert(ca_id);
          }
        }
        for (auto i : c10::irange(tv->getAllocationDomain().size())) {
          auto id = tv->getAllocationDomain()[i];
          if (exclude_ca_ids.find(id) == exclude_ca_ids.end()) {
            if (ir_utils::isMemoryPartitionedAcross(
                    tv->getMemoryType(), id->getParallelType())) {
              continue;
            }
            allocation_domains.push_back(id);
            contiguity.push_back(tv->domain()->contiguity()[i]);
          } else {
            exclude_ca_ids.erase(id);
          }
        }
        NVF_ERROR(
            exclude_ca_ids.empty(),
            "The non-allocating compute-at IDs are not found in the allocation domain. ",
            "It is unclear how to allocate the tensor: ",
            tv->toString(),
            " allocation domain: ",
            ir_utils::toString(tv->getAllocationDomain()));
      }
    } else {
      // If allocation domain is not set, assume that:
      // - Global: logical domains
      // - Local/Shared: loop domains to the right of the CA position
      if (tv->getMemoryType() == MemoryType::Global) {
        allocation_domains = tv->getLogicalDomain();
        contiguity = tv->domain()->contiguity();
      } else {
        // Allocation position is not always the same as the CA
        // position. See also lower_utils::getAllocInformation.
        int64_t allocation_pos =
            lower_utils::getAllocInformation(tv, for_loops).alloc_pos;
        for (const auto i : c10::irange(tv->nDims())) {
          auto loop_id = tv->getLoopDomain().at(i);
          auto pt = loop_id->getParallelType();
          if (!mayRequireAllocation(tv, loop_id)) {
            continue;
          }

          // If the position is left of the inlining position, no need to
          // allocate the domain unless it's shared. For example, if this
          // is a Shared tensor and the domain is parallelized with TID,
          // even if it's outside of the CA position, since the domain
          // is shared, it must be allocated.
          if (i < allocation_pos &&
              !ir_utils::isMemorySharedAcross(tv->getMemoryType(), pt)) {
            continue;
          }

          allocation_domains.push_back(loop_id);
        }
        // Assume Local and Shared are always fully contiguous
        contiguity =
            std::vector<std::optional<bool>>(allocation_domains.size(), true);
      }

      if (auto reordered_domains =
              reorderAllocationDomains(tv, allocation_domains);
          reordered_domains.has_value()) {
        allocation_domains = reordered_domains.value();
        NVF_ERROR(
            std::all_of(
                contiguity.begin(),
                contiguity.end(),
                [](auto b) { return b.has_value() && b.value(); }),
            tv->toString());
      }

      // WAR for transpose
      if (auto transposed_smem_alloc_dom =
              patchAllocationOfTransposedSmemTensor(
                  tv,
                  allocation_domains,
                  GpuLower::current()->idModel().idGraph(IdMappingMode::EXACT));
          transposed_smem_alloc_dom.has_value()) {
        allocation_domains = transposed_smem_alloc_dom.value();
        // Make sure the original allocation domains are fully contiguous
        NVF_ERROR(std::all_of(contiguity.begin(), contiguity.end(), [](auto b) {
          return b.has_value() && b.value();
        }));
        // Set the new allocation domains fully contiguous
        contiguity =
            std::vector<std::optional<bool>>(allocation_domains.size(), true);
      }
    }

    NVF_ERROR(allocation_domains.size() == contiguity.size());

    return {allocation_domains, contiguity};
  }

  // Get allocation info used for indexing. Loop promotion is
  // considered. Strides are also calculated.
  IndexingAllocationInfo getIndexingAllocationInfo(
      TensorView* tv,
      std::vector<IterDomain*> allocation_domains,
      std::vector<std::optional<bool>> contiguity) {
    const IdModel& id_model = GpuLower::current()->idModel();

    std::vector<IterDomain*> promoted_allocation_domains;
    promoted_allocation_domains.reserve(allocation_domains.size());

    // Loop promotion may affect allocations. Promotions of intermediate
    // domains may not be defined correctly. Only consider loop domains
    // for now.
    for (const auto& allocation_domain : allocation_domains) {
      bool is_loop = !tv->isFusionInput() &&
          std::find(
              tv->getLoopDomain().begin(),
              tv->getLoopDomain().end(),
              allocation_domain) != tv->getLoopDomain().end();
      IterDomain* promotion_domain = nullptr;
      if (is_loop) {
        promotion_domain = getLoopPromotion(allocation_domain, id_model);
      } else {
        promotion_domain = allocation_domain;
      }
      promoted_allocation_domains.push_back(promotion_domain);
    }

    // Compute the strides from innermost to outermost domains
    std::vector<Val*> strides(allocation_domains.size(), nullptr);
    Val* cur_contig_stride = tv->fusion()->oneVal();
    for (const auto i : c10::irange(allocation_domains.size())) {
      auto dim = allocation_domains.size() - i - 1;
      auto allocation_domain = allocation_domains.at(dim);
      auto promotion_domain = promoted_allocation_domains.at(dim);

      if (!mayRequireAllocation(tv, allocation_domain)) {
        continue;
      }

      const std::optional<bool> contig_flag = contiguity.at(dim);
      // Broadcast doesn't have contig flag but it must have been
      // already filtered out
      NVF_ERROR(contig_flag.has_value());

      if (contig_flag.value()) {
        strides[dim] = cur_contig_stride;
        cur_contig_stride = SimplifyingIrBuilder::mulExpr(
            cur_contig_stride, promotion_domain->extent());
      } else {
        // Assume that the tensor should always be a Global memory
        // tensor if it has non-contig allocation domains
        NVF_ERROR(tv->getMemoryType() == MemoryType::Global);
        strides[dim] = getStrideOfGlobalMemoryTensor(tv, (int64_t)dim);
        cur_contig_stride = SimplifyingIrBuilder::mulExpr(
            strides[dim], promotion_domain->extent());
      }
    }

    // Filter out non-allocated domains. This is already done for Local
    // and Shared tensors with no set allocation domains, but not for
    // the other cases. For example, a reduction output tensor that is
    // also a fusion output may still have reduction domains in their
    // allocation domains, which aren't relevant for indexing
    std::vector<IterDomain*> actual_allocation_domains;
    std::vector<Val*> actual_strides;
    std::vector<bool> actual_contiguity;
    for (const auto i : c10::irange(allocation_domains.size())) {
      auto allocation_domain = allocation_domains.at(i);
      auto promotion_domain = promoted_allocation_domains.at(i);
      if (!mayRequireAllocation(tv, allocation_domain)) {
        continue;
      }
      auto stride = strides.at(i);
      NVF_ERROR(stride != nullptr);
      actual_allocation_domains.push_back(promotion_domain);
      actual_strides.push_back(stride);
      auto contig = contiguity.at(i);
      NVF_ERROR(contig.has_value());
      actual_contiguity.push_back(contig.value());
    }

    NVF_ERROR(actual_allocation_domains.size() == actual_strides.size());
    NVF_ERROR(actual_allocation_domains.size() == actual_contiguity.size());

    return IndexingAllocationInfo{
        actual_allocation_domains, actual_strides, actual_contiguity};
  }

  // Reorder non-logical allocation domains to follow the ordering of
  // the set allocation domain. This is necessary when an allocation
  // domain includes a vectorized loop iter domain since it must be at the
  // innermost position but that may not be the case in the loop
  // domain. It is also necessary when the tensor is a producer of a
  // vectorized store. Not strictly necessary otherwise, but this should also
  // minimize the deviation from the old indexing scheme which always
  // uses the logical domain to index.
  //
  // Returns reordered allocation domains if reordering is done.
  std::optional<std::vector<IterDomain*>> reorderAllocationDomains(
      const TensorView* tv,
      const std::vector<IterDomain*>& allocation_domains) const {
    // Use getMaybeAllocationDomain instead of getLogicalDomain. When
    // this tv is a producer of a vectorized store, the consumer
    // tensor shoud be a global memory tensor and this is likely a
    // cache tensor created by cacheBefore. The consumer tensor may
    // have a reordered allocation domain and that dictates the actual
    // allocation ordering of this producer local tensor as well. If
    // getLogicalDomain is used, DistributedTransformerTest.Backward
    // fails at the result validation.
    auto exprs = DependencyCheck::getAllExprsBetween(
        {tv->getMaybeAllocationDomain().begin(),
         tv->getMaybeAllocationDomain().end()},
        {allocation_domains.begin(), allocation_domains.end()});

    if (exprs.empty()) {
      return std::nullopt;
    }

    // Replay exprs from the logical domain to get the non-reordered
    // domains
    auto ordered_domains = tv->getMaybeAllocationDomain();
    for (auto expr : exprs) {
      // Find the position to insert the outputs.
      int64_t insertion_pos = -1;
      for (auto inp : expr->inputs()) {
        auto it =
            std::find(ordered_domains.begin(), ordered_domains.end(), inp);
        if (it == ordered_domains.end()) {
          continue;
        }
        // Insert right after the input
        int64_t pos = std::distance(ordered_domains.begin(), it) + 1;
        if (insertion_pos == -1 || pos > insertion_pos) {
          insertion_pos = pos;
        }
      }
      NVF_ERROR(
          insertion_pos >= 0,
          "Failed to replay: ",
          expr->toString(),
          " in ",
          tv->toString());
      // Insert the outputs
      for (auto out : expr->outputs()) {
        ordered_domains.insert(
            ordered_domains.begin() + insertion_pos, out->as<IterDomain>());
        ++insertion_pos;
      }
      // Delete the inputs
      for (auto inp : expr->inputs()) {
        auto it =
            std::find(ordered_domains.begin(), ordered_domains.end(), inp);
        if (it == ordered_domains.end()) {
          continue;
        }
        ordered_domains.erase(it);
      }
    }

    // At this point, all domains of allocation_domains must exist in
    // domains.
    for (auto alloc_dom : allocation_domains) {
      auto it =
          std::find(ordered_domains.begin(), ordered_domains.end(), alloc_dom);
      NVF_ERROR(
          it != ordered_domains.end(),
          "Missing allocation domain: ",
          alloc_dom->toString(),
          ", domains: ",
          toDelimitedString(ordered_domains));
    }

    // Pick only the allocation domains from the ordered domains
    std::vector<IterDomain*> reordered_allocation_domains;
    reordered_allocation_domains.reserve(allocation_domains.size());

    for (auto dom : ordered_domains) {
      auto it =
          std::find(allocation_domains.begin(), allocation_domains.end(), dom);
      if (it == allocation_domains.end()) {
        continue;
      }
      reordered_allocation_domains.push_back(dom);
    }

    // If it's the same order, just return nullopt to tell nothing
    // needs to be reordered
    if (reordered_allocation_domains == allocation_domains) {
      return std::nullopt;
    }

    return reordered_allocation_domains;
  }

  // Transpose with shared memory may need to change the ordering of
  // allocation domains when shared memory is used as an input to
  // vectorized stores. The transpose scheduler stages data to shared
  // memory for vectorized stores to global memory. The layout of the
  // shared memory staging buffer needs to be compatible with the
  // vectorized stores. More specifically, here's a typical pattern of
  // the transpose scheduler:
  //
  // t0_g: [I0, I1]
  // t1_l = transpose(0, 1); // [I1, I0]
  // t2_s = t1_l; // [I1, I0]
  // t3_g = t2_s; // [I1, I0]
  //
  // t0, t1, t2:
  //   split I0 by 32 -> I/32a, 32a
  //   split I1 by 32 -> I/32b, 32b
  //   merge 32a and 32b -> 32a*32b
  //   split 32a*32b by 4 -> 32a*32b/4, 4
  //  -> loop domain: [I0/32a, I1/32b, 32a*32b/4, 4]
  // t3:
  //   split I0 by 32 -> I/32a, 32a
  //   split I1 by 32 -> I/32b, 32b
  //   merge 32b and 32a -> 32b*32a
  //   split 32*32 by 4 -> 32b*32a/4, 4
  //  -> loop domain: [I0/32a, I1/32b, 32b*32a/4, 4]
  //
  // Notice that t2 has 32a*32b, whereas t3 has 32b*32a. When the innermost
  // domain of t3 is vectorized, this means that 32a must be the
  // innermost in the allocation domain of t2. However, the inferred
  // allocation domain has [..., 32a*32b/4, 4], so 32a is not the
  // innermost.
  //
  // When a given tensor is found to have this pattern, allocation
  // domains as ordered in the same way as the vectorized global
  // memory tensor are returned. In the case of the above example,
  // [32b, 32a] is returned.
  std::optional<std::vector<IterDomain*>> patchAllocationOfTransposedSmemTensor(
      const TensorView* tv,
      const std::vector<IterDomain*>& allocation_domains,
      const ValGraph& exact_graph) const {
    // First, do pattern matching to see if this tensor is a shared
    // memory tensor transpose. Pattern matching conditions include:
    //
    // - Shared memory tensor
    // - BID/DID should not be used with allocation domains
    // - Consumer tensor must be a global memory tensor with vectorization
    // - There must be a merge op whose two outputs are the dominating
    //   domains of the allocation domains
    // - The consumer tensor also has a merge but with the inner and
    //   outer reversed

    if (allocation_domains.empty()) {
      return std::nullopt;
    }

    if (tv->getMemoryType() != MemoryType::Shared) {
      return std::nullopt;
    }

    // No BID/DID parallel type should be used
    if (std::any_of(
            allocation_domains.begin(),
            allocation_domains.end(),
            [](IterDomain* id) -> bool {
              return isParallelTypeDeviceDim(id->getParallelType()) ||
                  isParallelTypeBlockDim(id->getParallelType());
            })) {
      return std::nullopt;
    }

    // Can there be multiple stores with a single smem buffer?
    if (tv->uses().size() != 1) {
      return std::nullopt;
    }

    auto ls_op = dynamic_cast<LoadStoreOp*>(tv->uses().front());
    if (ls_op == nullptr) {
      return std::nullopt;
    }

    auto consumer = ls_op->out()->as<TensorView>();

    if (consumer->getMemoryType() != MemoryType::Global) {
      return std::nullopt;
    }

    IterDomain* consumer_vectorized_domain = nullptr;
    if (auto it = std::find_if(
            consumer->getLoopDomain().begin(),
            consumer->getLoopDomain().end(),
            [](IterDomain* loop_id) {
              return loop_id->getParallelType() == ParallelType::Vectorize;
            });
        it != consumer->getLoopDomain().end()) {
      consumer_vectorized_domain = *it;
    } else {
      return std::nullopt;
    }

    // May be naive, but assume a simple pattern that all allocation
    // domains are derived from a merge.

    // First, find the closest merge
    auto getOriginatingMerge = [](IterDomain* id) -> Merge* {
      while (id != nullptr) {
        auto def = id->definition();
        if (auto merge = dynamic_cast<Merge*>(def)) {
          return merge;
        } else if (auto split = dynamic_cast<Split*>(def)) {
          id = split->in();
        } else {
          // Unsupported op
          return nullptr;
        }
      }
      return nullptr;
    };

    Merge* producer_common_merge =
        getOriginatingMerge(allocation_domains.front());
    if (producer_common_merge == nullptr) {
      return std::nullopt;
    }

    // Test if all allocation domains and the merge output are
    // equivalent
    auto producer_merge_dep_exprs = DependencyCheck::getAllExprsBetween(
        {producer_common_merge->out()},
        {allocation_domains.begin(), allocation_domains.end()});

    std::unordered_set<IterDomain*> equiv_domain_set(
        allocation_domains.begin(), allocation_domains.end());

    // Traverse back from the allocation domains to the merge output
    // and see if they are equivalent
    for (auto it = producer_merge_dep_exprs.rbegin();
         it != producer_merge_dep_exprs.rend();
         ++it) {
      Expr* expr = *it;
      for (auto out : expr->outputs()) {
        auto it = equiv_domain_set.find(out->as<IterDomain>());
        if (it == equiv_domain_set.end() &&
            mayRequireAllocation(tv, out->as<IterDomain>())) {
          // missing dependency
          return std::nullopt;
        }
        if (it != equiv_domain_set.end()) {
          equiv_domain_set.erase(it);
        }
      }
      for (auto input : expr->inputs()) {
        equiv_domain_set.insert(input->as<IterDomain>());
      }
    }

    // If they are equivalent, the merge output should be the only
    // remaining domain
    if (!(equiv_domain_set.size() == 1 &&
          *(equiv_domain_set.begin()) == producer_common_merge->out())) {
      // Not all allocation domains are used, meaning the merge output
      // is not equivalent to the allocation domains
      return std::nullopt;
    }

    // Look for a reverse merge in the consumer that uses the same
    // inputs but outer and inner are reversed

    IterDomain* merge_outer = producer_common_merge->outer();
    const ValGroup& merge_outer_group = exact_graph.toGroup(merge_outer);
    IterDomain* merge_inner = producer_common_merge->inner();
    const ValGroup& merge_inner_group = exact_graph.toGroup(merge_inner);

    const ExprGroups& merge_outer_uses = exact_graph.getUses(merge_outer_group);
    ExprGroup reverse_merge;
    for (const auto& merge_outer_use : merge_outer_uses) {
      Merge* merge = dynamic_cast<Merge*>(merge_outer_use->front());
      if (merge == nullptr) {
        continue;
      }
      if (exact_graph.toGroup(merge->outer()) == merge_inner_group &&
          exact_graph.toGroup(merge->inner()) == merge_outer_group) {
        reverse_merge = merge_outer_use;
        break;
      }
    }

    if (reverse_merge.get() == nullptr) {
      return std::nullopt;
    }

    ValGroup reverse_merge_output =
        exact_graph.outputGroups(reverse_merge).at(0);
    // Look for a matching merge in the consumer
    const auto consumer_all_ids = consumer->domain()->allIDs();
    IterDomain* consumer_merge_out = nullptr;
    for (auto consumer_id : consumer_all_ids) {
      if (reverse_merge_output->has(consumer_id)) {
        consumer_merge_out = consumer_id;
        break;
      }
    }

    if (consumer_merge_out == nullptr) {
      return std::nullopt;
    }

    // If there's a loop id that depends on consumer_merge_output, the
    // producer tensor needs to use the memory layout that works for
    // the vectorized store of the consumer tensor.
    if (!DependencyCheck::isDependencyOf(
            consumer_merge_out, consumer_vectorized_domain)) {
      return std::nullopt;
    }

    std::vector<IterDomain*> patched_allocation_domains{
        merge_inner, merge_outer};

    return patched_allocation_domains;
  }

  std::unordered_map<TensorView*, IndexingAllocationInfo> tv_alloc_info_map;
  std::unordered_set<TensorView*> used_as_producer;
};

} // namespace

TensorIndexer::TensorIndexer(IdModel& id_model) : id_model_(id_model) {
  buildLoopIndexMap();

  if (isDebugDumpEnabled(DebugDumpOption::IndexingVerbose)) {
    std::ofstream ofs("indexing_traversal_graph.dot", std::ofstream::trunc);
    auto dot_string = traversalGraph().toGraphvizDotGraph();
    ofs << dot_string;
    ofs.close();
  }
}

void TensorIndexer::buildLoopIndexMap() {
  if (id_model_.empty()) {
    return;
  }

  Fusion* fusion = id_model_.fusion();

  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }
    // It's assumed that all sibling outputs share the same for-loops,
    // thus only one of the outputs is considered.
    auto tv_output = ir_utils::getTvOutput(expr);
    for (auto loop_id : tv_output->getLoopDomain()) {
      const ValGroup& loop_group =
          id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_id);

      if (loop_index_map_.find(loop_group) != loop_index_map_.end()) {
        // Index already assigned
        continue;
      }

      Val* loop_index = nullptr;

      if (shouldUseZeroIndex(loop_group, id_model_)) {
        loop_index = fusion->zeroVal();
      } else {
        loop_index = GpuLower::current()->getLoopIndexVariable(loop_id);
      }

      loop_index_map_[loop_group] = loop_index;
    }
  }
}

Val* TensorIndexer::getLoopIndex(
    IterDomain* loop_id,
    const std::vector<ForLoop*>& for_loops) const {
  // loop_id must be a loop domain.
  const auto& loop_group =
      id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_id);
  auto loop_index_map_it = loop_index_map_.find(loop_group);
  NVF_ERROR(
      loop_index_map_it != loop_index_map_.end(),
      "No loop index found for ",
      loop_id->toString());

  Val* loop_index = loop_index_map_it->second;

  // War for circular buffering
  if (auto circular_buffer_loop_index =
          getLoopIndexOfCircularBufferLoop(loop_id, for_loops, id_model_)) {
    loop_index = circular_buffer_loop_index;
  }

  return loop_index;
}

std::unordered_map<ValGroup, Val*> TensorIndexer::getInitialIndexMap(
    const std::vector<IterDomain*>& loop_domains,
    const std::vector<ForLoop*>& for_loops) const {
  std::unordered_map<ValGroup, Val*> initial_index_map;

  // For a given list of the loop domains, assign its corresponding
  // index Val.
  for (IterDomain* loop_id : loop_domains) {
    Val* initial_index = getLoopIndex(loop_id, for_loops);
    const auto& almost_exact_group = traversalGraph().toGroup(loop_id);

    if (initial_index_map.find(almost_exact_group) != initial_index_map.end()) {
      // Initial index already set. This can happen as this is an
      // almost exact group. It should be just size-1 domain.
      NVF_ERROR(
          initial_index->isZeroInt(),
          "Unexpected initial index: ",
          initial_index->toInlineString());
      auto existing_index = initial_index_map.at(almost_exact_group);
      NVF_ERROR(
          existing_index->isZeroInt(),
          "Unexpected initial index: ",
          existing_index->toInlineString());
      continue;
    }

    initial_index_map.emplace(almost_exact_group, initial_index);
  }

  return initial_index_map;
}

std::vector<Val*> TensorIndexer::getIndexFor(
    const Expr* expr,
    bool as_consumer,
    const std::vector<IterDomain*>& index_ids,
    const std::vector<ForLoop*>& for_loops) const {
  auto info = computeIndex(expr, index_ids, for_loops);
  const auto& replacement_map = getIndexReplacementMap(
      expr, as_consumer, info.loop_domains, for_loops, info.index_map);

  // Note that IDs of index_ids may be mapped as the traversal graph
  // is the AlmostExact graph.

  std::vector<Val*> result;
  result.reserve(index_ids.size());
  for (IterDomain* index_id : index_ids) {
    const auto& index_group = traversalGraph().toGroup(index_id);
    auto it = info.index_map.find(index_group);
    NVF_ERROR(
        it != info.index_map.end(),
        "Index not found for ",
        index_id->toString());
    result.push_back(
        ir_utils::replaceValRecursively(it->second, replacement_map));
  }
  return result;
}

Val* TensorIndexer::getLinearIndex(
    TensorView* tv,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops) const {
  NVF_ERROR(tv != nullptr);
  NVF_ERROR(expr != nullptr);
  NVF_ERROR(
      (std::find(expr->inputs().begin(), expr->inputs().end(), tv) !=
       expr->inputs().end()) ||
          (std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
           expr->outputs().end()),
      "Inconsistent tensor and expr. Tensor, ",
      tv->toString(),
      " not found in ",
      expr->toString());

  const bool as_consumer =
      std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
      expr->outputs().end();

  const auto alloc_info = getIndexingAllocationInfo(tv);

  const auto [contig_indices, contig_strides] =
      getContigIndexFor(expr, as_consumer, alloc_info, for_loops);

  // Linearize the indices with strides.
  Val* linear_index = tv->fusion()->zeroVal();
  for (const auto i : c10::irange(contig_indices.size())) {
    Val* stride = contig_strides.at(i);
    linear_index = SimplifyingIrBuilder::addExpr(
        linear_index,
        SimplifyingIrBuilder::mulExpr(contig_indices.at(i), stride));
  }

  // If a tensor is circular buffered, it also requires indexing of
  // the circular buffer itself
  if (tv->isCircularBuffered()) {
    auto circular_buffer_offset =
        getOffsetForCircularBufferTensor(tv, as_consumer, for_loops);
    linear_index =
        SimplifyingIrBuilder::addExpr(linear_index, circular_buffer_offset);
  }

  return linear_index;
}

// Get the loop domains of a given expr, which are (potentially
// promoted) loop domains of the consumer tensor.
std::vector<IterDomain*> TensorIndexer::getLoopDomains(const Expr* expr) const {
  // Assume consumer-based indexing. Needs to revisit for ops like
  // scatter
  auto loop_domains = ir_utils::getTvOutput(expr)->getLoopDomain();

  for (auto& loop_id : loop_domains) {
    loop_id = getLoopPromotion(loop_id, id_model_);
  }

  return loop_domains;
}

IndexingInfo TensorIndexer::computeIndex(
    const Expr* expr,
    const std::vector<IterDomain*>& index_ids,
    const std::vector<ForLoop*>& for_loops) const {
  const auto loop_domains = getLoopIds(expr, id_model_);

  // Set aside broadcast IDs as their indices should always be zero
  // and they may not be reachable from the loop domain
  std::vector<IterDomain*> broadcast_index_ids;
  std::vector<IterDomain*> non_broadcast_index_ids;
  for (const auto index_id : index_ids) {
    if (index_id->isBroadcast()) {
      broadcast_index_ids.push_back(index_id);
    } else {
      non_broadcast_index_ids.push_back(index_id);
    }
  }

  const ValGroups loop_groups = traversalGraph().toGroups(loop_domains);
  const ExprPath<ExprGroup> traversal_path = IndexingTraversal::getExprsBetween(
      expr, traversalGraph(), loop_domains, non_broadcast_index_ids);

  const std::unordered_map<ValGroup, Val*> initial_index_map =
      getInitialIndexMap(loop_domains, for_loops);

  IdGraphIndexCompute index_compute(traversalGraph(), initial_index_map);

  // In addition to indices themselves, keep track of the
  // dependency from each domain to loop domains. This dependency is
  // represented as a map from ValGroup of the traversal graph to
  // ValGroup of the LOOP graph.
  std::unordered_map<ValGroup, ValGroups> loop_group_dependencies;

  // Initialize the loop dependency mappings
  for (const auto& loop_domain : loop_domains) {
    const auto& traversal_graph_group = traversalGraph().toGroup(loop_domain);
    const auto& loop_graph_group =
        id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_domain);
    loop_group_dependencies[traversal_graph_group].pushBack(loop_graph_group);
  }

  for (const auto& [expr_group, direction] : traversal_path) {
    index_compute.propagate(expr_group, direction);

    // Propagate loop dependencies from inputs to outputs
    const auto input_groups = direction == Direction::Forward
        ? traversalGraph().inputGroups(expr_group)
        : traversalGraph().outputGroups(expr_group);
    const auto output_groups = direction == Direction::Forward
        ? traversalGraph().outputGroups(expr_group)
        : traversalGraph().inputGroups(expr_group);
    for (const auto& output : output_groups) {
      for (const auto& input : input_groups) {
        const auto& input_loop_groups = loop_group_dependencies.at(input);
        loop_group_dependencies[output].pushBack(input_loop_groups);
      }
    }
  }

  // Fill in broadcast index groups by zero
  auto index_map = index_compute.indexMap();
  for (const auto broadcast_index_id : broadcast_index_ids) {
    index_map[traversalGraph().toGroup(broadcast_index_id)] =
        broadcast_index_id->fusion()->zeroVal();
  }

  IndexingInfo info{
      loop_domains, traversal_path, index_map, loop_group_dependencies};
  return info;
}

std::unordered_map<Val*, Val*> TensorIndexer::getIndexReplacementMap(
    const Expr* expr,
    bool as_consumer,
    const std::vector<IterDomain*>& loop_domains,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<ValGroup, Val*>& index_map) const {
  std::unordered_map<Val*, Val*> replacement_map;

  for (const auto loop_id : loop_domains) {
    Val* cur_index = getLoopIndex(loop_id, for_loops);

    Val* replacement_index = nullptr;
    // Replace the index of a vectorized/bulk domain with zero. Note that
    // vectorized domains may need to use N-1, where N is the extent
    // of the domain, for predication, so the replacement is not
    // always done with zero.
    if (loop_id->getParallelType() == ParallelType::Vectorize ||
        loop_id->getParallelType() == ParallelType::Bulk ||
        loop_id->getParallelType() == ParallelType::Mma) {
      replacement_index = loop_id->fusion()->zeroVal();
    } else {
      ForLoop* for_loop = indexing_utils::getForLoop(
          loop_id, for_loops, id_model_.idGraph(IdMappingMode::LOOP));

      // for_loop is nullptr if no matching loop is found, which
      // happens when loop_id is a reduction domain and this loop-nest
      // is for initializing the reduction buffer.
      if (for_loop != nullptr) {
        // Replace circular buffer index with zero value if for-loop is trivial
        if (for_loop->circularBufferLoopStage() !=
            CircularBufferLoopStage::NotApplicable) {
          Val* base_index =
              replacement_index != nullptr ? replacement_index : cur_index;
          replacement_index =
              for_loop->isTrivial() ? for_loop->start() : base_index;
        }

        // If this for-loop is a circular buffer loop, the loop index
        // may need to have an additional offset.
        if (!as_consumer) {
          if (auto circular_buffer_offset =
                  getLoopIndexOffsetForProducerOfCircularBuffer(
                      expr, for_loop, id_model_)) {
            replacement_index = SimplifyingIrBuilder::addExpr(
                replacement_index, circular_buffer_offset);
          }
        }
      }
    }

    if (replacement_index == nullptr || replacement_index == cur_index) {
      continue;
    }

    replacement_map.emplace(cur_index, replacement_index);
  }

  return replacement_map;
}

void TensorIndexer::setupAllocationDomains(const std::vector<Expr*>& exprs) {
  AllocationDomainSetup alloc_setup;
  alloc_setup.setup(exprs);
  alloc_info_ = std::move(alloc_setup.tv_alloc_info_map);
}

std::vector<PredicateInfo> TensorIndexer::getPredicates(
    TensorView* tv,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops,
    ForLoop* unswitched_loop) const {
  const auto& zero_val = tv->fusion()->zeroVal();

  const std::vector<IterDomain*>& predicate_domains =
      getPredicateDomains(tv, expr);

  const IndexingInfo& index_info =
      computeIndex(expr, predicate_domains, for_loops);

  const auto& index_map = index_info.index_map;

  const std::unordered_map<Val*, Val*> replacement_map_start =
      getPredicateIndexReplacementMap(
          tv,
          for_loops,
          index_map,
          traversalGraph(),
          index_info.traversal_path,
          id_model_,
          /*is_start_predicate=*/true,
          /*unswitched_loop=*/unswitched_loop);

  const std::unordered_map<Val*, Val*> replacement_map_stop =
      getPredicateIndexReplacementMap(
          tv,
          for_loops,
          index_map,
          traversalGraph(),
          index_info.traversal_path,
          id_model_,
          /*is_start_predicate=*/false,
          /*unswitched_loop=*/unswitched_loop);

  const std::unordered_map<IterDomain*, ValGroup> contig_domains =
      isContigIndexingEnabled()
      ? getContigDomains(
            predicate_domains,
            std::vector<bool>(predicate_domains.size(), true),
            reverse(index_info.traversal_path),
            traversalGraph(),
            /*is_predicate_pass=*/true)
      : std::unordered_map<IterDomain*, ValGroup>{};

  auto getCoveredPredicatedDomains =
      [&predicate_domains, &contig_domains](const ValGroup& contig_group) {
        std::unordered_set<IterDomain*> covered_domains;
        for (const auto& predicate_domain : predicate_domains) {
          auto contig_domains_it = contig_domains.find(predicate_domain);
          NVF_ERROR(contig_domains_it != contig_domains.end());
          if (contig_group == contig_domains_it->second) {
            covered_domains.emplace(predicate_domain);
          }
        }
        return covered_domains;
      };

  const CircularBufferLoopStage loop_stage = getCircularBufferLoopStage(
      tv, for_loops, id_model_.idGraph(IdMappingMode::LOOP));

  std::vector<PredicateInfo> info_vec;
  info_vec.reserve(predicate_domains.size());

  std::unordered_set<ValGroup> already_indexed_domains;

  // Follow the same approach as Index::getReferenceRootPredicates.
  for (const auto& predicate_domain : predicate_domains) {
    const auto& predicate_domain_group =
        traversalGraph().toGroup(predicate_domain);
    IterDomain* actual_predicate_domain = predicate_domain;
    ValGroup actual_predicate_domain_group = predicate_domain_group;
    std::unordered_set<IterDomain*> actual_predicate_domains = {
        predicate_domain};

    if (isContigIndexingEnabled()) {
      auto contig_domains_it = contig_domains.find(predicate_domain);
      NVF_ERROR(
          contig_domains_it != contig_domains.end(),
          "No contig domain mapping found for ",
          predicate_domain->toString());
      const ValGroup& contig_domain_group = contig_domains_it->second;
      if (already_indexed_domains.find(contig_domain_group) !=
          already_indexed_domains.end()) {
        continue;
      }
      already_indexed_domains.emplace(contig_domain_group);

      actual_predicate_domain_group = contig_domain_group;
      actual_predicate_domain =
          actual_predicate_domain_group->front()->as<IterDomain>();
      actual_predicate_domains =
          getCoveredPredicatedDomains(contig_domain_group);
    }

    auto idx_it = index_map.find(actual_predicate_domain_group);
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        nvfuser::toString(actual_predicate_domain_group));

    Val* idx = idx_it->second;
    Val* start_idx =
        ir_utils::replaceValRecursively(idx, replacement_map_start);
    Val* stop_idx = ir_utils::replaceValRecursively(idx, replacement_map_stop);

    // Generate predicates as follows:
    //
    // (start_idx + start_offset) >= 0 &&
    // (stop_idx + stop_offset) < extent.

    PredicateInfo info;
    // For now, just set zero for both start and stop offsets by
    // assuming the domain is not partial.
    NVF_ERROR(!predicate_domain->maybePartial());
    info.start_offset_ = tv->fusion()->zeroVal();
    info.stop_offset_ = tv->fusion()->zeroVal();
    info.loop_stage_ = loop_stage;

    info.start_predicate_ = SimplifyingIrBuilder::geExpr(
        SimplifyingIrBuilder::addExpr(start_idx, info.start_offset_), zero_val);

    info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
        SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
        actual_predicate_domain->extent());

    info.predicated_domains_ = actual_predicate_domains;

    // Set the used loop ID groups for this predicated domain
    const ValGroups& loop_deps =
        index_info.loop_group_dependencies.at(actual_predicate_domain_group);
    for (const auto& loop_dep : loop_deps) {
      info.loop_domains_.insert(loop_dep->front()->as<IterDomain>());
    }

    info_vec.emplace_back(info);
  }

  // Add predicates for non-divisible splits.
  // If this is a reduction init expr, then no need to take care of
  // non divisible splits
  if (!lower_utils::isReductionInitExpr(expr)) {
    for (const auto& [eg, direction] : index_info.traversal_path) {
      // NOTE: Fundamentally, the problem of non divisiblity should be
      // checked while traversing the indexing path. Currently, it uses
      // the information gathered in a tensor-by-tensor basis. This
      // should be fine currently, but may not work if, e.g., the
      // indexing path involved both backward and forward traversals.
      if (!isNonDivisibleSplit(eg)) {
        continue;
      }

      NVF_ERROR(eg->front()->isA<Split>());
      auto split_to_predicate = eg->front()->as<Split>();

      IterDomain* non_divisible_domain = split_to_predicate->in();
      const auto& non_divisible_domain_group =
          traversalGraph().toGroup(non_divisible_domain);

      PredicateInfo info;
      info.loop_stage_ = loop_stage;
      // The start predicate should always be true
      info.start_offset_ = zero_val;
      info.start_predicate_ = non_divisible_domain->fusion()->trueVal();

      info.stop_offset_ = zero_val;

      auto idx_it = index_map.find(non_divisible_domain_group);
      NVF_ERROR(
          idx_it != index_map.end(),
          "Index not found for non-divisible split domain: ",
          non_divisible_domain->toString());

      auto idx =
          ir_utils::replaceValRecursively(idx_it->second, replacement_map_stop);
      info.stop_predicate_ =
          SimplifyingIrBuilder::ltExpr(idx, non_divisible_domain->extent());
      info.predicated_domains_ = {non_divisible_domain};

      const ValGroups& loop_deps =
          index_info.loop_group_dependencies.at(non_divisible_domain_group);
      for (const auto& loop_dep : loop_deps) {
        info.loop_domains_.insert(loop_dep->front()->as<IterDomain>());
      }

      info_vec.emplace_back(info);
    }
  }

  return info_vec;
}

std::pair<std::vector<ValGroup>, std::vector<Val*>> TensorIndexer::
    getContigDomainsAndStrides(
        const IndexingAllocationInfo& alloc_info,
        const ExprPath<ExprGroup>& traversal_path) const {
  const std::unordered_map<IterDomain*, ValGroup>& contig_domains =
      getContigDomains(
          alloc_info.domains,
          alloc_info.contiguity,
          reverse(traversal_path),
          traversalGraph(),
          /*is_predicate_pass=*/false);

  // Find contiguous domains to index
  std::unordered_set<ValGroup> already_indexed_domains;
  std::deque<ValGroup> contig_alloc_groups;
  std::deque<Val*> contig_strides;
  for (const auto i : c10::irange(alloc_info.domains.size())) {
    // Traverse back from the innermost domains so that the right
    // stride val is picked up for each contiguous domain
    auto i1 = alloc_info.domains.size() - 1 - i;
    IterDomain* allocation_domain = alloc_info.domains.at(i1);
    auto contig_domains_it = contig_domains.find(allocation_domain);
    NVF_ERROR(
        contig_domains_it != contig_domains.end(),
        "No contig domain mapping found for ",
        allocation_domain->toString());

    const ValGroup& contig_domain_group = contig_domains_it->second;
    if (already_indexed_domains.find(contig_domain_group) !=
        already_indexed_domains.end()) {
      continue;
    }
    already_indexed_domains.emplace(contig_domain_group);

    contig_alloc_groups.push_front(contig_domain_group);
    contig_strides.push_front(alloc_info.strides.at(i1));
  }

  return {
      {contig_alloc_groups.begin(), contig_alloc_groups.end()},
      {contig_strides.begin(), contig_strides.end()}};
}

std::pair<std::vector<Val*>, std::vector<Val*>> TensorIndexer::
    getContigIndexFor(
        const Expr* expr,
        bool as_consumer,
        const IndexingAllocationInfo& alloc_info,
        const std::vector<ForLoop*>& for_loops) const {
  auto index_info = computeIndex(expr, alloc_info.domains, for_loops);
  const auto& index_map = index_info.index_map;
  const auto& replacement_map = getIndexReplacementMap(
      expr, as_consumer, index_info.loop_domains, for_loops, index_map);

  std::vector<ValGroup> contig_alloc_groups;
  std::vector<Val*> contig_strides;

  if (isContigIndexingEnabled()) {
    const auto& contig_alloc_strides =
        getContigDomainsAndStrides(alloc_info, index_info.traversal_path);
    contig_alloc_groups = contig_alloc_strides.first;
    contig_strides = contig_alloc_strides.second;
  } else {
    std::transform(
        alloc_info.domains.begin(),
        alloc_info.domains.end(),
        std::back_inserter(contig_alloc_groups),
        [&](IterDomain* allocation_domain) {
          return traversalGraph().toGroup(allocation_domain);
        });
    contig_strides = {alloc_info.strides.begin(), alloc_info.strides.end()};
  }

  std::vector<Val*> result;
  result.reserve(contig_alloc_groups.size());

  for (const auto i : c10::irange(contig_alloc_groups.size())) {
    const auto& contig_domain_group = contig_alloc_groups.at(i);
    auto idx_it = index_map.find(contig_domain_group);
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        contig_domain_group->front()->toString());
    Val* idx = idx_it->second;
    Val* replaced_idx = ir_utils::replaceValRecursively(idx, replacement_map);
    result.push_back(replaced_idx);
  }

  return {result, contig_strides};
}

} // namespace nvfuser

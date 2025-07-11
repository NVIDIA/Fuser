// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bfs.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/allocation.h>
#include <expr_evaluator.h>
#include <expr_simplifier.h>
#include <id_model/utils.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

#include <unordered_set>

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
      !id->isReduction() && !id->isStride();
}

// Get the allocation stride of a given allocation domain
Val* getStrideOfGlobalMemoryTensor(TensorView* tv, int64_t alloc_dim) {
  NVF_ERROR(tv->getMemoryType() == MemoryType::Global);

  // Allocation domains can include reduction domains, but
  // alloc_stride arrays do not.
  const auto& alloc_dom = tv->getMaybeAllocationDomain();
  int64_t stride_dim = -1;
  for (const auto i : arange(alloc_dim + 1)) {
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

      // Not yet set. This must be an input tensor or it must be aliased via
      // aliasTensorProducer, in which case it will not be allocated.
      NVF_ERROR(
          producer_tv->isFusionInput() ||
              GpuLower::current()->getTensorProducerAlias(producer_tv) !=
                  nullptr,
          "Expected a fusion input or aliased tensor but found: ",
          producer_tv->toString());

      // For fusion input, we can just use getMaybeAllocationDomain.

      auto alloc_info = getAllocationDomainInfo(
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
            getAllocationDomainInfo(out_tv, alloc_domains, contiguity);
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
                tv->getAllocationDomain().begin(),
                tv->getAllocationDomain().end())) {
          use_set_allocation_domain = true;
        }

        // Honor the set allocation domain if the tensor is used by a
        // TMA store or MmaOp
        if (std::ranges::any_of(tv->uses(), [](Expr* expr) {
              return ir_utils::isCpAsyncBulkStore(expr) || expr->isA<MmaOp>();
            })) {
          use_set_allocation_domain = true;
        }

        // If a shared memory output produced by scatter has an
        // allocation domain explicitly set, it's likely to be the
        // valid allocation domain.
        if (auto def = tv->definition();
            def != nullptr && def->isA<ScatterOp>()) {
          use_set_allocation_domain = true;
        }
      }
    }

    // Allocation position is not always the same as the CA
    // position. See also lower_utils::getAllocInformation.
    int64_t allocation_pos =
        lower_utils::getAllocPosInfo(tv, for_loops).alloc_pos;

    if (use_set_allocation_domain) {
      if (tv->getMemoryType() == MemoryType::Global) {
        // For global memory tensors we always allocate the entire tensor
        // TODO: do we really want to treat global memory tensors differently?
        // need to think about this more.
        allocation_domains = tv->getAllocationDomain();
        contiguity = tv->domain()->contiguity();
      } else {
        std::unordered_set<IterDomain*> exclude_ca_ids;
        for (auto i : arange(allocation_pos)) {
          auto ca_id = tv->axis(i);
          if (!ir_utils::isMemorySharedAcross(
                  tv->getMemoryType(), ca_id->getParallelType())) {
            exclude_ca_ids.insert(ca_id);
          }
        }
        for (auto i : arange(tv->getAllocationDomain().size())) {
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
            "The non-allocating compute-at IDs are not found in the allocation "
            "domain. ",
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
        for (const auto i : arange(tv->nDims())) {
          auto loop_id = tv->getLoopDomain().at(i);
          auto pt = loop_id->getParallelType();

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

      if (auto indexed_alloc_dom =
              patchAllocationOfIndexedProducerTensor(tv, allocation_domains);
          indexed_alloc_dom.has_value()) {
        allocation_domains = indexed_alloc_dom.value();
        // Make sure the original allocation domains are fully contiguous
        NVF_ERROR(std::all_of(contiguity.begin(), contiguity.end(), [](auto b) {
          return b.has_value() && b.value();
        }));
        // Set the new allocation domains fully contiguous
        contiguity =
            std::vector<std::optional<bool>>(allocation_domains.size(), true);
      }

      // reorderAllocationDomains and
      // patchAllocationOfTransposedSmemTensor assume unallocated IDs
      // are removed
      std::vector<IterDomain*> actual_allocation_ids;
      std::vector<std::optional<bool>> actual_contiguity;
      for (auto [i, id] : enumerate(allocation_domains)) {
        if (mayRequireAllocation(tv, id)) {
          actual_allocation_ids.push_back(id);
          actual_contiguity.push_back(contiguity.at(i));
        }
      }
      std::swap(allocation_domains, actual_allocation_ids);
      std::swap(contiguity, actual_contiguity);

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

  // Get allocation info necessary for allocation and indexing. Loop promotion
  // is considered. Strides are also calculated.
  AllocationDomainInfo getAllocationDomainInfo(
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
      bool is_loop = std::find(
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
    for (const auto i : arange(allocation_domains.size())) {
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

    // Filter out non-allocated domains
    std::vector<IterDomain*> actual_allocation_domains;
    std::vector<Val*> actual_strides;
    std::vector<bool> actual_contiguity;
    for (const auto i : arange(allocation_domains.size())) {
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

    return AllocationDomainInfo{
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

  // If a producer tensor is accessed through supplied indices, the
  // indexed logical IDs need to be entirely allocated.
  std::optional<std::vector<IterDomain*>> patchAllocationOfIndexedProducerTensor(
      const TensorView* tv,
      const std::vector<IterDomain*>& allocation_ids) const {
    VectorOfUniqueEntries<Val*> indexed_logical_ids;
    for (auto use_expr : tv->uses()) {
      auto indexed_id = ir_utils::getIndexedProducerID(use_expr);
      if (indexed_id == nullptr ||
          std::find(
              tv->getLogicalDomain().begin(),
              tv->getLogicalDomain().end(),
              indexed_id) == tv->getLogicalDomain().end()) {
        continue;
      }

      // This indexed_id is indirectly accessed and needs to be
      // allocated entirely.

      // If it's already in the allocation ID set, nothing further
      // needs to be done
      if (std::find(allocation_ids.begin(), allocation_ids.end(), indexed_id) !=
          allocation_ids.end()) {
        continue;
      }

      indexed_logical_ids.pushBack(indexed_id);
    }

    if (indexed_logical_ids.empty()) {
      return std::nullopt;
    }

    // indexed_logical_ids is not in the current allocation ID
    // list. Find the allocation IDs that are equivalent to the
    // indexed IDs. The indexed IDs should be reachable from the
    // allocation IDs, and those allocation IDs used in the traversal
    // path should be the ones that should be replaced with the
    // indexed IDs.

    // In order to retain the original ordering of allocation IDs,
    // each indexed logical ID is examined one by one. Specifically,
    // for each of them, we find the corresponding IDs in the current
    // allocation ID vector and replace them with the indexed logical
    // ID.
    auto patched_allocation_ids = allocation_ids;
    for (auto indexed_logical_id : indexed_logical_ids) {
      auto [path, all_visited] = getExprsBetween<IRBFS>(
          {patched_allocation_ids.begin(), patched_allocation_ids.end()},
          {indexed_logical_id},
          /*require_all_to_visited=*/false);
      NVF_ERROR(
          all_visited,
          "Failed to infer valid allocation IDs. Indexed logical IDs need to "
          "be entirely allocated but not found in the inferred allocation ID "
          "set. Indexed logical ID: ",
          indexed_logical_id->toString(),

          ". Allocation IDs: ",
          toDelimitedString(patched_allocation_ids));

      auto dependent_allocation_ids = getInputsOfExprPath<IRBFS>(path);

      // Insert indexed_logical_id at the innermost position of
      // dependent_allocation_ids.
      int num_dependent_allocation_ids = 0;
      std::vector<IterDomain*> pathched_allocation_ids_next;
      for (auto id : allocation_ids) {
        if (std::find(
                dependent_allocation_ids.begin(),
                dependent_allocation_ids.end(),
                id) != dependent_allocation_ids.end()) {
          ++num_dependent_allocation_ids;
          if (num_dependent_allocation_ids ==
              std::ssize(dependent_allocation_ids)) {
            pathched_allocation_ids_next.push_back(
                indexed_logical_id->as<IterDomain>());
          }
        } else {
          pathched_allocation_ids_next.push_back(id);
        }
      }

      std::swap(patched_allocation_ids, pathched_allocation_ids_next);
    }

    return patched_allocation_ids;
  }

  std::unordered_map<TensorView*, AllocationDomainInfo> tv_alloc_info_map;
  std::unordered_set<TensorView*> used_as_producer;
};

} // namespace

namespace {

enum class CircularBufferWaitType { ReadAfterWrite, WriteAfterRead };

// This function creates kir::Loop with range based on stage depth. It is
// used for mbarrier initialization and invalidation.
ForLoop* createStageDepthForLoop(ForLoop* circular_buffer_loop) {
  int64_t stage_depth =
      GpuLower::current()
          ->circularBufferInfo()
          .getCircularBufferOptionsFor(circular_buffer_loop->iter_domain())
          .stage;
  return ir_utils::createRangeLoop(stage_depth);
}

// This helper function initializes mbarrier for all circular buffer stage.
//
// Expected result:
// for (unsigned i = 0; i < stages; ++i) {
//   if (warp_id == 0 && electSync()()) {
//     mbarrier::init(...);
//   }
// }
Expr* initializeMbarrier(
    ForLoop* circular_buffer_loop,
    TensorView* all_mbarriers,
    CircularBufferWaitType wait_type) {
  NVF_ERROR(circular_buffer_loop != nullptr);
  ForLoop* loop = createStageDepthForLoop(circular_buffer_loop);

  int64_t stage_depth =
      GpuLower::current()
          ->circularBufferInfo()
          .getCircularBufferOptionsFor(circular_buffer_loop->iter_domain())
          .stage;

  // We use mbarrier[0:stage_depth] for RAW, and
  // mbarrier[stage_depth:2*stage_depth] for WAR.
  Val* mbarrier_index = wait_type == CircularBufferWaitType::ReadAfterWrite
      ? loop->index()
      : SimplifyingIrBuilder::addExpr(loop->index(), stage_depth);

  // Get mbarrier for this circular buffer stage.
  kir::TensorIndex* stage_mbarrier =
      IrBuilder::create<kir::TensorIndex>(all_mbarriers, mbarrier_index);

  auto circular_buffered_tvs =
      GpuLower::current()->circularBufferInfo().getCircularBufferTvs(
          circular_buffer_loop);

  Val* num_of_arrives = nullptr;
  if (wait_type == CircularBufferWaitType::ReadAfterWrite) {
    // The mbarrier of RAW is used to wait for the completion of the TMA
    // load of the circular buffer tensor. The number of arrives is the
    // number of TMA issued for the circular buffer tensor.
    int64_t num_of_tvs_loaded_by_tma = std::count_if(
        circular_buffered_tvs.begin(),
        circular_buffered_tvs.end(),
        [](const TensorView* tv) {
          return ir_utils::isCpAsyncBulkLoad(tv->definition());
        });
    num_of_arrives =
        IrBuilder::create<Val>(num_of_tvs_loaded_by_tma, DataType::UInt32);
  } else {
    // The mbarrier of WAR is used to wait for the completion of the reading
    // of the circular buffer tensor. The number of arrives is the number of
    // threads in the CTA.
    num_of_arrives = SimplifyingIrBuilder::maybeCastExpr(
        DataType::UInt32,
        GpuLower::current()
            ->parallelDimensionMap()
            .getNumComputeThreadsEachBlock());
  }

  // Initialize mbarrier for each circular buffer stage. Use the thread
  // count from the MBarrierInit created in the allocation pass. The wait
  // condition for mbarrier is a all threads in CTA and the expected number
  // of transaction bytes
  kir::MBarrierInit* mbarrier_init =
      IrBuilder::create<kir::MBarrierInit>(stage_mbarrier, num_of_arrives);

  Expr* pred_mbarrier_init = mbarrier_init->withPredicate(
      IrBuilder::create<kir::Predicate>(PredicateType::ElectSync));
  loop->body().push_back(pred_mbarrier_init);
  return loop;
}

// This helper function invalidates mbarrier for all circular buffer stage after
// TMA memory operations.
//
// Expected result:
// for (unsigned i = 0; i < stages; ++i) {
//   if (warp_id == 0 && electSync()()) {
//     mbarrier::inval(...);
//   }
// }
Expr* invalidateMbarrier(
    ForLoop* circular_buffer_loop,
    TensorView* all_mbarriers,
    CircularBufferWaitType wait_type) {
  NVF_ERROR(circular_buffer_loop != nullptr);
  ForLoop* loop = createStageDepthForLoop(circular_buffer_loop);

  int64_t stage_depth =
      GpuLower::current()
          ->circularBufferInfo()
          .getCircularBufferOptionsFor(circular_buffer_loop->iter_domain())
          .stage;

  // We use mbarrier[0:stage_depth] for RAW, and
  // mbarrier[stage_depth:2*stage_depth] for WAR.
  Val* mbarrier_index = wait_type == CircularBufferWaitType::ReadAfterWrite
      ? loop->index()
      : SimplifyingIrBuilder::addExpr(loop->index(), stage_depth);

  // Get mbarrier for this circular buffer stage.
  kir::TensorIndex* stage_mbarrier =
      IrBuilder::create<kir::TensorIndex>(all_mbarriers, mbarrier_index);

  // Invalidate the mbarrier for each circular buffer stage.
  kir::MBarrierInvalidate* mbarrier_inval =
      IrBuilder::create<kir::MBarrierInvalidate>(stage_mbarrier);

  Expr* pred_mbarrier_inval = mbarrier_inval->withPredicate(
      IrBuilder::create<kir::Predicate>(PredicateType::ElectSync));

  loop->body().push_back(pred_mbarrier_inval);
  return loop;
}

class AllocationInserter : public kir::ExprMutator {
 private:
  using kir::ExprMutator::handle;

  // Expanded version of BasicAllocInfo in lower_utils.h helps to track
  // additional information
  struct AllocationInformation {
    // The for loop that the initialization of this allocation must be
    // placed in, nullptr if not within a loop
    ForLoop* init_for_loop = nullptr;

    // The expression that the initialization of this allocation must
    // be placed before
    Expr* init_place_before = nullptr;

    // Keep track of the actual allocation loop. This can be different
    // from init_for_loop only with unswitched shared memory allocations,
    // which are moved outer loops to avoid duplicated allocations
    // (see issue #1133).
    ForLoop* alloc_for_loop = nullptr;

    // The expression that this allocation must be placed
    // before. Similar to alloc_for_loop, this is different from
    // init_place_before only with unswitched shared memory allocations.
    Expr* alloc_place_before = nullptr;

    // The allocation position relative to buffer
    int64_t alloc_pos = 0;

    // The buffer this allocation is for
    TensorView* buffer = nullptr;

    // Local Iterdomains that this allocation covers
    std::unique_ptr<std::vector<IterDomain*>> allocation_domains;
  };

  // Find allocation point
  // Fills info.buffer, info.alloc_pos, info.init_for_loop,
  // info.init_place_before, info.alloc_for_loop, info.alloc_place_before
  void fillAllocationInformation(AllocationInformation& info, Expr* expr) {
    auto loop_alloc_info =
        lower_utils::getAllocPosInfo(info.buffer, for_loops_);

    info.init_for_loop = loop_alloc_info.init_for_loop;
    info.alloc_for_loop = loop_alloc_info.alloc_for_loop;
    info.alloc_pos = loop_alloc_info.alloc_pos;

    auto next_fl = [](ForLoop* fl, const std::vector<ForLoop*> fls) {
      for (auto i : arange(fls.size())) {
        if (fl == fls[i]) {
          if (i + 1 < fls.size()) {
            return fls[i + 1];
          }
        }
      }
      NVF_THROW("Could not find desired loop.");
    };

    if (info.init_for_loop == nullptr) {
      info.init_place_before = !for_loops_.empty() ? for_loops_[0] : expr;
    } else {
      if (info.init_for_loop == for_loops_.back()) {
        // Inline allocation, place before expr
        info.init_place_before = expr;
      } else {
        // Place allocation after the last computeAt axis
        // TODO: may be more efficient to place before the first non-computeAt
        // axis
        info.init_place_before = next_fl(info.init_for_loop, for_loops_);
      }
    }

    // Set the allocation loop and the place_before expression in the
    // same way as the initialization loop and place_before expression
    if (info.alloc_for_loop == info.init_for_loop) {
      info.alloc_for_loop = info.init_for_loop;
      info.alloc_place_before = info.init_place_before;
    } else {
      if (info.alloc_for_loop == nullptr) {
        info.alloc_place_before = !for_loops_.empty() ? for_loops_[0] : expr;
      } else {
        // Since there must be an inner unswitched domain,
        // alloc_for_loop should never be the inner-most loop.
        NVF_ERROR(info.alloc_for_loop != for_loops_.back());
        info.alloc_place_before = next_fl(info.alloc_for_loop, for_loops_);
      }
    }
  }

  // Create initialization expression if init_val is non-null.
  Expr* createInitExpr(AllocationInformation& info, Val* init_val) {
    if (init_val == nullptr) {
      return nullptr;
    }

    std::vector<IterDomain*> init_dims;
    for (const auto axis_i : arange(info.alloc_pos, info.buffer->nDims())) {
      if (info.buffer->axis(axis_i)->isReduction() ||
          info.buffer->axis(axis_i)->isBroadcast()) {
        continue;
      }
      auto concrete_id =
          lower_utils::getConcreteLoopID(info.buffer->axis(axis_i));
      init_dims.push_back(concrete_id);
    }
    Expr* init_expr = IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, info.buffer, init_val);
    for (auto init_loop_it = init_dims.rbegin();
         init_loop_it != init_dims.rend();
         ++init_loop_it) {
      auto id = *init_loop_it;
      ForLoop* new_loop = IrBuilder::create<ForLoop>(id);
      new_loop->body().push_back(init_expr);
      init_expr = new_loop;
    }
    return init_expr;
  }

  kir::Allocate* createAllocExpr(AllocationInformation& info) {
    // Note that Allocate nodes are created for fusion outputs too

    TensorView* tv_to_alloc = info.buffer;
    const MemoryType memory_type = tv_to_alloc->getMemoryType();

    NVF_ERROR(
        tv_to_alloc->definition() != nullptr,
        "Unexpected to have a tensor with no definition: ",
        tv_to_alloc->toString());

    const auto& alloc_ids =
        GpuLower::current()->getAllocationInfo(tv_to_alloc).ids;
    std::vector<Val*> alloc_dims;
    alloc_dims.reserve(alloc_ids.size());
    for (const auto& id : alloc_ids) {
      alloc_dims.push_back(id->extent());
    }
    info.allocation_domains =
        std::make_unique<std::vector<IterDomain*>>(alloc_ids);

    if (alloc_dims.empty() && !info.buffer->domain()->noReductions().empty()) {
      alloc_dims.push_back(info.buffer->container()->oneVal());
    }

    // Multiply the allocation size if circular-buffered. Record the
    // original size for indexing.
    if (info.buffer->isCircularBuffered()) {
      Val* original_alloc_size = nullptr;
      for (auto alloc_dim : alloc_dims) {
        if (original_alloc_size == nullptr) {
          original_alloc_size = alloc_dim;
        } else {
          original_alloc_size =
              IrBuilder::mulExpr(original_alloc_size, alloc_dim);
        }
      }
      GpuLower::current()->circularBufferInfo().setOriginalAllocSize(
          info.buffer, original_alloc_size);
      int64_t circular_buffer_stage =
          info.buffer->circularBufferOptions().stage;
      alloc_dims.push_back(
          IrBuilder::create<Val>(circular_buffer_stage, DataType::Index));
    }

    // Create the allocation node
    auto alloc_expr = IrBuilder::create<kir::Allocate>(
        info.buffer, info.buffer->getMemoryType(), alloc_dims);

    // Fill in the base address, lane offset, and column offset for tensor
    // memory allocations
    if (memory_type == MemoryType::Tensor) {
      const auto& regions = GpuLower::current()->tmemInfo().allocation.regions;
      for (const auto& region : regions) {
        auto tv_info_it = std::find_if(
            region.covered_tensors.begin(),
            region.covered_tensors.end(),
            [&](const auto& tv_info) { return tv_info.tensor == info.buffer; });
        if (tv_info_it != region.covered_tensors.end()) {
          auto address_ti = IrBuilder::create<kir::TensorIndex>(
              region.address, region.address->fusion()->zeroVal());
          alloc_expr->setAddress(address_ti);
          alloc_expr->setLaneOffset(tv_info_it->lane_offset);
          alloc_expr->setColOffset(tv_info_it->column_offset);
          break;
        }
      }
      NVF_ERROR(
          alloc_expr->address() != nullptr,
          "Could not find region for tensor memory allocation of ",
          info.buffer);
    }

    return alloc_expr;
  }

  void dispatch(Expr* expr) override {
    if (!ir_utils::isTvOp(expr) || expr->isA<kir::Allocate>() ||
        expr->isA<kir::AllocTMem>()) {
      ExprMutator::dispatch(expr);
      return;
    }

    int64_t circular_buffer_depth = 1;

    // Found where the allocation needs to be inserted

    for (const auto i : arange(expr->outputs().size())) {
      auto out = expr->output(i);
      if (!out->isA<TensorView>()) {
        continue;
      }

      auto out_tv = out->as<TensorView>();
      auto default_val =
          gpu_lower_->predicateElimination().getInitValue(out_tv);

      Val* init = nullptr;
      if (out_tv->dtype() == DataType::Float4_e2m1fn) {
        // TODO: fp4 is smaller than one byte, it is impossible to specify a
        // fp4 value in computer. For now, we just skip the initialization.
        init = nullptr;
      } else if (expr->isA<ReductionOp>() && out_tv->hasReduction()) {
        NVF_ERROR(
            default_val == nullptr,
            "Reduction should not have a default initialization value for "
            "predicate elimination.");
        init = expr->as<ReductionOp>()->init();
      } else if (expr->isA<GroupedReductionOp>() && out_tv->hasReduction()) {
        NVF_ERROR(
            default_val == nullptr,
            "Reduction should not have a default initialization value for "
            "predicate elimination.");
        init = expr->as<GroupedReductionOp>()->initVal(i);
      } else if (MmaOp* mma = dynamic_cast<MmaOp*>(expr)) {
        // On Hopper and Blackwell, we generate code like:
        // for k in ...:
        //   mma(acc, a, b, use_input_acc=(k!=0))
        // For this case, there is no need to initialize the accumulator
        // as it is initialized with zero by the MMA instruction.
        if (mma->isHopper() || mma->isBlackwell()) {
          NVF_ERROR(
              mma->init() == nullptr || mma->init()->isZero(),
              "Hopper and Blackwell MMA should not have a non-zero "
              "initialization value.");
          init = nullptr;
        } else {
          // On Turing and Ampere, we manually initialize the accumulator
          init = mma->init();
        }
      } else if (expr->isA<WelfordOp>()) {
        NVF_ERROR(
            default_val == nullptr,
            "Welford should not have a default initialization value for "
            "predicate elimination.");
        const auto welford = expr->as<WelfordOp>();
        if (out->name() == welford->outVar()->name()) {
          init = welford->initVar() == nullptr ? IrBuilder::create<Val>(0.0)
                                               : welford->initVar();
        } else if (out->name() == welford->outAvg()->name()) {
          init = welford->initAvg() == nullptr ? IrBuilder::create<Val>(0.0)
                                               : welford->initAvg();
        } else {
          NVF_ERROR(out->name() == welford->outN()->name(), "Unreachable");
          init = welford->initN();
        }
      } else if (expr->isA<GroupedWelfordOp>()) {
        NVF_ERROR(
            default_val == nullptr,
            "Welford should not have a default initialization value for "
            "predicate elimination.");
        init = expr->as<GroupedWelfordOp>()->getInitValOfOutput(out);
      } else if (out_tv->getMemoryType() != MemoryType::Tensor) {
        // TODO: TMem should not be initialized as ... = 0, because it must be
        // accessed with special instructions. We do not have a good way to
        // initialize TMem yet. For now, we just skip the initialization for
        // TMem.
        init = default_val;
      }

      if (ir_utils::isCpAsyncOp(expr) || ir_utils::isCpAsyncBulk(expr)) {
        NVF_CHECK(
            init == nullptr || init->isZero(),
            "cp.async and cp.async.bulk initialized with non-zero is not "
            "supported");
        // cp.async will automatically fill zero when out of bound
        init = nullptr;
      }

      AllocationInformation allocation;
      allocation.buffer = out_tv;
      fillAllocationInformation(allocation, expr);

      auto alloc_expr = createAllocExpr(allocation);
      auto init_expr = createInitExpr(allocation, init);

      // Check that all circular buffer depth match
      if (out_tv->isCircularBuffered() && circular_buffer_depth == 1) {
        circular_buffer_depth = out_tv->circularBufferOptions().stage;
      }
      NVF_ERROR(
          circular_buffer_depth == 1 ||
              circular_buffer_depth == out_tv->circularBufferOptions().stage,
          "Expected all output TensorViews for the same expression ",
          "to have the same circular_buffer_depth");

      // Write information to GPULower
      writeInfoToGPULower(allocation, alloc_expr);

      // Register allocations before initializations to keep them in the right
      // order
      if (alloc_expr != nullptr) {
        if (allocation.buffer->getMemoryType() == MemoryType::Shared) {
          // Shared allocations go at the begining of scope
          NVF_ERROR(!exprs_.empty());
          registerInsertBefore(exprs_[0], alloc_expr, nullptr);
        } else {
          NVF_ERROR(allocation.alloc_place_before != nullptr);
          Scope* scope = allocation.alloc_for_loop == nullptr
              ? nullptr
              : &allocation.alloc_for_loop->body();
          registerInsertBefore(
              allocation.alloc_place_before, alloc_expr, scope);
        }
      }

      if (init_expr != nullptr) {
        NVF_ERROR(allocation.init_place_before != nullptr);
        Scope* scope = allocation.init_for_loop == nullptr
            ? nullptr
            : &allocation.init_for_loop->body();
        registerInsertBefore(allocation.init_place_before, init_expr, scope);

        if (auto mma = dynamic_cast<MmaOp*>(expr)) {
          if (mma->isHopper()) {
            if (lower_utils::allMmaInputsGuardedByMBarrier(mma)) {
              // When all inputs are guarded by mbarrier, we will not insert
              // generic-async proxy fence and wgmma fence before each mma
              // instruction. For this case, we need to insert these fences
              // after the initialization of the accumulator, so that the
              // initialization is visible to the async proxy.
              // When all inputs are guarded by mbarrier, we will insert these
              // fences before each mma instruction, so there is no need to
              // insert them after the initialization of the accumulator here.
              auto wgmma_fence = IrBuilder::create<kir::WgMmaFence>();
              registerInsertBefore(
                  allocation.init_place_before, wgmma_fence, scope);
              auto fence_async = IrBuilder::create<kir::FenceAsyncProxy>();
              registerInsertBefore(
                  allocation.init_place_before, fence_async, scope);
            }
          }
        }
      }
    }

    // Allocate mbarrier for cp.async.bulk, for non-circular buffered cases by
    // lowering a single cp.async.bulk as:
    //    alloc mbarrier
    //    init mbarrier
    //    block_sync
    //    cp.async.bulk
    //    inval mbarrier
    //    block_sync
    //
    // * The circular buffer case is handled in handle(ForLoop* fl) and the
    // circular buffering pass.
    // * Assume that the tma load is in ComputeWarp if it is not circular
    // buffered.
    if ((ir_utils::isCpAsyncBulkLoad(expr) && circular_buffer_depth == 1) ||
        (expr->isA<MmaOp>() && expr->as<MmaOp>()->isBlackwell())) {
      // create and allocate a memory barrier
      TensorView* mbarrier = TensorViewBuilder()
                                 .shape(std::vector<int64_t>{})
                                 .dtype(DataType::UInt64)
                                 .contiguity(true)
                                 .build();
      mbarrier->setMemoryType(MemoryType::Shared);
      auto mbarrier_init = IrBuilder::create<kir::MBarrierInit>(
          mbarrier,
          simplifyExpr(SimplifyingIrBuilder::maybeCastExpr(
              DataType::UInt32,
              expr->isA<MmaOp>() ? expr->fusion()->oneVal()
                                 : lower_utils::getNumThreadsInTensorView(
                                       expr->output(0)->as<TensorView>()))));
      auto sync_init = IrBuilder::create<kir::BlockSync>(
          /*war_sync=*/false, /*optional_compute_or_load_sync=*/true);
      auto mbarrier_inval =
          IrBuilder::create<kir::MBarrierInvalidate>(mbarrier);
      auto sync_inval = IrBuilder::create<kir::BlockSync>(
          /*war_sync=*/false, /*optional_compute_or_load_sync=*/true);

      kir::Allocate* mbarrier_alloc =
          IrBuilder::create<kir::Allocate>(mbarrier, MemoryType::Shared);
      Scope* expr_scope = scope_.empty() ? nullptr : scope_.back();
      registerInsertBefore(expr, mbarrier_alloc, expr_scope);
      registerInsertBefore(expr, mbarrier_init, expr_scope);
      registerInsertBefore(expr, sync_init, expr_scope);
      registerInsertAfter(expr, mbarrier_inval, expr_scope);
      registerInsertAfter(expr, sync_inval, expr_scope);
      GpuLower::current()->mbarrierMap()[expr] = mbarrier;
    }
  }

  void handle(ForLoop* fl) final {
    ExprMutator::handle(fl);

    // If fl is a circular buffered loop, the we should lowering the loop as:
    //    alloc mbarrier
    //    init mbarrier
    //    block_sync
    //    for-loop with cpAsyncBulk expression (the `fl` parameter)
    //    inval mbarrier

    auto circular_buffer_tvs =
        GpuLower::current()->circularBufferInfo().getCircularBufferTvs(fl);

    bool circular_buffer_load_is_tma = std::any_of(
        circular_buffer_tvs.begin(),
        circular_buffer_tvs.end(),
        [](const TensorView* tv) {
          return ir_utils::isCpAsyncBulkLoad(tv->definition());
        });

    if (circular_buffer_load_is_tma) {
      // Create and allocate a memory barrier. If this is a circular buffer,
      // then allocate an array of mbarier objects. mbarrier::init and
      // mbarrier::inval will be updated in circular buffering pass, but we
      // add them here to handle shared memory correctly in alias memory pass.
      const auto& opt =
          GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
              fl->iter_domain());

      // We use mbarrier[0:stage] for RAW, that is, to wait for the completion
      // of the TMA load of the circular buffer tensor, and
      // mbarrier[stage:2*stage] for WAR, that is, to wait for the completion of
      // the reading of the circular buffer tensor.
      int64_t num_mbarriers =
          opt.usesMBarrierForWAR() ? opt.stage * 2 : opt.stage;

      TensorView* mbarrier = TensorViewBuilder()
                                 .shape(std::vector<int64_t>{num_mbarriers})
                                 .dtype(DataType::UInt64)
                                 .contiguity(true)
                                 .build();
      mbarrier->setMemoryType(MemoryType::Shared);

      kir::Allocate* mbarrier_alloc =
          IrBuilder::create<kir::Allocate>(mbarrier, MemoryType::Shared);

      // Initialize and invalidate mbarriers that are used to notify that
      // the load of the circular buffer is complete.
      auto mbarrier_init_raw = initializeMbarrier(
          fl, mbarrier, CircularBufferWaitType::ReadAfterWrite);
      auto mbarrier_inval_raw = invalidateMbarrier(
          fl, mbarrier, CircularBufferWaitType::ReadAfterWrite);

      // Block sync is necessary to finish mbarrier initialization.
      kir::BlockSync* sync = IrBuilder::create<kir::BlockSync>(false);

      // Add mbarriers, init, and inval operations around tma expression like
      // this:
      //
      // __shared__ mbarrier[num_stages];
      // for (circular_buffer_stage) {
      //   // initialize mbarrier for RAW
      //   init(mbarrier[stage]);
      // }
      // for (circular_buffer_stage) {
      //   // initialize mbarrier for WAR
      //   init(mbarrier[stage]);
      // }
      // block_sync();
      //
      // for (circular_buffer_loop) {
      //   cp.async.bulk(data, mbarrier);
      // }
      //
      // for (circular_buffer_stage) {
      //   // invalidate mbarrier for WAR
      //   inval(mbarrier[stage]);
      // }
      // for (circular_buffer_stage) {
      //   // invalidate mbarrier for RAW
      //   inval(mbarrier[stage]);
      // }
      //
      Scope* current_scope = scope_.empty() ? nullptr : scope_.back();
      registerInsertBefore(fl, mbarrier_alloc, current_scope);
      registerInsertBefore(fl, mbarrier_init_raw, current_scope);
      registerInsertAfter(fl, mbarrier_inval_raw, current_scope);

      if (opt.usesMBarrierForWAR()) {
        auto mbarrier_init_war = initializeMbarrier(
            fl, mbarrier, CircularBufferWaitType::WriteAfterRead);
        auto mbarrier_inval_war = invalidateMbarrier(
            fl, mbarrier, CircularBufferWaitType::WriteAfterRead);
        registerInsertBefore(fl, mbarrier_init_war, current_scope);
        registerInsertAfter(fl, mbarrier_inval_war, current_scope);
      }
      registerInsertBefore(fl, sync, current_scope);

      for (auto tv : circular_buffer_tvs) {
        // short-circuit: circular buffered tv is not defined with TMA load.
        if (!ir_utils::isCpAsyncBulkLoad(tv->definition())) {
          continue;
        }
        // Map LoadStoreOp expression to ir nodes created in this pass
        GpuLower::current()->mbarrierMap()[tv->definition()] = mbarrier;
      }
    }
  }

  // Sends alloc_expr, info.allocation_domains to GpuLower
  void writeInfoToGPULower(
      const AllocationInformation& allocation,
      kir::Allocate* alloc_expr) {
    auto& lower_alloc_info_map = GpuLower::current()->localAllocationInfoMap();
    if (alloc_expr == nullptr) {
      // Skip output allocation.
      return;
    }
    NVF_ERROR(
        !lower_alloc_info_map.count(alloc_expr),
        "duplicated allocation info entry");

    // Create info entry for GPULower
    auto lower_alloc_info_ptr = std::make_unique<LocalAllocationInfo>();
    lower_alloc_info_ptr->alloc_expr = alloc_expr;
    if (allocation.allocation_domains) {
      lower_alloc_info_ptr->alloc_domains = *(allocation.allocation_domains);
    }

    // Write entry to the stored map
    lower_alloc_info_map[alloc_expr] = std::move(lower_alloc_info_ptr);
  }

  void handle(kir::IfThenElse* ite) final {
    // TODO: Currently we just naively dispatch into the IfThenElse node
    // assuming that this does not affect the analysis. For now, this assumption
    // is true, but in the future, we might need to revisit this.
    kir::ExprMutator::handle(ite);
  }

  AllocationInserter(const std::vector<Expr*>& exprs)
      : gpu_lower_(GpuLower::current()) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

 private:
  GpuLower* gpu_lower_ = nullptr;

 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    AllocationInserter inserter(exprs);
    return inserter.exprs_;
  }
};

namespace {

// Create `if (is first warp)`, depending on whether the parallel types are
// used in the schedule, the generated code may be different.
kir::IfThenElse* createFirstWarpITE() {
  const auto& pdim = GpuLower::current()->parallelDimensionMap();
  Val* tid = FusionGuard::getCurFusion()->zeroVal();
  Val* bdimx = pdim.getRaw(ParallelType::TIDx);
  Val* bdimy = pdim.getRaw(ParallelType::TIDy);
  Val* bdimz = pdim.getRaw(ParallelType::TIDz);
  if (bdimx != nullptr) {
    tid = NamedScalar::getParallelIndex(ParallelType::TIDx);
  }
  if (bdimy != nullptr) {
    Val* tidy = NamedScalar::getParallelIndex(ParallelType::TIDy);
    if (bdimx != nullptr) {
      tidy = SimplifyingIrBuilder::mulExpr(tidy, bdimx);
    }
    tid = SimplifyingIrBuilder::addExpr(tid, tidy);
  }
  if (bdimz != nullptr) {
    Val* tidz = NamedScalar::getParallelIndex(ParallelType::TIDz);
    if (bdimy != nullptr) {
      tidz = SimplifyingIrBuilder::mulExpr(tidz, bdimy);
    }
    if (bdimx != nullptr) {
      tidz = SimplifyingIrBuilder::mulExpr(tidz, bdimx);
    }
    tid = SimplifyingIrBuilder::addExpr(tid, tidz);
  }
  Val* first_warp =
      SimplifyingIrBuilder::ltExpr(tid, IrBuilder::create<Val>(32));
  kir::Predicate* pred = IrBuilder::create<kir::Predicate>(first_warp);
  return IrBuilder::create<kir::IfThenElse>(pred);
}

} // namespace

// Insert IR nodes that allocate and deallocate TMem regions.
// See note [Tensor Memory Allocation] for the overall design.
// We insert the tcgen05.allocs of each region and the relinquish of the right
// to allocate at the beginning of the top-level scope of the kernel. We insert
// the tcgen05.deallocs after the outermost serial loop containing the last read
// of each TMem region into whatever scope containing this outermost serial
// loop. The allocation of each TMem TensorView within each region is inserted
// by AllocationInserter::insert, therefore not handled here.
std::vector<Expr*> insertTMemRegionAllocsAndDeallocs(
    const std::vector<Expr*>& exprs) {
  // Expressions to be inserted at the beginning of the top-level scope.
  std::list<Expr*> prologue;
  {
    const auto& regions = GpuLower::current()->tmemInfo().allocation.regions;
    // For each TMem region, allocate its address in shared memory, and insert
    // the tcgen05.alloc for tensor memory allocation.
    for (const auto& region : regions) {
      // kir::Allocate for the address tensor on shared memory
      auto address_alloc_expr =
          IrBuilder::create<kir::Allocate>(region.address, MemoryType::Shared);
      prologue.push_back(address_alloc_expr);
      // the tcgen05.alloc instruction
      auto first_warp = createFirstWarpITE();
      auto alloc_expr =
          IrBuilder::create<kir::AllocTMem>(region.address, region.num_columns);
      first_warp->thenBody().push_back(alloc_expr);
      prologue.push_back(first_warp);
    }

    if (!regions.empty()) {
      // Relinquish the right to allocate after all regions have been allocated
      auto first_warp = createFirstWarpITE();
      auto tcgen05_relinquish_expr = IrBuilder::create<kir::Asm>(
          "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
          std::vector<Val*>{},
          std::vector<Val*>{},
          kir::Asm::Options{/*volatile=*/true});
      first_warp->thenBody().push_back(tcgen05_relinquish_expr);
      prologue.push_back(first_warp);

      // Block sync that makes allocation visible to all threads
      auto block_sync = IrBuilder::create<kir::BlockSync>();
      prologue.push_back(block_sync);
    }
  }

  // Add deallocations to existing expressions
  std::vector<Expr*> exprs_with_deallocs;
  {
    class DeallocInserter : public kir::ExprMutator {
      // A map:
      //   region -> a function that registers the deallocation expression for
      //             this region
      //
      // This map is updated during traversal. For example, if we have a kernel
      // like below:
      //   ...
      //   T1_t = T0_r; // expr1
      //   ...
      //   T2_r = T1_t; // expr2
      // Assume that T1_t is in region R1. Then after we handle(expr1), we will
      // have an entry:
      //    R1 -> a function registering insertion of "dealloc R1" after expr1
      // After handle(expr2), this entry becomes:
      //    R1 -> a function registering insertion of "dealloc R1" after expr2
      //
      // After traversing the entire kernel, this map will contain the final
      // register functions we want to execute.
      std::unordered_map<
          const TMemAlllocationInfo::Region*,
          std::function<void()>>
          region_to_register_dealloc_map_;

      // A map:
      //   expr -> the regions that this expr is accessing
      // Note that if expr is a container such as ForLoop or IfThenElse, then
      // the mapped regions will be all the regions the contained exprs are
      // accessing.
      //
      // This map only contain information of accesses that we have discovered,
      // and is updated during traversal. For example, if we have a kernel:
      //   ForLoop: // loop1
      //     T2_t = T0_r; // expr1
      //     ...
      //     T3_t = T1_r; // expr2
      // Assume T2_t is in region R2 and T3_t is in region R3. Then after
      // handle(expr1), we will have a map:
      //    expr1 -> {R2}
      //    loop1 -> {R2}
      // After handle(expr2), this map becomes:
      //    expr1 -> {R2}
      //    expr2 -> {R3}
      //    loop1 -> {R2, R3}
      std::unordered_map<
          Expr*,
          VectorOfUniqueEntries<const TMemAlllocationInfo::Region*>>
          access_map_;

      // Analyze expr to see if it has any accesses to tensor memory. If yes
      // update the access map for this expr and its container exprs.
      void updateAccessMap(Expr* expr) {
        std::unordered_set<Val*> io_vals;
        std::copy(
            expr->inputs().begin(),
            expr->inputs().end(),
            std::inserter(io_vals, io_vals.end()));
        std::copy(
            expr->outputs().begin(),
            expr->outputs().end(),
            std::inserter(io_vals, io_vals.end()));
        if (io_vals.empty()) {
          return;
        }
        for (const auto& region :
             GpuLower::current()->tmemInfo().allocation.regions) {
          for (auto tv_info : region.covered_tensors) {
            if (io_vals.count(tv_info.tensor)) {
              access_map_[expr].pushBack(&region);
              for (auto container : scope_exprs_) {
                access_map_[container].pushBack(&region);
              }
            }
            break;
          }
        }
      }

      // Update the region_to_register_dealloc_map_ to register insertion of
      // deallocation expression after expr for the regions accessed by expr.
      void maybeRegisterDeallocsAfterExpr(Expr* expr) {
        // If expr is a trivial for loop, then we don't need to move the
        // deallocation after it. This is because the trivial is not generated
        // in the final code.
        if (auto fl = dynamic_cast<ForLoop*>(expr)) {
          if (fl->isTrivial()) {
            return;
          }
        }
        // If expr is not accessing any tensor memory, then nothing to do.
        if (!access_map_.count(expr)) {
          return;
        }
        for (auto region : access_map_.at(expr)) {
          auto current_scope = scope_.empty() ? nullptr : scope_.back();
          region_to_register_dealloc_map_[region] =
              [this, expr, region, current_scope]() {
                auto first_warp = createFirstWarpITE();
                auto tcgen05_dealloc_expr = IrBuilder::create<kir::Asm>(
                    "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
                    std::vector<Val*>{},
                    std::vector<Val*>{
                        IrBuilder::create<kir::TensorIndex>(
                            region->address, expr->fusion()->zeroVal()),
                        GpuLower::current()->commonScalarMap().hoistScalar(
                            region->num_columns, for_loops_)},
                    kir::Asm::Options{/*volatile=*/true});
                first_warp->thenBody().push_back(tcgen05_dealloc_expr);
                registerInsertAfter(expr, first_warp, current_scope);
                auto block_sync = IrBuilder::create<kir::BlockSync>();
                registerInsertAfter(expr, block_sync, current_scope);
              };
        }
      }

      void dispatch(Expr* expr) final {
        updateAccessMap(expr);
        ExprMutator::dispatch(expr);
        maybeRegisterDeallocsAfterExpr(expr);
      }

     public:
      DeallocInserter(
          const std::vector<Expr*>& exprs,
          std::vector<Expr*>& exprs_with_deallocs) {
        handle(exprs);
        for (const auto& region :
             GpuLower::current()->tmemInfo().allocation.regions) {
          region_to_register_dealloc_map_.at (&region)();
        }
        exprs_with_deallocs = mutate();
      }
    } inserter(exprs, exprs_with_deallocs);
  }

  // Combine prologue and exprs_with_deallocs
  std::vector<Expr*> result;
  result.reserve(prologue.size() + exprs_with_deallocs.size());
  result.insert(result.end(), prologue.begin(), prologue.end());
  result.insert(
      result.end(), exprs_with_deallocs.begin(), exprs_with_deallocs.end());
  return result;
}

} // namespace

std::vector<Expr*> insertAllocations(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertAllocations");

  AllocationDomainSetup alloc_setup;
  alloc_setup.setup(exprs);
  GpuLower::current()->allocationInfo() =
      std::move(alloc_setup.tv_alloc_info_map);

  // If the fusion uses tensor memory, insert the following things to the
  // fusion:
  // - A tcgen05.alloc for each tensor memory region
  // - A kir::Allocate for a shared memory TensorView for each tensor memory
  //   region for storing addresses of these regions. Because tcgen05.alloc
  //   writes the address of allocated memory to the shared memory, there must
  //   be shared memory TensorViews to store these addresses. These address
  //   TensorViews are not part of the fusion math, and not handled by
  //   AllocationInserter::insert. Note that these address TensorViews are not
  //   the tensor memory TensorViews in fusion math.
  // - A tcgen05.relinquish_alloc_permit after all tcgen05.allocs
  auto result = insertTMemRegionAllocsAndDeallocs(exprs);
  // Insert kir::Allocate for each Val, including the kir::Allocate for tensor
  // memory TensorViews, in fusion math.
  return AllocationInserter::insert(result);
}

} // namespace nvfuser

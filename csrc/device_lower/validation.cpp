// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/validation.h>

#include <contiguity.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/id_model.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <scheduler/mma_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>
#include <transform_iter.h>
#include <transform_replay.h>
#include <type.h>
#include <utils.h>
#include <val_graph_visitor.h>

#include <ATen/cuda/CUDAContext.h>
#include "ir/base_nodes.h"

namespace nvfuser {

namespace {

//! Validate multiple output tensors of the same expression, i.e.,
//! siblings, have valid domains and parallel types. Since siblings
//! are placed in the same loop nest, they must be parallelized the
//! same way. Will infer and modify serial parallel types if other
//! output/s are parallelized, so that user wouldn't have to specify
//! the same parallelization 3 times. Will throw if conflicts are
//! detected, i.e. TIDx vs BIDx etc.
class ValidateSiblings : public IterVisitor {
 public:
  static void validate(Fusion* fusion) {
    ValidateSiblings validator;
    validator.traverse(fusion);
  }

 private:
  using IterVisitor::handle;

  void dispatch(Expr* expr) final {
    if (!ir_utils::isTvOp(expr) || expr->outputs().size() < 2 ||
        !ir_utils::hasUniformSiblings(expr)) {
      IterVisitor::dispatch(expr);
      return;
    }

    auto ref_output = expr->outputs().at(0)->as<TensorView>();
    auto ref_ndims = ref_output->nDims();
    const auto& ref_root = ref_output->getMaybeRootDomain();
    std::unordered_map<IterDomain*, IterDomain*> id_map;

    for (const auto sibling :
         ir_utils::filterByType<TensorView>(expr->outputs())) {
      if (ref_output == sibling) {
        continue;
      }

      NVF_ERROR(
          sibling->nDims() == ref_ndims,
          "Mismatched dimensionality detected. Expr: ",
          expr->toString(),
          "Ref output: ",
          ref_output->toString(),
          ". Sibling: ",
          sibling->toString());

      for (const auto i : arange(ref_ndims)) {
        validateParallelTypes(ref_output->axis(i), sibling->axis(i));
      }

      for (const auto i : arange(ref_root.size())) {
        id_map[ref_root[i]] = sibling->getMaybeRootDomain().at(i);
      }

      auto replay =
          BestEffortReplay(
              sibling->getLoopDomain(), ref_output->getLoopDomain(), id_map)
              .getIterDomainEquivalence();

      for (const auto i : arange(ref_ndims)) {
        NVF_ERROR(
            replay.strictAreMapped(ref_output->axis(i), sibling->axis(i)),
            "Matching sibling ID not found. Expr: ",
            expr->toString(),
            "Ref ID: ",
            ref_output->axis(i)->toString(),
            "Sibling ID: ",
            sibling->axis(i)->toString());
      }
    }
  }

  // Parallelize id1 and id0 consistently if one is serial and the other isn't
  void validateParallelTypes(IterDomain* id0, IterDomain* id1) {
    const auto ptype0 = id0->getParallelType();
    const auto ptype1 = id1->getParallelType();

    if (ptype0 == ParallelType::Vectorize ||
        ptype1 == ParallelType::Vectorize) {
      auto other_type = ptype0 == ParallelType::Vectorize ? ptype1 : ptype0;
      NVF_ERROR(
          other_type == ParallelType::Vectorize ||
              (!isParallelTypeThreadDim(other_type) &&
               !isParallelTypeBlockDim(other_type)),
          "Vectorize type was parallelized inconsistently in. ",
          "Detected during promoting parallel types.");
      return;
    }

    if (ptype0 != ptype1) {
      NVF_CHECK(
          ptype0 == ParallelType::Serial || ptype1 == ParallelType::Serial,
          "Error promoting parallel types: ",
          ptype0,
          " and ",
          ptype1);
      if (ptype0 == ParallelType::Serial) {
        id0->parallelize(ptype1);
      }
      if (ptype1 == ParallelType::Serial) {
        id1->parallelize(ptype0);
      }
    }
  }
};

// Make sure all IterDomains are only used for a unique
// TensorView. Several mappings from IterDomains are
// created during lowering, which relies on the unique usage of
// IterDomains.
void validateIterDomainUsage(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateIterDomainUse");
  FusionGuard fg(fusion);

  auto used_vals = fusion->usedMathVals();
  std::unordered_map<IterDomain*, TensorView*> domain_use_map;

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->domain()->allIDs()) {
      auto it = domain_use_map.find(id);
      NVF_ERROR(
          it == domain_use_map.end(),
          "Multiple use of ",
          id,
          " detected.",
          " Used in both TV",
          tv->name(),
          " and TV",
          it->second->name());
      domain_use_map.insert({id, tv});
    }
  }
}

void validateCpAsyncBulk(const std::vector<TensorView*>& tvs) {
  for (auto tv : tvs) {
    bool is_cp_async_bulk = ir_utils::isCpAsyncBulk(tv->definition());
    for (auto id : tv->getLoopDomain()) {
      if (id->getParallelType() == ParallelType::Bulk) {
        NVF_ERROR(
            is_cp_async_bulk,
            "ParallelType::Bulk is only supported for cp.async.bulk.");
      }
    }
    if (!is_cp_async_bulk) {
      continue;
    }
    std::unordered_set<ParallelType> valid_parallel_types{
        ParallelType::DIDx,
        ParallelType::BIDz,
        ParallelType::BIDy,
        ParallelType::BIDx,
        ParallelType::TIDz,
        ParallelType::TIDy,
        ParallelType::TIDx,
        ParallelType::Unroll,
        ParallelType::Unswitch,
        ParallelType::Serial};
    std::unordered_set<IterType> valid_iter_types{
        IterType::Iteration, IterType::Broadcast};
    for (auto id : tv->getLoopDomain()) {
      if (id->getParallelType() == ParallelType::Bulk) {
        NVF_ERROR(
            id->getIterType() == IterType::Iteration,
            "ParallelType::Bulk is only supported for IterType::Iteration.");
        continue;
      }
      NVF_ERROR(
          valid_parallel_types.find(id->getParallelType()) !=
              valid_parallel_types.end(),
          "Invalid parallel type for cp.async.bulk: ",
          id->getParallelType());
      NVF_ERROR(
          valid_iter_types.find(id->getIterType()) != valid_iter_types.end(),
          "Invalid iter type for cp.async.bulk: ",
          id->getIterType());
    }
  }
}

// For each expressions, update the frontier based on merge and
// split operations. Removes non-contiguous merges from frontier.
void traverseFrontierWithContiguityCheck(
    std::deque<IterDomain*>& frontier,
    Expr* expr) {
  // expr is skipped if any of the inputs is missing.
  if (auto merge = dynamic_cast<Merge*>(expr)) {
    // Check if this merge is logically contiguous merge, that is,
    // both of the two inputs are adjacent to each other
    auto outer_it = std::ranges::find(frontier, merge->outer());
    if (outer_it == frontier.end()) {
      return;
    }
    auto inner_it = std::ranges::find(frontier, merge->inner());
    if (inner_it == frontier.end()) {
      return;
    }
    auto outer_pos = std::distance(frontier.begin(), outer_it);
    auto inner_pos = std::distance(frontier.begin(), inner_it);

    bool is_contig = outer_pos + 1 == inner_pos;
    frontier.erase(inner_it);

    // If it's contig, we can continue the analysis by proceeding to
    // the output. If not, no further analysis is possible, so the
    // two inputs are just removed from the frontier list
    if (is_contig) {
      frontier[outer_pos] = merge->out();
    } else {
      frontier.erase(frontier.begin() + outer_pos);
    }
  } else if (auto split = dynamic_cast<Split*>(expr)) {
    auto in_it = std::ranges::find(frontier, split->in());
    if (in_it == frontier.end()) {
      return;
    }
    frontier.insert(in_it + 1, split->inner());
    *in_it = split->outer();
  } else {
    NVF_ERROR(expr != nullptr);
    NVF_THROW("Unexpected expression: ", expr->toString());
  }
}

// Check if maybe_innermost_id is derived from base_id and corresponds to the
// innermost subregion of base_id. The split/merge exprs between
// base_id and id must not include any ID that is not produced from
// base_id.
bool isInnermost(IterDomain* base_id, IterDomain* maybe_innermost_id) {
  auto exprs =
      DependencyCheck::getAllExprsBetween({base_id}, {maybe_innermost_id});

  std::deque<IterDomain*> frontier;
  frontier.push_back(base_id);

  for (auto expr : exprs) {
    traverseFrontierWithContiguityCheck(frontier, expr);
  }

  // Once the traversal is done, if the target id located at the
  // rightmost position of the frontier list, it is guaranteed to
  // correspond to the innermost subregion of the base ID.
  return !frontier.empty() && frontier.back() == maybe_innermost_id;
}

// Validate the swizzling pattern:
// We support a very restricted pattern from 2D logical to 5D allocation
// Expected pattern:
// m, k -> m, k/4, 4 (split k by 4)
// m, k/4, 4 -> m/128, 128, k/4, 4 (split m by 128)
// m/128, 128, k/4, 4 -> m/128, 4(m_o), 32(m_i), k/4, 4 (split 128 by 32)
// Then reorder to: m/128, k/4, 32(m_i), 4(m_o), 4(k)
void isValidBlockScaleSwizzle(TensorView* block_scale) {
  auto logical_domain =
      TensorDomain::noReductions(block_scale->getLogicalDomain());
  auto allocation_domain =
      TensorDomain::noReductions(block_scale->getAllocationDomain());

  // check that size of logical domain is 2 and allocation domain is 5
  NVF_ERROR(
      logical_domain.size() == 2 && allocation_domain.size() == 5,
      "Block scale swizzle must have 2D logical domain and 5D allocation "
      "domain. Found: ",
      logical_domain.size(),
      "D logical and ",
      allocation_domain.size(),
      "D allocation for TensorView: ",
      block_scale->toString());

  // keep count of splits
  int num_splits = 0;

  // keeps track of the split
  // M -> M/128, 128
  Split* middle_split = nullptr;

  // A lambda to check the transforms from logical to allocation domain
  // Each transform must be a split, and there can be only 3 splits.
  auto check_transform = [block_scale,
                          &logical_domain,
                          &num_splits,
                          &middle_split](Expr* expr) {
    if (auto split_expr = dynamic_cast<Split*>(expr)) {
      // Can have a max of 3 splits - checked later
      num_splits++;

      // If expr and it's input is logical_domain back()
      // the inner split output should have an extent of 4.
      // Check K -> K/4, 4
      if (split_expr->in() == logical_domain.back()) {
        NVF_ERROR(
            split_expr->inner()->extent()->isConstInt() &&
                split_expr->inner()->extent()->evaluate().as<int64_t>() == 4,
            "The innermost split in block scale swizzle must have an extent of "
            "4. "
            "Found extent: ",
            split_expr->inner()->extent()->toString(),
            " in expr: ",
            expr->toString(),
            " for TensorView: ",
            block_scale->toString());
      } else if (split_expr->in() == logical_domain.front()) {
        // Check M -> M/128, 128
        NVF_ERROR(
            split_expr->inner()->extent()->isConstInt() &&
                split_expr->inner()->extent()->evaluate().as<int64_t>() == 128,
            "The outermost split in block scale swizzle must have an extent of "
            "128. "
            "Found extent: ",
            split_expr->inner()->extent()->toString(),
            " in expr: ",
            expr->toString(),
            " for TensorView: ",
            block_scale->toString());

        // Cache the M -> M/128, 128 split
        middle_split = split_expr;
      } else {
        // Check that the input to this split is the inner output of
        // middle_split. As we should have 128 -> 4, 32
        NVF_ERROR(
            middle_split && split_expr->in() == middle_split->inner(),
            "The third split in block scale swizzle must split the inner "
            "output "
            "(extent 128) of the second split. Expected input to be the inner "
            "output "
            "of the M/128, 128 split. Found expr: ",
            split_expr->toString(),
            " for TensorView: ",
            block_scale->toString());

        NVF_ERROR(
            split_expr->inner()->extent()->isConstInt() &&
                split_expr->inner()->extent()->evaluate().as<int64_t>() == 32,
            "The third split in block scale swizzle (128 -> 4, 32) must have "
            "an "
            "inner extent of 32. "
            "Found extent: ",
            split_expr->inner()->extent()->toString(),
            " in expr: ",
            split_expr->toString(),
            " for TensorView: ",
            block_scale->toString());
      }
    } else {
      NVF_THROW(
          "Logical to allocation domain transforms for block scale swizzle "
          "can only contain split operations");
    }
  };

  // Get all exprs between logical and allocation domain
  auto transform_exprs = DependencyCheck::getAllExprsBetween(
      {logical_domain.begin(), logical_domain.end()},
      {allocation_domain.begin(), allocation_domain.end()});

  std::vector<IterDomain*> ids_to_transform = logical_domain;

  // Transform the logical domain to the allocation domain
  // without the permutation.
  scheduler_utils::applyTransforms(
      ids_to_transform, transform_exprs, check_transform);

  // Check that there are exactly 3 splits
  NVF_ERROR_EQ(
      num_splits,
      3,
      "Block scale swizzle must have exactly 3 splits. Found ",
      num_splits,
      " splits in TensorView: ",
      block_scale->toString());

  // Get the permutation.
  auto permutation =
      ir_utils::computePermutation(ids_to_transform, allocation_domain);

  // m/128, 4(m_o), 32(m_i), k/4, 4(k)
  // -> m/128, k/4, 32(m_i), 4(m_o), 4(k)
  // check that permutation has a value and it is 0, 3, 2, 1, 4
  NVF_ERROR(
      permutation.has_value() &&
          permutation.value() == std::vector<int64_t>({0, 3, 2, 1, 4}),
      "Block scale swizzle permutation is invalid for TensorView: ",
      block_scale->toString());
}

// Expr-specific validaion
//
// TODO: Move individual validations to here, e.g.,
// validateCpAsyncBulk can be moved here
class ExprValidator : public OptOutDispatch {
 public:
  static void validate(Fusion* fusion) {
    ExprValidator validator(fusion);
  }

 private:
  ExprValidator(Fusion* fusion) {
    for (auto expr : fusion->exprs()) {
      dispatch(expr);
    }
  }

  // The lowering of ArgsortOp depends on specific scheduling
  // assumptions. This tight coupling isn't ideal, but it's not a
  // problem for now.
  void handle(ArgsortOp* aop) final {
    validateGroupedOp(
        ir_utils::getTvInput(aop), ir_utils::getTvOutput(aop), aop->dim());
  }

  void handle(ScanOp* sop) final {
    validateGroupedOp(
        ir_utils::getTvInput(sop), ir_utils::getTvOutput(sop), sop->dim());
  }

  void handle(TopKOp* top) final {
    validateGroupedOp(
        ir_utils::getTvInput(top), ir_utils::getTvOutput(top), top->dim());
  }

  static void validateGroupedOp(
      TensorView* inp_tv,
      TensorView* out_tv,
      int64_t logical_dim_to_group) {
    IterDomain* grouped_id = nullptr;
    for (const auto& loop_id : out_tv->getLoopDomain()) {
      if (loop_id->getParallelType() == ParallelType::Group) {
        NVF_ERROR(grouped_id == nullptr, "Multiple IDs found to be grouped");
        grouped_id = loop_id;
      }
    }

    // Even if grouping is not used, CudaKernelGenerator assumes these
    // properties for simplicity

    // Both input and output must be Local tensors so that the op can be
    // executed without explicit predication. This makes it simpler to
    // generate code for grouped calls
    NVF_ERROR_EQ(inp_tv->getMemoryType(), MemoryType::Local);
    NVF_ERROR_EQ(out_tv->getMemoryType(), MemoryType::Local);

    // If not grouped, no more validation to do
    if (grouped_id == nullptr) {
      return;
    }

    // The input will be initialized for this op. To avoid any
    // potential conflict of initialization, require the input to be
    // exclusively used by the grouped op.
    NVF_ERROR_EQ(
        inp_tv->uses().size(), 1, "Invalid tensor uses: ", inp_tv->toString());

    // If it's grouped, it must correspond to the innermost subregion
    // of the logical ID to group
    IterDomain* logical_id_to_group =
        out_tv->getLogicalDomain().at(logical_dim_to_group);
    NVF_ERROR(
        isInnermost(logical_id_to_group, grouped_id),
        "Invalid ID to group: ",
        grouped_id->toString(),
        " of ",
        out_tv->toString());

    // The output is allocated on registers, so the allocation
    // domain should be just the same as the loop domain. The
    // grouped ID must have the unit stride, which means all of the
    // inner IDs must not contribute to the allocation
    validateUnitStride(out_tv, out_tv->getLoopDomain(), grouped_id);

    // All of the inputs per thread must be provided to the device
    // function as a contiguous chunk of memory where each element
    // has a unit stride within its allocated buffer. More
    // concretely, for example, in the case of argsort, the input would be
    // passed to the device function as follows:
    //
    //  // Each thread computes 8 argsort elements
    //  float T1[8];
    //  ...
    //  blockArgsort(..., T1, ...);
    //
    // Here, the input is required to have a loop ID that is exactly
    // mapped with the grouped loop ID of the output, and that
    // producer loop ID must be indeed allocated, i.e., not parallelized.
    // Furthermore, like the output, none of the inner loop
    // IDs is allowed to contribute to the allocation of the input so that the
    // grouped ID has a unit stride.
    //
    // The requirement of the input being transformed exactly in the
    // same as the output is a sufficient but not required
    // condition. It could be relaxed if necessary.
    const auto& c2p_replay = BestEffortReplay::replayPasC(
        inp_tv, out_tv, -1, PairwiseLogicalDomainMap(inp_tv, out_tv));
    auto producer_grouped_id_it = c2p_replay.getReplay().find(grouped_id);
    NVF_ERROR(
        producer_grouped_id_it != c2p_replay.getReplay().end(),
        "No corresponding producer ID for the grouped consumer ID found: ",
        grouped_id->toString());
    auto producer_grouped_id = producer_grouped_id_it->second;

    NVF_ERROR(
        std::ranges::find(inp_tv->getLoopDomain(), producer_grouped_id) !=
            inp_tv->getLoopDomain().end(),
        "Corresponding grouped producer ID is not a loop ID: ",
        producer_grouped_id->toString(),
        " of ",
        inp_tv->toString());

    NVF_ERROR(
        ir_utils::mayRequireAllocation(inp_tv, producer_grouped_id),
        "The corresponding producer loop ID for grouping must be actualy "
        "allocated without parallelization");

    // Make sure all inner loop IDs should not contribute to the
    // allocation
    validateUnitStride(inp_tv, inp_tv->getLoopDomain(), producer_grouped_id);
  }

  static void validateUnitStride(
      TensorView* tv,
      const std::vector<IterDomain*>& alloc_domain,
      IterDomain* id) {
    auto it = std::ranges::find(alloc_domain, id);
    NVF_ERROR(
        it != alloc_domain.end(),
        "ID, ",
        id->toString(),
        " not found in ",
        toDelimitedString(alloc_domain));
    ++it;
    for (; it != alloc_domain.end(); ++it) {
      NVF_ERROR(
          !ir_utils::mayRequireAllocation(tv, *it),
          "Not guaranteed to have a unit stride: ",
          id->toString(),
          " of ",
          tv->toString(),
          " due to ",
          (*it)->toString());
    }
  }

  // The block quantization operator is implemented via a runtime function.
  // This runtime function expects the inputs to be in local memory. The
  // quantized output will also be in local memory, but the block scaling
  // factors will be written out to global memory. The device runtime currently
  // works on 2/4 elements per thread (also 8 for bf16/fp16). The runtime
  // function is based on a parallelization scheme that expects TIDx and BIDx,
  // and optionally TIDy and BIDy. 3D parallelization is not supported. Based
  // on the above, we have the following basic validation checks:

  // Input is in local memory.
  // Block scaling factor is in global memory and
  // quantized output is in local memory.
  // The Group ID has an extent of 2/4/8 depending on the data
  // type.
  // There are no TIDz/BIDz IDs. We don't support 3D parallelization here.

  // For this op, the indices for block scaling factor is partially computed
  // in nvfuser's index computation. It is done do by linearizing the logical
  // index of the quantized outputs and the extents of the allocation domain
  // of the quantized output. This index is passed to the runtime function,
  // where is it divided by 16 (blocksize) to compute the output index for block
  // scaling factor. Because of this indexing scheme we have to put the
  // following restrictions. Our aim for the following checks is to ensure that
  // the group ID is contiguous and has unit stride, and then after the group
  // ID, we have TIDx, such that (G -- extent of GID) * ThreadIdx.x + GID is
  // contiguous. We have these restrictions because 4 threads (x) will be
  // working on contiguous data in the input (actually #threads *
  // #elements_per_thread == blocksize(16)) - so we conservatively want all
  // threads(x) to be accessing contiguous data.

  // We do so by checking that the group ID has unit stride.
  // It should be derived from the innermost logical IDs via contiguous merges
  // only.
  // Next, we check that TIDx is the next inner-most ID, and if there is
  // any other ID between TIDx and the group ID, then it must have an extent
  // of 1.
  // TIDx must also be derived from contiguous merges of the logical IDs.
  // Any loop ID that is not TIDx, TIDy, BIDx, BIDy, or Group
  // has an extent of 1. (we don't want the runtime kernel to be called multiple
  // times by a thread).
  void handle(BlockQuantizationOp* bqop) final {
    auto inp_tv = bqop->input(0)->as<TensorView>();
    auto quantized_output = bqop->quantizedOutput()->as<TensorView>();
    auto block_scaling_factor = bqop->blockScales()->as<TensorView>();
    auto output_dtype = quantized_output->dtype();
    bool is_mxfp8_output = output_dtype == DataType::Float8_e4m3fn;

    NVF_ERROR_EQ(
        inp_tv->getMemoryType(),
        MemoryType::Local,
        "Input must be a local memory tensor. Found: ",
        inp_tv->getMemoryType());

    NVF_ERROR_EQ(
        quantized_output->getMemoryType(),
        MemoryType::Local,
        "Quantized output must be a local memory tensor. Found: ",
        quantized_output->getMemoryType());

    NVF_ERROR_EQ(
        block_scaling_factor->getMemoryType(),
        MemoryType::Global,
        "Block scaling factor must be a global memory tensor. Found: ",
        block_scaling_factor->getMemoryType());

    if (is_mxfp8_output) {
      NVF_ERROR(
          !bqop->hasGlobalScale(),
          "Global scale is not supported when quantizing to Float8_e4m3fn.");

      NVF_ERROR(
          !block_scaling_factor->hasAllocation(),
          "Block scaling factor must not have an allocation domain when "
          "quantizing to Float8_e4m3fn.");
    }

    if (bqop->hasGlobalScale()) {
      auto global_scale = bqop->globalScale()->as<TensorView>();

      NVF_ERROR_EQ(
          global_scale->getMemoryType(),
          MemoryType::Global,
          "Global scaling factor must be a global memory tensor. Found: ",
          global_scale->getMemoryType());

      NVF_ERROR_EQ(
          global_scale->dtype(),
          DataType::Float,
          "Global scaling factor must be of type float. Found: ",
          global_scale->dtype());
    }

    // Outputs have the same allocation domain
    // as the logical domain - no allocation domain.
    NVF_ERROR(
        !quantized_output->hasAllocation(),
        "Quantized output must not have an allocation domain.");

    // When output scales is swizzled we will need to allow these checks
    // to be relaxed. We will need to ensure that the swizzling
    // allocation allowed is a fixed pattern:
    // 2D logical and 5D allocation domain.
    // https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
    if (block_scaling_factor->hasAllocation()) {
      isValidBlockScaleSwizzle(block_scaling_factor);
      NVF_ERROR_EQ(
          bqop->isSwizzledScales(),
          true,
          "Block scaling factor with allocation domain requires swizzled "
          "scales.");
    }

    NVF_ERROR(
        std::all_of(
            block_scaling_factor->getContiguity().begin(),
            block_scaling_factor->getContiguity().end(),
            [](std::optional<bool> c) { return c.value_or(true); }),
        "Block scaling factor not contiguous");

    IterDomain* grouped_id = nullptr;
    IterDomain* thread_x = nullptr;
    IterDomain* block_x = nullptr;
    IterDomain* thread_z = nullptr;
    IterDomain* block_z = nullptr;

    for (const auto& loop_id : quantized_output->getLoopDomain()) {
      if (loop_id->getParallelType() == ParallelType::Group) {
        grouped_id = loop_id;
      } else if (loop_id->getParallelType() == ParallelType::TIDx) {
        thread_x = loop_id;
      } else if (loop_id->getParallelType() == ParallelType::BIDx) {
        block_x = loop_id;
      } else if (loop_id->getParallelType() == ParallelType::TIDz) {
        thread_z = loop_id;
      } else if (loop_id->getParallelType() == ParallelType::BIDz) {
        block_z = loop_id;
      } else if (
          loop_id->getParallelType() == ParallelType::Serial ||
          loop_id->getParallelType() == ParallelType::Unswitch ||
          loop_id->getParallelType() == ParallelType::Unroll) {
        // Check this is ID has a constant extent and is 1
        NVF_ERROR(
            loop_id->extent()->isConstInt(),
            "Expected constant extent for Serial/Unswitch/Unroll ID in "
            "BlockQuantizationOp");
        NVF_ERROR_EQ(
            loop_id->extent()->evaluate().as<int64_t>(),
            1,
            "Expected non-TID/BID/Group ID to have extent of 1 for "
            "BlockQuantizationOp: ",
            bqop->toString());
      }
    }

    NVF_ERROR(
        grouped_id != nullptr || is_mxfp8_output,
        "One of the output IDs must be grouped for "
        "BlockQuantizationOp: ",
        bqop->toString());

    NVF_ERROR(
        thread_x != nullptr && block_x != nullptr,
        "Need to have both TIDx and BIDx when using BlockQuantizationOp: ",
        bqop->toString());

    NVF_ERROR(
        !thread_z && !block_z,
        "Parallelization along z axis is not supported for "
        "BlockQuantizationOp: ",
        bqop->toString());

    auto inner_extent =
        grouped_id ? grouped_id->extent()->evaluate().as<int64_t>() : 1;
    auto input_dtype = inp_tv->dtype();

    // Check the extents of group id based on inputdata type
    // if group id is present
    NVF_ERROR(
        (!grouped_id ||
         ((inner_extent == 4 || inner_extent == 2) &&
          input_dtype == DataType::Float) ||
         ((inner_extent == 8 || inner_extent == 4 || inner_extent == 2) &&
          (input_dtype == DataType::BFloat16 ||
           input_dtype == DataType::Half))),
        "The group dimension must be  2/4 (FP32) or 2/4/8 "
        "(BF16). Found: ",
        inner_extent,
        ". Expr: ",
        bqop->toString());

    //                   M    K
    //                 │    │
    //                 ▼    ▼
    //              ┌────────────┐
    //              │   merge    │
    //              └─────┬──────┘
    //                    │
    //                    ▼
    //                   M*K
    //               ┌──────────┐
    //               │  split   ┼──┐
    //               └─┬────────┘  │
    //                 ▼           ▼
    //           (M*K)/4          4(G)
    //           ┌────────┐
    //           │ split  ┼────┐
    //           └─┬──────┘    │
    //             ▼           ▼
    //         (M*K)/4        1(U)
    //     ┌─────────┐
    //     │  split  │
    //   ┌─┼         ┼───┐
    //   │ └─────────┘   │
    //   ▼               ▼
    // (M*K)/4/128      128(Tx)

    // Next we check the following scheduling requirements for
    // BlockQuantizationOp - the above figure is an example of a valid schedule.
    // 1. The Group ID must be derived from the innermost logical IDs
    // 2. TIDx must follow the Group ID in the schedule -- that is when derived
    // from the logical domain, group ID must be inner-most, the next
    // "inner-most" should be TIDx (unless there is an ID with a unit trip
    // count)
    // 3. All merges involved from logical domains to group and thread ID must
    // combine contiguous IDs

    auto transform_exprs = DependencyCheck::getAllExprsBetween(
        {quantized_output->getLogicalDomain().begin(),
         quantized_output->getLogicalDomain().end()},
        {quantized_output->getLoopDomain().begin(),
         quantized_output->getLoopDomain().end()});

    std::vector<IterDomain*> ids_to_transform =
        quantized_output->getLogicalDomain();

    std::deque<IterDomain*> frontier(
        quantized_output->getLogicalDomain().begin(),
        quantized_output->getLogicalDomain().end());

    // This will get the xforms from logical to loop and apply them on the
    // logical domain. We will get a loop domain minus the reordering.
    // This pass also removes all IDs from frontier that were derived using
    // non-contiguous merges.
    scheduler_utils::applyTransforms(
        ids_to_transform, transform_exprs, [&frontier](Expr* expr) {
          traverseFrontierWithContiguityCheck(frontier, expr);
        });

    // Check that TIDx is multiple of 32
    // TIDx is innermost since there is no group IDs.
    if (is_mxfp8_output && !grouped_id) {
      auto tidx_extent = thread_x->extent();
      inp_tv->fusion()->print();
      NVF_ERROR(
          tidx_extent->isConstInt() &&
              (tidx_extent->evaluate().as<int64_t>() % 32 == 0),
          "When quantizing to Float8_e4m3fn without grouping, TIDx extent must "
          "be a "
          "multiple of 32. Found extent: ",
          tidx_extent->toString(),
          ". Expr: ",
          bqop->toString());

      NVF_ERROR(
          ids_to_transform.back() == thread_x,
          "When quantizing to Float8_e4m3fn without grouping, TIDx must be the "
          "innermost ID. Expr: ",
          bqop->toString());

      return;
    }

    // The grouped ID must correspond to the innermost loop-like domain
    NVF_ERROR(
        ids_to_transform.back() == grouped_id,
        "The grouped ID must correspond to the innermost of all splits "
        "from logical domains to loop domains for BlockQuantizationOp. "
        "TV: ",
        quantized_output->toString());

    // Iterate from the back to find TIDx, skipping group_id (last element)
    // Ensure all IDs between group_id and TIDx have extent 1
    bool found_tidx = false;
    for (auto it = ids_to_transform.rbegin() + 1; it != ids_to_transform.rend();
         ++it) {
      if (*it == thread_x) {
        found_tidx = true;
        break;
      }
      // All non-TIDx IDs between Group ID and TIDx must have extent of 1
      NVF_ERROR(
          (*it)->extent()->isConstInt() &&
              (*it)->extent()->evaluate().as<int64_t>() == 1,
          "Expected IDs between Group ID and TIDx to have extent of 1 for "
          "BlockQuantizationOp: ",
          quantized_output->toString());
    }

    NVF_ERROR(
        found_tidx,
        "TIDx must follow the Group ID in the schedule for "
        "BlockQuantizationOp: ",
        quantized_output->toString());

    // Check if grouped_id in frontier
    auto grouped_it = std::ranges::find(frontier, grouped_id);
    NVF_ERROR(
        grouped_it != frontier.end(),
        "All merge operations deriving the grouped ID must combine "
        "contiguous IDs from the logical domain for BlockQuantizationOp: ",
        quantized_output->toString());
    // Do the same for thread_x
    auto threadx_it =
        std::ranges::find(frontier.begin(), frontier.end(), thread_x);
    NVF_ERROR(
        threadx_it != frontier.end(),
        "All merge operations deriving the TIDx ID must combine "
        "contiguous IDs from the logical domain for BlockQuantizationOp: ",
        quantized_output->toString());
  }
};

} // namespace

void validateIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateIr");

  FusionGuard fg(fusion);

  fusion->validateInputs();

  // Validate Parallelization
  ValidateSiblings::validate(fusion);

  validateIterDomainUsage(fusion);

  auto dynamic_tvs = ir_utils::getTVsWithDynamicTransform(fusion);
  NVF_ERROR(
      dynamic_tvs.empty(),
      "Tensor with dynamic transform must be concretized before lowering: ",
      toDelimitedString(dynamic_tvs.begin(), dynamic_tvs.end()));

  auto all_tvs = fusion->allTvs();
  validateCpAsyncBulk(all_tvs);

  ExprValidator::validate(fusion);
}

namespace {

class VectorizeValidator : public OptInDispatch {
 private:
  // Initially, vectorized_id is the IterDomain with Vectorize ParallelType
  // After processing all merge and split operations,
  // vectorized_id is the corresponding allocation domain
  VectorizeValidator(IterDomain* vectorized_id)
      : vectorized_id_(vectorized_id) {}

  using OptInDispatch::handle;

  void handle(Split* s) final {
    if (s->outer() == vectorized_id_) {
      is_valid = false;
    } else if (s->inner() == vectorized_id_) {
      vectorized_id_ = s->in();
    }
    domains_.insert(s->in());
  }

  void handle(Merge* m) final {
    if (m->out() == vectorized_id_) {
      if (m->inner()->isBroadcast() && !m->outer()->isBroadcast()) {
        vectorized_id_ = m->outer();
      } else {
        vectorized_id_ = m->inner();
      }
    }
    domains_.insert(m->outer());
    domains_.insert(m->inner());
  }

  void handle(Resize* r) final {
    if (r->out() == vectorized_id_) {
      vectorized_id_ = r->in();
    }
    domains_.insert(r->in());
  }

  void handle(Swizzle* swizzle) final {
    if (swizzle->outX() == vectorized_id_ || swizzle->inX() == vectorized_id_ ||
        swizzle->outY() == vectorized_id_ || swizzle->inY() == vectorized_id_) {
      // Do not (yet) allow vectorization across any swizzled id.
      is_valid = false;
    }
  }

  void handle(Swizzle2D* swizzle) final {
    if (swizzle->outX() == vectorized_id_ || swizzle->inX() == vectorized_id_ ||
        swizzle->outY() == vectorized_id_ || swizzle->inY() == vectorized_id_) {
      // Do not (yet) allow vectorization across any swizzled id.
      is_valid = false;
    }
  }

  // Given the vectorized loop ID in a tensor, find its innermost
  // ancestors in the allocation domain. Broadcast IDs are ignored.
  // All dependent allocation IDs are also returned.
  static std::pair<IterDomain*, std::unordered_set<IterDomain*>>
  getDependentAllocIDs(IterDomain* v_id, TensorView* tv) {
    auto replay_exprs = DependencyCheck::getAllExprsBetween(
        {tv->getMaybeAllocationDomain().begin(),
         tv->getMaybeAllocationDomain().end()},
        {v_id});

    VectorizeValidator validator(v_id);

    for (auto expr_it = replay_exprs.rbegin(); expr_it != replay_exprs.rend();
         ++expr_it) {
      auto expr = *expr_it;
      validator.dispatch(expr);
    }

    NVF_CHECK(
        validator.is_valid,
        "Invalid vectorized pattern found, vectorization iter domains must be "
        "descendants of inner-most dimension.",
        "Issue found in, ",
        tv,
        "\n");

    std::unordered_set<IterDomain*> dep_alloc_ids;
    for (auto alloc : tv->getMaybeAllocationDomain()) {
      if (validator.domains_.find(alloc) != validator.domains_.end()) {
        dep_alloc_ids.emplace(alloc);
      }
    }

    return {validator.vectorized_id_, dep_alloc_ids};
  }

  // Given the vectorized loop ID in a tensor, find its innermost
  // ancestors in the allocation domain. Broadcast IDs are ignored.
  // All dependent allocation IDs are also returned.
  //
  // This should work return almost the same results as
  // getDependentAllocIDs, except when loop IDs are not fully
  // derived from logical IDs. The above version does not work
  // correctly for such a case, whereas this version addresses the
  // limitation by using ValGraphBFS.
  static std::pair<IterDomain*, std::unordered_set<IterDomain*>>
  getDependentAllocIDsIdModel(
      IterDomain* v_id,
      TensorView* tv,
      Expr* load_store) {
    NVF_ERROR(GpuLower::hasCurrent());
    NVF_ERROR(GpuLower::current()->info().hasIdModel());

    const auto& id_model = GpuLower::current()->info().idModel();
    const auto& graph = id_model.idGraph(IdMappingMode::EXACT);

    // Traverse from the complete set of loop IDs to the allocation
    // domain of this tensor. Note that the allocation domain may
    // include unused IDs such as broadcast IDs. They may not be
    // reachable, so the require_all_to_visited needs to be
    // false. Here, only the innermost allocation ID needs to be
    // reachable, which is asserted at the end of the function.
    //
    // Note that previously this traversal was from the allocation
    // domain to v_id only. It does not work when the allocation
    // domain has a broadcast ID that is promoted to a concrete ID
    // and then is used to generate v_id. See
    // LoopDomainSchedulingTest.VecValidationRepro for a concrete
    // case.
    //
    // Although actual indexing traversal starts from promoted loop
    // IDs on the AlmostExact graph, the loop IDs of the consumer
    // tensor is used here without promotion on the Exact graph. This
    // was changed to avoid the error as reported in issue #5377.
    const auto loop_domain = ir_utils::getTvOutput(load_store)->getLoopDomain();
    auto expr_path = ValGraphBFS::getExprGroupsBetween(
                         graph,
                         graph.toGroups(loop_domain),
                         graph.toGroups(tv->getMaybeAllocationDomain()),
                         /*require_all_to_visited=*/false)
                         .first;

    ValGroup cur_group = graph.toGroup(v_id);
    std::unordered_set<ValGroup> visited_ids;
    visited_ids.insert(cur_group);

    for (const auto& [expr_g, dir] : expr_path) {
      Expr* expr = expr_g->front();
      NVF_ERROR(
          expr->isA<Merge>() || expr->isA<Split>() || expr->isA<Resize>() ||
              expr->isA<Swizzle>() || expr->isA<Swizzle2D>(),
          "Unexpected expr: ",
          expr->toString());

      const auto& inputs = dir == Direction::Forward
          ? graph.inputGroups(expr_g)
          : graph.outputGroups(expr_g);
      const auto& outputs = dir == Direction::Forward
          ? graph.outputGroups(expr_g)
          : graph.inputGroups(expr_g);

      if (expr->isOneOf<Swizzle, Swizzle2D>()) {
        // Not supported.
        // TODO: Checking the outputs too since that is what
        // VectorizeValidator::handle(Swizzle*) and
        // VectorizeValidator::handle(Swizzle2D*) do, but unclear
        // why.
        if (std::find(inputs.begin(), inputs.end(), cur_group) !=
                inputs.end() ||
            std::find(outputs.begin(), outputs.end(), cur_group) !=
                outputs.end()) {
          cur_group.reset();
          break;
        }
      }

      if (std::find(inputs.begin(), inputs.end(), cur_group) == inputs.end()) {
        continue;
      }

      visited_ids.insert(outputs.begin(), outputs.end());

      if (expr->isA<Resize>()) {
        // No validatiton is done at this moment
        cur_group = outputs[0];
      } else if (inputs.size() == 2) {
        NVF_ERROR(outputs.size() == 1);
        if (cur_group == inputs[1]) {
          cur_group = outputs[0];
        } else if (cur_group == inputs[0]) {
          cur_group.reset();
          break;
        }
      } else {
        NVF_ERROR(inputs.size() == 1);
        NVF_ERROR(outputs.size() == 2);
        if (outputs[1]->front()->as<IterDomain>()->isBroadcast()) {
          NVF_ERROR(!outputs[0]->front()->as<IterDomain>()->isBroadcast());
          cur_group = outputs[0];
        } else {
          cur_group = outputs[1];
        }
      }
    }

    NVF_ERROR(
        cur_group.get() != nullptr,
        "Valid corresponding allocation ID not found. ",
        tv->toString(),
        ", vec ID: ",
        v_id->toString());

    IterDomain* innermost_alloc_id = nullptr;
    std::unordered_set<IterDomain*> dep_alloc_ids;
    for (auto alloc : tv->getMaybeAllocationDomain()) {
      const auto& alloc_group = graph.toGroup(alloc);
      if (visited_ids.find(alloc_group) != visited_ids.end()) {
        dep_alloc_ids.emplace(alloc);
      }
      if (cur_group == alloc_group) {
        innermost_alloc_id = alloc;
      }
    }

    if (innermost_alloc_id == nullptr) {
      // Failed to find the innermost allocation ID
      std::stringstream ss;
      ss << "Failed to find a corresponding innermost allocation ID of "
         << tv->toString() << " with vectorized ID of " << v_id->toString()
         << "\n"
         << "Vectorized load-store: " << load_store->toString()
         << "Allocation domain: "
         << toDelimitedString(tv->getMaybeAllocationDomain()) << "\n"
         << "Loop domain: " << toDelimitedString(loop_domain) << "\n"
         << "Current visited group: " << nvfuser::toString(cur_group) << " ("
         << cur_group->front()->toString() << ")\n";
      for (auto expr : tv->domain()->allExprs()) {
        ss << expr->toString();
      }
      NVF_THROW(ss.str());
    }

    return {innermost_alloc_id, dep_alloc_ids};
  }

  static void validateAllocationVectorizedId(
      IterDomain* vec_alloc_id,
      const std::unordered_set<IterDomain*>& dep_alloc_ids,
      TensorView* tv,
      std::string name,
      int64_t vector_word_size_bit) {
    // aten_element_size_bit is the minimum unit (one element) of tv's
    // corresponding at::Tensor. It may or may not be the same as
    // dataTypeSizeBit(tv->dtype()), because we support non-ATen data types as
    // ATen tensor. See the comment of AdjustLastDim in type.h for more details.
    // For example, for fp4 tensor, we use Byte as the corresponding ATen
    // ScalarType, so aten_element_size_bit is 8 bits instead of 4 bits.
    int64_t aten_element_size_bit =
        c10::elementSize(
            data_type_to_aten(tv->dtype(), GpuLower::current()->indexType())) *
        8;
    // Contiguity is based on logical domain.
    IterDomain* last_alloc_dim = nullptr;
    size_t last_alloc_dim_pos = 0;
    for (size_t i = tv->getMaybeAllocationDomain().size(); i > 0; i--) {
      auto r_id = tv->getMaybeAllocationDomain()[i - 1];
      if (r_id->isReduction() || r_id->isBroadcast() || r_id->isDeviceDim()) {
        continue;
      }
      if ((tv->getMemoryType() == MemoryType::Shared ||
           tv->getMemoryType() == MemoryType::Local) &&
          r_id->isBlockDim()) {
        // Inner-most parallelized dimensions don't count in allocation of
        // shared and local tensors.
        continue;
      }
      if (tv->getMemoryType() == MemoryType::Local && r_id->isThreadDim()) {
        continue;
      }
      last_alloc_dim = r_id;
      last_alloc_dim_pos = i - 1;
      break;
    }

    if (last_alloc_dim == nullptr) {
      // Should never get here, but that would mean there are no concrete
      // dims, so we should be fine.
      return;
    }

    auto ldst = dynamic_cast<LoadStoreOp*>(tv->definition());

    bool is_ldmatrix_trans =
        ldst != nullptr && mma_utils::isLdMatrixTranspose(ldst);
    if (!is_ldmatrix_trans && name.compare("consumer") != 0) {
      // ldmatrix.trans is a hardware transpose instruction that can do
      // "vectorized" read from discontiguous memory
      // We don't think allocation domain of consumer is used in allocation. We
      // skip it in validation here. Note that this assert was hit for
      // vectorized pad, because we do not propagate allocation domain for
      // PadOp. See: https://github.com/NVIDIA/Fuser/pull/3439
      NVF_CHECK(
          last_alloc_dim == vec_alloc_id,
          "Vectorized dim for ",
          name,
          " has to be from an inner most position. tv: ",
          tv,
          ", allocation domain: ",
          tv->getMaybeAllocationDomain(),
          ", vectorized id: ",
          vec_alloc_id->toString(),
          ", innermost id: ",
          last_alloc_dim);

      // Because aten_element_size_bit is the minimum unit (one element) in
      // ATen, if one vector is smaller than one element, regardless of the
      // contiguity of the ATen tensor, we can always vectorize because an
      // element in ATen tensor is always contiguous by design.
      auto contiguity = tv->domain()->contiguity().at(last_alloc_dim_pos);
      NVF_CHECK(
          aten_element_size_bit % vector_word_size_bit == 0 ||
              contiguity.value_or(false),
          "The innermost position has to be contiguous. tv: ",
          tv,
          ", allocation domain: ",
          tv->getMaybeAllocationDomain(),
          ", innermost id: ",
          last_alloc_dim->toString(),
          ", contiguity: ",
          contiguity.has_value() ? (*contiguity ? "t" : "f") : "n");
    }
  }

  static IterDomain* getAndValidateVectorizedIdInAllocationDomain(
      IterDomain* v_id,
      TensorView* tv,
      std::string name,
      Expr* load_store,
      int64_t vector_word_size_bit) {
    const auto& [vec_alloc_id, dep_alloc_ids] =
        GpuLower::current()->info().hasIdModel()
        ? getDependentAllocIDsIdModel(v_id, tv, load_store)
        : getDependentAllocIDs(v_id, tv);

    validateAllocationVectorizedId(
        vec_alloc_id, dep_alloc_ids, tv, name, vector_word_size_bit);

    return vec_alloc_id;
  }

 private:
  std::unordered_set<IterDomain*> domains_;
  IterDomain* vectorized_id_ = nullptr;
  bool is_valid = true;

 public:
  static void validate(TensorView* tv) {
    // Make sure there's only one vectorized ID
    IterDomain* v_id = nullptr;
    for (auto id : tv->getLoopDomain()) {
      if (isParallelTypeVectorize(id->getParallelType())) {
        NVF_ERROR(
            v_id == nullptr,
            "Found two vectorized domains in ",
            tv,
            " only one is allowed.");
        v_id = id;
      }
    }

    // If no vectorized ids found simply return. If vectorized access is
    // broadcast, it won't generate an actual vector instruction, so can
    // be safely ignored
    if (v_id == nullptr || v_id->isBroadcast()) {
      return;
    }

    NVF_CHECK(
        v_id->extent()->isConstInt(),
        "Vectorizing a domain requires a constant integer size.");

    auto tv_def = tv->definition();
    NVF_ERROR(
        tv_def != nullptr,
        "Tv has no definition, cannot validate vectorization:",
        tv);

    auto vector_word_size = v_id->extent()->evaluate().as<int64_t>();
    auto vector_size_bit =
        dataTypeSizeBit(
            tv->getDataType().value(), GpuLower::current()->indexType()) *
        vector_word_size;
    if (tv_def->isA<LoadStoreOp>()) {
      // Except for TMem, allow half2, float2, float4 and same sized vtypes.
      std::vector<int64_t> allowed_vector_sizes_bit = {8, 16, 32, 64, 128};
      // with cuda-12.9 or later, devices 10.0 support 256 bit vectorization
      if (getMaxVectorizationSizeInBit() == 256) {
        allowed_vector_sizes_bit.push_back(256);
      }
      // TMem can vectorize up to 4096 bits.
      if (auto ldst = dynamic_cast<LoadStoreOp*>(tv_def); ldst != nullptr &&
          (ldst->opType() == LoadStoreOpType::LdTMem ||
           ldst->opType() == LoadStoreOpType::StTMem)) {
        if (allowed_vector_sizes_bit.back() != 256) {
          allowed_vector_sizes_bit.push_back(256);
        }
        allowed_vector_sizes_bit.push_back(512);
        allowed_vector_sizes_bit.push_back(1024);
        allowed_vector_sizes_bit.push_back(2048);
        allowed_vector_sizes_bit.push_back(4096);
      }

      NVF_CHECK(
          std::find(
              allowed_vector_sizes_bit.begin(),
              allowed_vector_sizes_bit.end(),
              vector_size_bit) != allowed_vector_sizes_bit.end(),
          "Tried to vectorize a dim resulting in a word size of ",
          vector_size_bit,
          " bits, however, vector sizes starting from and including ",
          allowed_vector_sizes_bit.front(),
          " bits upto and including ",
          allowed_vector_sizes_bit.back(),
          " bits are supported.");
    }

    if (!tv_def->isA<LoadStoreOp>()) {
      return;
    }

    auto consumer_vectorized_id = getAndValidateVectorizedIdInAllocationDomain(
        v_id, tv, "consumer", tv_def, vector_size_bit);
    if (consumer_vectorized_id == nullptr) {
      return;
    }

    // Save info required to lowering and runtime validation
    auto consumer_word_size_it =
        GpuLower::current()->vectorizedAccesses().find(tv);
    if (consumer_word_size_it !=
        GpuLower::current()->vectorizedAccesses().end()) {
      consumer_word_size_it->second =
          std::max(vector_word_size, consumer_word_size_it->second);
    } else {
      GpuLower::current()->vectorizedAccesses().emplace(tv, vector_word_size);
    }

    TensorView* producer_tv = nullptr;
    for (auto input : tv_def->inputs()) {
      // TernaryOp(where) could have multiple inputs. But we only support single
      // TensorView input for vectorization.
      if (!input->isA<TensorView>()) {
        continue;
      }
      // IndexSelectOp during validation has already been lowered after
      // indexing. At this point, we can only vectorize load on lookup_tv
      // (input0). Prior to lowering, if vectorized load happened for index_tv,
      // it would be handled by a load op via cached input. So we don't need to
      // consider them here.
      if (tv_def->isA<IndexSelectOp>()) {
        if (producer_tv == tv_def->as<IndexSelectOp>()->lookupTv()) {
          break;
        }
      }
      // Schedule operations are Fusion IR dependencies that do not appear in
      // CUDA kernel, so we skip them here.
      if (ir_utils::isScheduleOp(input->as<TensorView>())) {
        continue;
      }
      NVF_ERROR(
          producer_tv == nullptr,
          "Vectorization validation only support op with a single TensorView "
          "input");
      producer_tv = input->as<TensorView>();
      auto producer_word_size_it =
          GpuLower::current()->vectorizedAccesses().find(producer_tv);
      if (producer_word_size_it !=
          GpuLower::current()->vectorizedAccesses().end()) {
        producer_word_size_it->second =
            std::max(vector_word_size, producer_word_size_it->second);
      } else {
        GpuLower::current()->vectorizedAccesses().emplace(
            producer_tv, vector_word_size);
      }
    }
    NVF_ERROR(
        producer_tv != nullptr,
        "Vectorization validation requires a TensorView input");

    VectorizedSetInfo vectorized_set_info;
    vectorized_set_info.consumer_tv = tv;
    vectorized_set_info.producer_tv = producer_tv;
    // Note that VectorizedSetInfo is about each instance of
    // vectorized set operations, so the word size is the size of this
    // specific vectorized set.
    vectorized_set_info.word_size = vector_word_size;
    vectorized_set_info.vectorized_loop_id = v_id;
    vectorized_set_info.vectorized_consumer_alloc_id = consumer_vectorized_id;

    // Validate producer
    if (GpuLower::current()->info().hasIdModel()) {
      // No need to do replayPasC when using IdModel
      vectorized_set_info.vectorized_producer_alloc_id =
          getAndValidateVectorizedIdInAllocationDomain(
              v_id, producer_tv, "producer", tv_def, vector_size_bit);
    } else {
      auto pairwise_map = PairwiseLogicalDomainMap(producer_tv, tv);
      auto producer_replayed_as_consumer =
          TransformReplay::replayPasC(
              producer_tv,
              tv,
              -1,
              pairwise_map,
              TransformReplayOptions().replayResize())
              .first;
      ir_utils::TVDomainGuard domain_guard(
          producer_tv, producer_replayed_as_consumer);
      auto c2p_map =
          BestEffortReplay::replayPasC(producer_tv, tv, -1, pairwise_map)
              .getReplay();
      vectorized_set_info.vectorized_producer_alloc_id =
          getAndValidateVectorizedIdInAllocationDomain(
              c2p_map.at(v_id),
              producer_tv,
              "producer",
              tv_def,
              vector_size_bit);
    }

    // For aligned vectorize, the extent of a vectorized domain must
    // be divisible by the vector word size. The domain is usually
    // just one of the allocation domains, but can be a merged domain of
    // contiguous domains. Those domains are saved in
    // VectorizedSetInfo.contig_alloc_ids in
    // fillConsumerVectorizedContigAllocationDomains called in
    // lower_index_compute.
    GpuLower::current()->vectorizedSetInfo().emplace_back(vectorized_set_info);
  }
};

} // namespace

// Uses ContigIDs to find allocation contig domains that a vectorized domain
// depends on. As ContigIDs depends on HaloInfo, this must be done
// after HaloInfo is created.
void validateAndCollectVectorizeInfo(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateVectorize");
  FusionGuard fg(fusion);

  std::vector<Val*> used_vals = fusion->usedMathVals();
  for (auto* tv : ir_utils::filterByType<TensorView>(used_vals)) {
    bool has_vectorize_dim = false;

    for (const auto i : arange(tv->nDims())) {
      IterDomain* id = tv->axis(i);
      IterDomain* concrete_id = lower_utils::getConcreteLoopID(id);

      auto ptype = concrete_id->getParallelType();

      if (ptype == ParallelType::Vectorize) {
        // If we want to do this check up front we would have to do 2 things:
        // (1) Check that the tensor view with vectorize being set on it is
        // getting set outside the local compute at position
        // (2) Check any producers of the tensor view with vectorize being set
        // on it to make sure their compute at position isn't to the right of
        // the vectorize dim.
        NVF_ERROR(
            i >= tv->getMaxComputePosition(),
            "IterDomains to the left of the compute at point cannot be "
            "vectorized: ",
            tv,
            "\n");
        has_vectorize_dim = true;
      }

      // ParallelType::Group is used for both reduction and normalization.
      // In grouped outer reduction, the runtime function uses vectorized data
      // transfers between registers and shared memory. The producer tensor is
      // stored in registers and loaded into shared memory in a vectorized
      // manner, so we add it to the vectorizedAccesses map to ensure register
      // alignment.
      if (ptype == ParallelType::Group) {
        auto def = tv->definition();
        auto grop = dynamic_cast<GroupedReductionOp*>(def);
        if (grop && (!grop->isAllreduce())) {
          auto vector_word_size =
              concrete_id->extent()->evaluate().as<int64_t>();
          auto producer_tv = def->inputs().at(0)->as<TensorView>();
          GpuLower::current()->vectorizedAccesses().emplace(
              producer_tv, vector_word_size);
        }
        break;
      }
    }

    if (has_vectorize_dim) {
      Expr* def = tv->definition();
      NVF_ERROR(
          def == nullptr || def->isA<LoadStoreOp>() || def->isA<SliceOp>() ||
              def->isA<PadOp>() || def->isA<IndexSelectOp>() ||
              (def->isA<TernaryOp>() &&
               def->as<TernaryOp>()->getTernaryOpType() ==
                   TernaryOpType::Where) ||
              (def->isA<ReductionOp>() &&
               def->as<ReductionOp>()->serialGridReductionRequested()) ||
              (def->isA<UnaryOp>() &&
               def->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Cast),
          "Vectorized accesses cannot be inline with computation: ",
          (def == nullptr ? tv->toString() : def->toString()));
    }
    // Validate the vectorized domain maps to the innermost domain of
    // tv. Note that we don't need to validate its producer tv as
    // Vectorize can only be used with
    // UnaryOp::Set.
    if (has_vectorize_dim) {
      VectorizeValidator::validate(tv);
    }
  }
}

namespace {

void fillVectorizedContigAllocationDomains(
    const TensorView* tv,
    const ContigIDs& contig_finder,
    IterDomain* vectorized_alloc_id,
    VectorizedSetInfo& info) {
  const auto& alloc_dom = tv->getMaybeAllocationDomain();

  // Find the allocation domains that are dependency of the merged contig
  // domain.

  auto consumer_indexed_it =
      contig_finder.allocToIndexedID().find(vectorized_alloc_id);
  NVF_ERROR(
      consumer_indexed_it != contig_finder.allocToIndexedID().end(),
      "Contiguity information not found for allocation domain: ",
      vectorized_alloc_id->toString());
  auto consumer_indexed_id = consumer_indexed_it->second;

  // Actual indexed allocation domains for this allocation domain. If
  // contig merge is done, multiple allocation domains are included.
  std::unordered_set<IterDomain*> indexed_alloc_ids;

  if (consumer_indexed_id == vectorized_alloc_id) {
    // Indexed domain is equal to the allocation domain, meaning no contig
    // merge is involved.
    indexed_alloc_ids.insert(vectorized_alloc_id);
  } else {
    auto consumer_within_contig_it =
        contig_finder.withinContigIDs().find(consumer_indexed_id);
    NVF_ERROR(
        consumer_within_contig_it != contig_finder.withinContigIDs().end());
    const auto& within_ids = consumer_within_contig_it->second;
    std::copy_if(
        alloc_dom.begin(),
        alloc_dom.end(),
        std::inserter(indexed_alloc_ids, indexed_alloc_ids.end()),
        [&](IterDomain* alloc_id) {
          return within_ids.find(alloc_id) != within_ids.end();
        });
  }

  // Store the contig merged allocation domains. If it is already set, pick
  // the smaller one as it is used for validating divisibility of the
  // merged extent.
  if (info.contig_alloc_ids.empty() ||
      indexed_alloc_ids.size() < info.contig_alloc_ids.size()) {
    info.contig_alloc_ids = indexed_alloc_ids;
  }
}

} // namespace

void fillConsumerVectorizedContigAllocationDomains(
    const TensorView* consumer_tv,
    const ContigIDs& contig_finder) {
  auto& info_vector = GpuLower::current()->vectorizedSetInfo();
  auto it = std::find_if(
      info_vector.begin(), info_vector.end(), [&consumer_tv](auto& info) {
        return info.consumer_tv == consumer_tv;
      });
  if (it == info_vector.end()) {
    return;
  }

  VectorizedSetInfo& info = *it;

  // info.vectorized_consumer_alloc_id is validated at this point to be the
  // last concrete allocation domain in consumer.
  auto consumer_alloc_id = info.vectorized_consumer_alloc_id;

  fillVectorizedContigAllocationDomains(
      consumer_tv, contig_finder, consumer_alloc_id, info);
}

void fillProducerVectorizedContigAllocationDomains(
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const ContigIDs& contig_finder) {
  auto& info_vector = GpuLower::current()->vectorizedSetInfo();
  auto it = std::find_if(
      info_vector.begin(),
      info_vector.end(),
      [&producer_tv, &consumer_tv](auto& info) {
        return info.consumer_tv == consumer_tv &&
            info.producer_tv == producer_tv;
      });
  if (it == info_vector.end()) {
    return;
  }

  VectorizedSetInfo& info = *it;

  fillVectorizedContigAllocationDomains(
      producer_tv, contig_finder, info.vectorized_producer_alloc_id, info);
}

namespace {

//! Validates that the operand and result tensors
//!  of mma ops are swizzled and also validates
//!  specialization of tidx as lane id.
void validateMmaTensors(MmaOp* mma) {
  bool tidx_validated = false;
  std::vector<TensorView*> to_validate = {mma->out()->as<TensorView>()};

  if (ir_utils::isLdMatrixOp(mma->inA()->definition())) {
    to_validate.push_back(mma->inA()->as<TensorView>());
  }
  if (ir_utils::isLdMatrixOp(mma->inB()->definition())) {
    to_validate.push_back(mma->inB()->as<TensorView>());
  }

  for (auto tv : to_validate) {
    for (auto id : tv->getLoopDomain()) {
      auto ptype = id->getParallelType();
      if (ptype == ParallelType::TIDx) {
        if (!tidx_validated) {
          // Check that TIDx is exact lane_id
          const auto& paralel_dim_map =
              GpuLower::current()->info().parallelDimensionMap();
          NVF_ERROR(
              lower_utils::isExtentEqualToMaxParallelTypeExtent(id) &&
                  paralel_dim_map.get(ptype)->isConstInt(),
              "TIDx is reserved for lane id in mma kernels");
          if (mma->isHopper()) {
            NVF_ERROR(
                paralel_dim_map.get(ptype)->evaluate() ==
                    at::cuda::warp_size() * 4,
                "TIDx must be exactly a warp group for Hopper");
          } else {
            NVF_ERROR(
                paralel_dim_map.get(ptype)->evaluate() == at::cuda::warp_size(),
                "TIDx must be exactly a warp for Turing/Ampere");
          }
          tidx_validated = true;
        }
      }
    }
  }

  // Note: this check will be relaxed in a follow up.
  auto validate_operand = [mma](const TensorView* tv, MmaOperand operand) {
    if (mma->isHopper() || mma->isBlackwell()) {
      if (operand == MmaOperand::B) {
        NVF_ERROR(
            tv->getMemoryType() == MemoryType::Shared,
            "Only supporting smem input for Hopper/Blackwell mma input B");
      } else if (mma->isHopper()) {
        NVF_ERROR(
            tv->getMemoryType() == MemoryType::Local ||
                tv->getMemoryType() == MemoryType::Shared,
            "Only supporting register or shared memory input for Hopper mma "
            "input A");
      } else if (mma->isBlackwell()) {
        NVF_ERROR(
            tv->getMemoryType() == MemoryType::Tensor ||
                tv->getMemoryType() == MemoryType::Shared,
            "Only supporting tensor or shared memory input for Blackwell mma "
            "input A");
      } else {
        NVF_THROW("Should not reach here");
      }
    } else {
      NVF_ERROR(
          tv->getMemoryType() == MemoryType::Local,
          "Only supporting register input for mma input on Ampere/Turing");
    }
  };

  validate_operand(mma->inA()->as<TensorView>(), MmaOperand::A);
  validate_operand(mma->inB()->as<TensorView>(), MmaOperand::B);
}

void validateSizeMemoryOp(LoadStoreOp* ldst) {
  if (!ldst->out()->isA<TensorView>()) {
    return;
  }

  if (ldst->opType() != LoadStoreOpType::CpAsync) {
    return;
  }

  int64_t byte_size = 1;
  auto output = ldst->out()->as<TensorView>();
  for (auto id : output->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      byte_size = id->extent()->evaluate().as<int64_t>();
      break;
    }
  }
  byte_size *= dataTypeSizeByte(
      *output->getDataType(), GpuLower::current()->indexType());

  switch (ldst->cacheOp()) {
    case CacheOp::Global:
      NVF_CHECK(byte_size == 16, "Not supported byte size for cp.async.cg");
      break;
    case CacheOp::AllLevels:
      NVF_CHECK(
          byte_size == 4 || byte_size == 8 || byte_size == 16,
          "Not supported byte size for cp.async.ca");
      return;
    default:
      return;
  }
}

} // namespace

//! Validate data format and GPU arch compatibility of scheduled
//!  mma operators on the fusion.
void validateMma(Fusion* fusion) {
  auto exprs = StmtSort::getExprs(fusion);

  for (auto expr : exprs) {
    if (auto mma = dynamic_cast<MmaOp*>(expr)) {
      validateMmaTensors(mma);
    }
    if (auto ldst = dynamic_cast<LoadStoreOp*>(expr)) {
      validateSizeMemoryOp(ldst);
    }
  }
}

namespace {

// Utility function to validate a loop swizzle:
//  1. Throws an error if any output of the swizzle is not in loop_domain set.
//  2. Warns if any output of the swizzle is not the concrete id of the loop
//  map.
// The second case would make the codegen ignore this swizzle, as if it was
// not there at all.
void validateLoopSwizzle(
    Expr* swizzle_expr,
    std::unordered_set<IterDomain*>& loop_domains) {
  for (auto out_id :
       ir_utils::filterByType<IterDomain>(swizzle_expr->outputs())) {
    NVF_ERROR(
        loop_domains.count(out_id),
        "Loop swizzle can only be direct producer of loop domains.");
    if (lower_utils::getConcreteLoopID(out_id) != out_id) {
      TORCH_WARN_ONCE("Ignored loop swizzle :", swizzle_expr->toString());
    }
  }
}

} // namespace

void validateSwizzle(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    if (tv->hasSwizzleOp()) {
      std::unordered_set<IterDomain*> tv_loop_domain_set(
          tv->getLoopDomain().begin(), tv->getLoopDomain().end());

      // Make sure no swizzle op is inlined:
      auto inlined_swizzles = ir_utils::getAllSwizzlesBetween(
          tv->getLogicalDomain(),
          {tv->getLoopDomain().begin(),
           tv->getLoopDomain().begin() + tv->getMaxComputePosition()});

      auto not_inlined_swizzles = ir_utils::getAllSwizzlesBetween(
          tv->getLogicalDomain(),
          {tv->getLoopDomain().begin() + tv->getMaxComputePosition(),
           tv->getLoopDomain().end()});

      // Check inlined swizzles: only loop swizzles can be inlined currently
      //  as inlining data swizzles would require addtional support of
      //  unswizzle operator, which currently doesn't have important use
      //  cases.
      for (auto swizzle_expr : inlined_swizzles) {
        NVF_ERROR(
            swizzle_expr->as<Swizzle2D>()->swizzleMode() == SwizzleMode::Loop,
            "Only support inlining loop swizzles");
        validateLoopSwizzle(swizzle_expr, tv_loop_domain_set);
      }

      std::unordered_set<Expr*> inlined_swizzle_set(
          inlined_swizzles.begin(), inlined_swizzles.end());

      // Check not inlined swizzles:
      //  Apply the loop swizzle check when it applies, and
      // also make sure that the no swizzle is also in inlined_swizzle set.
      // The latter would mean that one output of the swizzle is inlined while
      //  the other is not. Such case will not be supported.
      for (auto swizzle_expr : not_inlined_swizzles) {
        NVF_ERROR(
            !inlined_swizzle_set.count(swizzle_expr),
            "Cannot partially inline across swizzle domains.",
            swizzle_expr->toString());
        if (swizzle_expr->as<Swizzle2D>()->swizzleMode() == SwizzleMode::Loop) {
          validateLoopSwizzle(swizzle_expr, tv_loop_domain_set);
        }
      }
    }
  }
}

void validateAndConvertIterDomainGrouping(Fusion* fusion) {
  for (auto tv : fusion->allTvs()) {
    bool is_grouped = false;
    for (const auto id_idx : arange(tv->nDims())) {
      const auto id = tv->axis(id_idx);
      auto ptype = lower_utils::getConcreteLoopID(id)->getParallelType();
      if (ptype != ParallelType::Group) {
        // Not a grouped ID
        continue;
      }

      // Remember if a grouped ID is found
      is_grouped = true;

      // Grouping only makes sense for the normal iteration or gather scatter
      // type
      NVF_CHECK(
          id->getIterType() == IterType::Iteration ||
              id->getIterType() == IterType::GatherScatter,
          "Invalid use of ParallelType::Group.",
          " Grouping of ",
          id->getIterType(),
          " is not allowed. ",
          tv->toString());

      // Extent must be static
      NVF_CHECK(
          id->extent()->value().is<int64_t>(),
          "Invalid use of ParallelType::Group.",
          " IterDomain must have a static extent: ",
          id->toString());

      // The CA position must be left of any grouped ID
      NVF_CHECK(
          tv->getMaxComputePosition() <= id_idx,
          "Invalid use of ParallelType::Group.",
          " ComputeAt position must be left of grouped IDs: ",
          tv->toString());

      // Similarly, the produce-at position must be left of any grouped ID
      NVF_CHECK(
          tv->getMaxProducerPosition() <= id_idx,
          "Invalid use of ParallelType::Group.",
          " ProduceAt position must be left of grouped IDs: ",
          tv->toString());
    }

    if (!is_grouped) {
      continue;
    }

    // Must be defined by ReductionOp
    auto def = tv->definition();
    NVF_CHECK(
        def != nullptr,
        "Invalid use of ParallelType::Group.",
        " Definition of tv with ParallelType::Group not found. ",
        tv->toString());

    NVF_CHECK(
        def->isA<ReductionOp>() || def->isA<GroupedReductionOp>() ||
            def->isA<WelfordOp>() || def->isA<GroupedWelfordOp>() ||
            def->isA<ArgsortOp>() || def->isA<ScanOp>() || def->isA<TopKOp>() ||
            def->isA<BlockQuantizationOp>(),
        "Invalid use of ParallelType::Group: ",
        def->toString());

    // Convert ReductionOp to GroupedReductionOp
    if (tv->definition()->isA<ReductionOp>()) {
      auto rop = def->as<ReductionOp>();
      auto is_allreduce = rop->isAllreduce();

      std::vector<BinaryOpType> op_types({rop->getReductionOpType()});
      std::vector<Val*> init_vals({rop->init()});
      std::vector<Val*> outputs({rop->out()});
      std::vector<Val*> inputs({rop->in()});

      fusion->removeExpr(rop);
      IrBuilder::createInContainer<GroupedReductionOp>(
          fusion, op_types, init_vals, outputs, inputs, is_allreduce);
    } else if (tv->definition()->isA<WelfordOp>()) {
      // Convert WelfordOp to GroupedWelfordOp
      auto wop = def->as<WelfordOp>();
      auto is_allreduce = wop->isAllreduce();

      NVF_CHECK(
          is_allreduce,
          "Invalid use of ParallelType::Group.",
          " Only enabled for allreduce reductions: ",
          wop->toString());

      NVF_CHECK(
          tv->domain()->hasGridReduction(),
          "Invalid use of ParallelType::Group.",
          " Only enabled for grid reductions: ",
          wop->toString());

      std::vector<WelfordTriplet> output_vals(
          {{wop->outAvg(), wop->outVar(), wop->outN()}});
      std::vector<WelfordTriplet> input_vals(
          {{wop->inAvg(), wop->inVar(), wop->inN()}});
      std::vector<WelfordTriplet> init_vals(
          {{wop->initAvg(), wop->initVar(), wop->initN()}});
      fusion->removeExpr(wop);
      IrBuilder::createInContainer<GroupedWelfordOp>(
          fusion, output_vals, input_vals, init_vals, is_allreduce);
    }
  }
}

void validateGroupedReductions(Fusion* fusion) {
  for (auto expr : StmtSort::getExprs(fusion)) {
    if (auto grouped_reduction_op = dynamic_cast<GroupedReductionOp*>(expr)) {
      const auto num_exprs =
          grouped_reduction_op->numHorizontallyGroupedExprs();
      int64_t num_grouped_iterations = 1;
      auto out_tv = ir_utils::getTvOutput(grouped_reduction_op);
      for (auto axis : out_tv->getLoopDomain()) {
        if (axis->getParallelType() == ParallelType::Group) {
          num_grouped_iterations *= axis->extent()->value().as<int64_t>();
        }
      }
      NVF_CHECK(
          num_exprs * num_grouped_iterations <= kMaxNumGroupedReductions,
          "Too many grouped reductions: ",
          grouped_reduction_op->toString(),
          ". Up to ",
          kMaxNumGroupedReductions,
          " reductions are allowed.");
    }
  }
}

void validateLookupTV(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    if (expr->isA<SelectOp>() || expr->isA<IndexSelectOp>()) {
      NVF_CHECK(
          expr->input(0)->isFusionInput(),
          "Lookup input must be a fusion input: ",
          expr->toString());
    }
  }
}

// 1. Must have one and only one clustered domain
// 2. clustered domain must be parallelized with ParallelType::BIDx
// 3. reduction data type must be float or double
// 4. cluster size must be in the range of [2, max allowed cluster
// size]
void validateClusterReduction(ReductionOp* rop) {
  auto out = rop->out()->as<TensorView>();
  // 1. Must have one and only one clustered domain
  NVF_ERROR(
      std::count_if(
          out->getLoopDomain().begin(),
          out->getLoopDomain().end(),
          [](IterDomain* id) { return id->isClusteredBlockDim(); }) == 1,
      "Must have one and only one clustered domain.");

  // 2. clustered domain must be parallelized with ParallelType::BIDx
  auto it = std::find_if(
      out->getLoopDomain().begin(),
      out->getLoopDomain().end(),
      [](IterDomain* id) { return id->isClusteredBlockDim(); });
  auto clustered_domain = *it;
  NVF_ERROR(
      clustered_domain->getParallelType() == ParallelType::BIDx,
      "Clustered domain must be parallelized with ParallelType::BIDx.");

  // 3. reduction data type must be float or double
  NVF_ERROR(
      out->getDataType() == DataType::Float ||
          out->getDataType() == DataType::Double,
      "Clustered reduction only supports float or double reduction data type.");

  // 4. cluster size must be in the range of [2, max allowed cluster size].
  //  This is a runtime check
  auto is_legal_cluster_size = SimplifyingIrBuilder::logicalAndExpr(
      SimplifyingIrBuilder::leExpr(
          clustered_domain->extent(),
          IrBuilder::create<Val>(
              scheduler_utils::getMaxClusterSize(), DataType::Index)),
      SimplifyingIrBuilder::geExpr(
          clustered_domain->extent(),
          IrBuilder::create<Val>(2, DataType::Index)));
  GpuLower::current()->validate(
      is_legal_cluster_size,
      "Clustered domain size must be less than or equal to max allowed cluster "
      "size "
      "and larger than 1.",
      clustered_domain->extent()->toInlineString(),
      " is not.");
}

void validateReductions(Fusion* fusion) {
  for (auto rop : ir_utils::getOpsOfType<ReductionOp>(fusion)) {
    auto in = rop->in()->as<TensorView>();
    auto out = rop->out()->as<TensorView>();
    PairwiseLogicalDomainMap c2p_map(in, out);
    c2p_map.mapBroadcast(true);
    auto c2p = c2p_map.mapConsumerToProducer();
    for (auto out_id : out->getMaybeRootDomain()) {
      if (out_id->isReduction()) {
        auto in_it = c2p.find(out_id);
        NVF_ERROR(
            in_it != c2p.end(),
            "Could not find producer IterDomain mapped to ",
            out_id->toString());
        IterDomain* in_id = in_it->second;
        NVF_ERROR(
            !in_id->isBroadcast() || in_id->hasExpandedExtent(),
            "Reductions of unexpanded broadcast domains should be ",
            "converted to squeeze before lowering.");
      }
    }
    // At this point, ReductionOp is not converted to ClusterReductionOp yet.
    // Do extra checks of ReductionOp when clustered domain is found.
    if (std::any_of(
            out->getLoopDomain().begin(),
            out->getLoopDomain().end(),
            [](IterDomain* id) { return id->isClusteredBlockDim(); })) {
      validateClusterReduction(rop);
    }
  }
}

//! Validate f split output domain is loaded with 1D TMA, the split must be
//! divisible
void validate1dTmaLoad(Fusion* fusion) {
  for (auto tv : fusion->allTvs()) {
    if (!tv->definition() || !ir_utils::isCpAsyncBulk1D(tv->definition())) {
      continue;
    }
    // ensure there is one and only one domain that is parallelized with
    // ParallelType::Bulk
    std::optional<int64_t> tma_axis = std::nullopt;
    for (auto id_idx : arange(tv->nDims())) {
      const auto id = tv->axis(id_idx);
      if (id->getParallelType() == ParallelType::Bulk) {
        NVF_ERROR(
            !tma_axis.has_value(),
            "Expect one and only one domain that is parallelized with "
            "ParallelType::Bulk, but found multiple in: ",
            tv->toString());
        tma_axis = id_idx;
      }
    }
    NVF_ERROR(
        tma_axis.has_value(),
        "Expect one and only one domain that is parallelized with "
        "ParallelType::Bulk, but found none in: ",
        tv->toString());
    const auto all_exprs = DependencyCheck::getAllExprsBetween(
        {tv->getMaybeRootDomain().begin(), tv->getMaybeRootDomain().end()},
        {tv->axis(tma_axis.value())});
    for (auto expr : all_exprs) {
      if (auto split = dynamic_cast<Split*>(expr)) {
        NVFUSER_LOWER_VALIDATE(
            split->isDivisible(),
            "If split output domain is loaded with 1D TMA, the split must be "
            "divisible, got: ",
            split->toString());
      }
    }
    // size must be divisible by 16 bytes
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk
    Val* tma_bytes = SimplifyingIrBuilder::mulExpr(
        tv->axis(tma_axis.value())->extent(), dataTypeSizeByte(tv->dtype()));
    Val* tma_bytes_is_multiple_of_16 = SimplifyingIrBuilder::eqExpr(
        SimplifyingIrBuilder::modExpr(
            tma_bytes, IrBuilder::create<Val>(16, DataType::Index)),
        fusion->zeroVal());
    NVFUSER_LOWER_VALIDATE(
        tma_bytes_is_multiple_of_16,
        "Expect 1dTMA load of inner-most dimension to be divisible by 16 "
        "bytes, "
        "but got: ",
        tma_bytes->toInlineString(),
        " bytes, ",
        " tv:",
        tv->toString());
  }
}

void validateScatter(Fusion* fusion) {
  for (auto sop : ir_utils::getOpsOfType<ScatterOp>(fusion)) {
    auto in_tv = sop->in()->as<TensorView>();
    auto out_tv = sop->out()->as<TensorView>();

    // TensorIndexer currently only supports exact scatter ops
    NVF_ERROR(
        sop->exactSizes(),
        "Non-exact scatter is not yet supported: ",
        sop->toString());

    // Scatter is implemented as an in-place op. To lower it safely, it
    // needs to be able to alias each other. Here are the conditions to
    // make sure they are valid input and output tensors.

    NVF_ERROR_EQ(
        in_tv->uses().size(),
        1,
        "Scatter input can only be used by the scatter op: ",
        toDelimitedString(in_tv->uses()));

    NVF_ERROR_EQ(in_tv->getMemoryType(), out_tv->getMemoryType());
    NVF_ERROR_EQ(in_tv->getDeviceMesh(), out_tv->getDeviceMesh());

    // To avoid making the inference of the allocation domain further
    // convoluted, both non-global input and output must have
    // explicitly set allocation domains
    NVF_ERROR(
        in_tv->getMemoryType() == MemoryType::Global || in_tv->hasAllocation(),
        "Non-global scatter input must have an allocation domain");
    NVF_ERROR(
        out_tv->getMemoryType() == MemoryType::Global ||
            out_tv->hasAllocation(),
        "Non-global scatter output must have an allocation domain");

    auto is_exact_mapped = [](const std::vector<IterDomain*>& ids1,
                              const std::vector<IterDomain*>& ids2) -> bool {
      const auto& exact_graph =
          GpuLower::current()->info().idModel().idGraph(IdMappingMode::EXACT);

      if (ids1.size() != ids2.size()) {
        return false;
      }

      for (const auto& [id1, id2] : zip(ids1, ids2)) {
        if (!exact_graph.disjointValSets().strictAreMapped(id1, id2)) {
          return false;
        }
      }

      return true;
    };

    NVF_ERROR(
        is_exact_mapped(
            in_tv->getAllocationDomain(), out_tv->getAllocationDomain()),
        "Scatter input and output must have equivalent allocation domains");

    // Fusion input as scatter input is not yet supported
    NVF_ERROR(
        !in_tv->isFusionInput(),
        "Scatter with fusion input not supported: ",
        in_tv->toString());

    // Fusion output as scatter input is not allowed since aliasing is
    // not possible between the input and output of the scatter
    NVF_ERROR(
        !in_tv->isFusionOutput(),
        "Scatter with fusion output not allowed: ",
        in_tv->toString());

    // If the scatter output is a fusion output, aliasing to a fusion
    // input is not yet supported
    if (out_tv->isFusionOutput()) {
      NVF_ERROR(
          fusion->getOutputAlias(out_tv).aliased_io == nullptr,
          "Scatter output to an aliasing fusion output is not supported: ",
          out_tv->toString());
    }
  }
}

} // namespace nvfuser

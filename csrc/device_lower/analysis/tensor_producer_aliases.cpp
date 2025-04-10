// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/tensor_producer_aliases.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>
#include <type.h>
#include <val_graph.h>

#include <unordered_set>
#include <vector>

namespace nvfuser {

namespace {

bool isTrivialExpr(Expr* expr) {
  TensorView* in = ir_utils::getTvInput(expr);
  TensorView* out = ir_utils::getTvOutput(expr);
  if (in == nullptr || out == nullptr ||
      in->getMemoryType() != MemoryType::Global ||
      out->getMemoryType() != MemoryType::Global || out->isFusionOutput() ||
      !expr->isOneOf<BroadcastOp, LoadStoreOp, SqueezeOp>()) {
    return false;
  }
  // This is a tensor op that does no computation. However, it may still be
  // non-trivial if the allocation domains are not equivalent, as that would
  // indicate data movement. We check that any non-Broadcast and non-Reduction
  // domains in the two allocation domains are exact mapped and appear in the
  // same order. We also check that the tensors are fully contiguous so that we
  // can re-use the exact same linear index.
  // TODO: support discontiguous inputs. The output should be an intermediate
  // tensor which we would always assume to be contiguous, but the input might
  // not be. This would necessitate using a different linear index.
  const std::vector<IterDomain*>& in_alloc = TensorDomain::noReductions(
      TensorDomain::noBroadcasts(in->getMaybeAllocationDomain()));
  const std::vector<IterDomain*>& out_alloc =
      TensorDomain::noBroadcasts(out->getMaybeAllocationDomain());

  if (in_alloc.size() != out_alloc.size()) {
    // Non-trivial allocation domains cannot be in bijective correspondence if
    // there are different numbers of them
    return false;
  }

  std::vector<bool> in_contig;
  std::vector<bool> out_contig;
  for (const std::optional<bool>& c : in->getContiguity()) {
    if (c.has_value()) {
      in_contig.push_back(c.value());
    }
  }
  for (const std::optional<bool>& c : out->getContiguity()) {
    if (c.has_value()) {
      out_contig.push_back(c.value());
    }
  }

  const ValGraph& exact_graph =
      GpuLower::current()->idModel().idGraph(IdMappingMode::EXACT);

  for (size_t pos : arange(in_alloc.size())) {
    // At this point in_pos and out_pos are both in range and point to
    // non-broadcast IDs
    IterDomain* in_id = in_alloc.at(pos);
    IterDomain* out_id = out_alloc.at(pos);

    // If this allocation ID is parallelized such that its loop index is not
    // used, then we can ignore it for this analysis.
    const auto id_is_indexed = [](TensorView* tv, IterDomain* id) {
      // isMemoryPartitionedAcross is true if the dimension is not allocated
      return !ir_utils::isMemoryPartitionedAcross(
          tv->getMemoryType(), id->getParallelType());
    };
    if (!id_is_indexed(in, in_id) || !id_is_indexed(out, out_id)) {
      continue;
    }

    NVF_ERROR(
        out_contig.at(pos),
        "Found discontiguous intermediate global tensor ",
        out->toString());
    if (in_contig.at(pos) != out_contig.at(pos)) {
      return false;
    }

    if (exact_graph.toGroup(in_id) != exact_graph.toGroup(out_id)) {
      // Unmapped pair
      return false;
    }
  }

  return true;
}

} // namespace

void findTensorProducerAliases(Fusion* fusion) {
  // First, find any inputs that have been aliased as an output. Since we don't
  // currently guarantee that all reads of the aliased global intermediate
  // tensors are performed before the aliased output is written, we must exclude
  // aliases to these inputs.
  std::unordered_set<TensorView*> inputs_aliased_by_outputs;
  for (Val* v : fusion->outputs()) {
    if (auto* io_alias_target =
            dynamic_cast<TensorView*>(fusion->getOutputAlias(v).aliased_io)) {
      inputs_aliased_by_outputs.insert(io_alias_target);
    }
  }
  for (Expr* expr : fusion->exprs()) {
    TensorView* in = ir_utils::getTvInput(expr);
    if (in != nullptr && inputs_aliased_by_outputs.count(in) == 0 &&
        isTrivialExpr(expr)) {
      GpuLower::current()->aliasTensorProducer(ir_utils::getTvOutput(expr), in);
    }
  }
}

std::vector<Expr*> removeTensorProducerAliases(
    const std::vector<Expr*>& exprs) {
  // There are no ForLoops or IfThenElse exprs yet since this pass runs before
  // LoopNestGenerator
  std::vector<Expr*> filtered_exprs;
  for (Expr* expr : exprs) {
    // Here we check whether this expression's output is a TensorView aliased
    // to one of its producer tensors. If so, that indicates that this
    // expression is trivial and should not be considered for codegen.
    if (TensorView* tv_out = ir_utils::getTvOutput(expr); tv_out &&
        GpuLower::current()->getTensorProducerAlias(tv_out) != nullptr) {
      continue;
    }
    filtered_exprs.push_back(expr);
  }
  return filtered_exprs;
}

} // namespace nvfuser

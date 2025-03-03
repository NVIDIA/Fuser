// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/tensor_producer_aliases.h>
#include <device_lower/lower2device.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>
#include <type.h>

#include <unordered_set>

namespace nvfuser {

namespace {

bool isTrivialExpr(Expr* expr) {
  TensorView* in = ir_utils::getTvInput(expr);
  TensorView* out = ir_utils::getTvOutput(expr);
  if (in == nullptr || out == nullptr ||
      in->getMemoryType() != MemoryType::Global ||
      out->getMemoryType() != MemoryType::Global ||
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
  size_t in_pos = 0;
  size_t out_pos = 0;
  const std::vector<IterDomain*>& in_alloc = in->getMaybeAllocationDomain();
  const std::vector<IterDomain*>& out_alloc = out->getMaybeAllocationDomain();
  const std::vector<std::optional<bool>>& in_contig = in->getContiguity();
  const std::vector<std::optional<bool>>& out_contig = out->getContiguity();

  const ValGraph& exact_graph =
      GpuLower::current()->idModel().idGraph(IdMappingMode::EXACT);

  while (true) {
    while (in_pos < in_alloc.size() &&
           (in_alloc.at(in_pos)->isBroadcast() ||
            in_alloc.at(in_pos)->isReduction())) {
      in_pos++;
    }
    while (out_pos < out_alloc.size() && out_alloc.at(out_pos)->isBroadcast()) {
      out_pos++;
    }
    if (in_pos >= in_alloc.size()) {
      NVF_ERROR(
          out_pos >= out_alloc.size(),
          "Found non-broadcast output domain ",
          out_alloc.at(out_pos)->toString(),
          " which has no corresponding input allocation domain");
      break;
    }
    if (out_pos >= out_alloc.size()) {
      NVF_ERROR(
          in_pos >= in_alloc.size(),
          "Found non-broadcast non-reduction input domain ",
          in_alloc.at(in_pos)->toString(),
          " which has no corresponding output allocation domain");
      break;
    }
    // At this point in_pos and out_pos are both in range and point to
    // non-broadcast IDs
    IterDomain* in_id = in_alloc.at(in_pos);
    IterDomain* out_id = out_alloc.at(out_pos);

    NVF_ERROR(
        out_contig.at(out_pos).has_value() &&
            out_contig.at(out_pos).value() == true,
        "Found discontiguous intermediate global tensor ",
        out->toString());
    if (in_contig.at(in_pos) != out_contig.at(out_pos)) {
      return false;
    }

    if (exact_graph.toGroup(in_id) != exact_graph.toGroup(out_id)) {
      // Unmapped pair
      return false;
    }

    in_pos++;
    out_pos++;
  }

  return true;
}

} // namespace

void findTensorProducerAliases(Fusion* fusion) {
  for (Expr* expr : fusion->exprs()) {
    if (isTrivialExpr(expr)) {
      GpuLower::current()->aliasTensorProducer(
          ir_utils::getTvOutput(expr), ir_utils::getTvInput(expr));
    }
  }
}

} // namespace nvfuser

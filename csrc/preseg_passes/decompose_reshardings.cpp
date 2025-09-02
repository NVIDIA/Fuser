// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/decompose_reshardings.h>

#include <device_lower/utils.h>
#include <fusion.h>
#include <host_ir/lower_to_communication.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <linked_hash_map.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/composite.h>
#include <ops/utils.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {
namespace {

enum class ReshardPosition {
  kBefore,
  kAfter,
};

// TODO: We can either reshard the inputs of a resharding expression or
// the outputs. Currently, we reshard the outputs when there is only one
// input, otherwise we reshard the inputs. This heuristic should be smarter
// and attempt to minimize communication.
// We do no support resharding multi-output expressions. Fusions may contain
// multi-output expressions if they don't require resharding.
ReshardPosition whereToReshard(Expr* e) {
  if ((e->inputs().size() == 1 && e->outputs().size() == 1) ||
       isOptionEnabled(EnableOption::InsertReshardingAfter)) {
    return ReshardPosition::kAfter;
  } else {
    return ReshardPosition::kBefore;
  }
}

// This is supposed to be a sufficient condition for `e` to be a communication.
// However, the current implementation, forked from HostIrLower::canLower, is
// merely a best effort.
bool isLowerableToCommunication(Expr* e) {
  if (auto* reduction = dynamic_cast<ReductionOp*>(e)) {
    auto in = reduction->in()->as<TensorView>();
    auto out = reduction->out()->as<TensorView>();
    // get the reduced axis
    std::vector<IterDomain*> reduction_axis;
    std::copy_if(
        out->getLogicalDomain().begin(),
        out->getLogicalDomain().end(),
        std::back_inserter(reduction_axis),
        [](IterDomain* id) { return id->isReduction(); });
    // check whether the reduction involves only one axis
    if (reduction_axis.size() != 1) {
      return false;
    }

    // We check whether the reduced axis is sharded on the input. I think this
    // can be simplified to check only the output. However, we need to make
    // sure sharding propagation and unit tests uses `rDID` instead of `r`.
    const auto c2p_map =
        PairwiseLogicalDomainMap(in, out).mapConsumerToProducer();
    auto c2p_map_it = c2p_map.find(reduction_axis.at(0));
    return c2p_map_it != c2p_map.end() && c2p_map_it->second->isDeviceDim();
  }

  if (auto* ldst = dynamic_cast<LoadStoreOp*>(e)) {
    return ldst->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set;
  }

  return false;
}

void insertReshardingSetsBefore(Fusion* fusion) {
  // Remove this after we refactor this as a pre-segmenter pass.
  FusionGuard fg(fusion);
  for (Expr* expr : fusion->exprs()) {
    if (!isResharding(expr)) {
      continue;
    }

    if (isLowerableToCommunication(expr)) {
      continue;
    }

    if (whereToReshard(expr) != ReshardPosition::kBefore) {
      continue;
    }

    // Verify that multi-output expression requires no resharding.
    if (expr->outputs().size() > 1) {
      NVF_CHECK(
          !isResharding(expr),
          "Cannot handle resharding a multi-output expression: ",
          expr);
      continue;
    }

    if (!expr->output(0)->isA<TensorView>()) {
      continue;
    }
    auto output = expr->output(0)->as<TensorView>();

    std::unordered_set<TensorView*> inputs;
    for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
      if (haveDifferentShardings(input, output)) {
        inputs.insert(input);
      }
    }

    // Reshard each input of expr to match output if necessary
    std::vector<TensorView*> new_inputs;
    for (auto input : inputs) {
      // TODO: reuse cacheAfter?
      // TODO: here we should add a mechanism to potentially reuse the
      // inserted resharding accross all the consumer of the resharded tensor.
      // This way we could avoid wasteful resharding set insertion.
      TensorView* new_input = set(input);
      new_inputs.push_back(new_input);
      expr = ir_utils::replaceValInExprInputs(expr, input, new_input);
    }

    shardAllLike(output, new_inputs, allParallelTypes());
  }
}

void insertReshardingSetsAfter(Fusion* fusion) {
  // Remove this after we refactor this as a pre-segmenter pass.
  FusionGuard fg(fusion);
  // Iterate backwards over fusion expressions. Reshard after will
  // replace expressions that occur downstream from the current expression.
  // This will ensure we don't process an expression that has been deleted.
  auto exprs = fusion->exprs();
  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (!isResharding(expr)) {
      continue;
    }

    if (isLowerableToCommunication(expr)) {
      continue;
    }

    if (whereToReshard(expr) != ReshardPosition::kAfter) {
      continue;
    }

    if (!expr->output(0)->isA<TensorView>()) {
      continue;
    }
    auto output = expr->output(0)->as<TensorView>();

    std::unordered_set<TensorView*> inputs;
    for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
      if (haveDifferentShardings(input, output)) {
        inputs.insert(input);
      }
    }

    // Insert resharding set after the expr and update
    // output of expr to match input's sharding.
    // input [expr] output [set] new_output
    if (!inputs.empty()) {
      TensorView* input = *inputs.begin();
      TensorView* new_output = set(output);
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);
      // Update shardings new_output takes output's sharding,
      // output takes input's sharding
      shardAllLike(output, {new_output}, deviceAndStreamParallelTypes());

      // Consider a reshard case:
      //
      //   input [DIDx(i0), i1] -> op -> output [i0, DIDx(i1)]
      //
      // This is decomposed into:
      //
      //   input [DIDx(i0), i1] -> op -> output [DIDx(i0), i1] -> set ->
      //   new_output [i0, DIDx(i1)]
      //
      // ParallelType::Serial is required here so the output is sharded as
      // [DIDx(i0), i1] instead of [DIDx(i0), DIDx(i1)] when sharding using
      // input as the reference.
      shardAllLike(
          input,
          {output},
          std::unordered_set<ParallelType>(
              {kParallelTypeDIDs.begin(), kParallelTypeDIDs.end()}));
    }
  }
}

// Canonicalizes tv's loop domain for simplicity and working around schedulers'
// limitations. Many schedulers panic when seeing the input fusion segment
// contains non-DID loop splits. For example, an rFactor tensor may look like
// the following:
//
//                            r{k}
//                            /  \.
// [i{m}         i{n}    iDIDx{d}  r{k/d}]
//               /  \.
//            i{d} i{n/d}
//
// The split of i{n} is unnecessary because i{d} and i{n/d} are both
// ParallelType::Serial. This function replaces the two with i{n} in the loop
// domain.
void canonicalizeLoopDomain(TensorView* tv) {
  LinkedHashMap<IterDomain*, std::monostate> loop;
  for (IterDomain* id : tv->getLoopDomain()) {
    loop.pushBack(id, std::monostate());
  }

  for (Expr* transform :
       DependencyCheck::getAllExprsBetween(
           {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()},
           {tv->getLoopDomain().begin(), tv->getLoopDomain().end()}) |
           std::views::reverse) {
    auto* split = dynamic_cast<Split*>(transform);
    NVF_ERROR(
        split != nullptr,
        "Only splits are expected so far, but found: ",
        transform);

    if (split->outer()->isParallelized() || split->inner()->isParallelized()) {
      continue;
    }

    if (!loop.contains(split->outer()) || !loop.contains(split->inner())) {
      continue;
    }

    loop.erase(split->outer());
    const auto inner_i = loop.erase(split->inner()).second;
    // `inner_i` is picked arbitrarily as the insertion point. Given `in`,
    // `outer` and `inner` are all serial, `in`'s position in the loop domain
    // doesn't matter.
    loop.insert(inner_i, split->in(), std::monostate());
  }

  auto new_loop = std::views::keys(loop);
  tv->setLoopDomain({new_loop.begin(), new_loop.end()});
}

void decomposeRowParallelLinearWithBias(Fusion* fusion) {
  // Iterate backwards over fusion expressions to avoid processing
  // expressions that have been deleted.
  // Recall that replaceValInAllExprInputsAndFusionOutputs invalidates
  // consumers.
  for (Expr* e : fusion->exprs() | std::views::reverse) {
    auto* linear_op = dynamic_cast<LinearOp*>(e);
    if (linear_op == nullptr) {
      continue;
    }

    if (!linear_op->hasBias()) {
      continue;
    }

    TensorView* out = linear_op->out();
    if (std::none_of(
            out->getLoopDomain().begin(),
            out->getLoopDomain().end(),
            [](IterDomain* id) {
              return id->isReduction() && id->isParallelized();
            })) {
      continue;
    }

    auto* without_bias = linear(linear_op->inA(), linear_op->inB());
    TransformReplay::selfReplay(out->domain(), without_bias->domain());

    TensorView* broadcasted_bias = [&]() {
      const int64_t rank_after_broadcast = std::ssize(
          TensorDomain::noReductions(without_bias->getLogicalDomain()));
      NVF_ERROR(
          rank_after_broadcast > 0,
          "without_bias is expected to be at least 1D: ",
          without_bias);
      std::vector<bool> is_broadcast_dim(rank_after_broadcast, true);
      is_broadcast_dim.back() = false;
      return broadcast(linear_op->bias(), is_broadcast_dim);
    }();

    TensorView* new_out =
        maybeCastOp(out->dtype(), add(without_bias, broadcasted_bias));
    TransformReplay::selfReplay(
        out->domain(), new_out->domain(), /*ignore_reductions=*/true);
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(out, new_out);
  }
}

// If a TensorView has a reduction dimension that's DID-split, we R-factor the
// TensorView into a local reduction followed by an allreduce.
//
// For example,
//
//   [i{m} i{k}]               [i{n} i{k}]
//         /   \                     /   \.
//     iDID{d} i{k/d}           iDID{d}  i{k/d}
//                    |
//                    | linear
//                    v
//               [i{m} i{n} r{k}]
//                         /   \.
//                    rDID{d}  r{k/d}
//
// is decomposed into
//
//                    |
//                    | linear (local)
//                    v
//                          r{k}
//                         /   \.
//          [i{m} i{n} iDID{d}  r{k/d}
//                    |
//                    | reduce (allreduce)
//                    v
//             [i{m} i{n} rDID{d}]
//
void rFactorLoopSplits(Fusion* fusion) {
  for (TensorView* tv : fusion->allTvs()) {
    std::vector<int64_t> rfactor_axes;
    rfactor_axes.reserve(tv->nDims());

    std::unordered_set<ParallelType> reduced_parallel_types;

    for (auto&& [i, loop_id] : enumerate(tv->getLoopDomain())) {
      if (!loop_id->isReduction()) {
        // rFactor only applies to reduction dimensions.
        continue;
      }

      if (std::count(
              tv->getLogicalDomain().begin(),
              tv->getLogicalDomain().end(),
              loop_id) > 0) {
        // No need to rFactor if loop_id is in the logical domain.
        continue;
      }

      const ParallelType parallel_type = loop_id->getParallelType();
      if (parallel_type == ParallelType::Serial) {
        // rFactor non-parallelized IDs so they get reduced locally.
        rfactor_axes.push_back(i);
      } else {
        reduced_parallel_types.insert(parallel_type);
      }
    }

    if (!rfactor_axes.empty()) {
      TensorView* local = tv->rFactor(rfactor_axes);
      // Before rFactor:
      //
      // [i{m}         i{n}         r{k}]
      //               /  \         /   \.
      //         iDIDx{d} i{n/d} rDIDx{d}  r{k/d}
      //
      // After rFactor:
      //
      //                            r{k}
      //                            /  \.
      // [i{m}         i{n}    iDIDx{d}  r{k/d}]
      //               /  \.
      //         iDIDx{d} i{n/d}
      //
      //                 |
      //                 | reduce
      //                 v
      //
      // [i{m}         i{n}    rDIDx{d}]
      //               /  \.
      //         iDIDx{d} i{n/d}
      //
      // The TensorView returned by rFactor has two iDIDx, which is disallowed.
      // The following code unparallelizes the first iDIDx{d}.
      for (IterDomain* loop_id : local->getLoopDomain()) {
        if (!loop_id->isRFactorProduct() &&
            reduced_parallel_types.count(loop_id->getParallelType())) {
          loop_id->parallelize(ParallelType::Serial);
        }
      }

      canonicalizeLoopDomain(local);
    }
  }
}

} // namespace

void DecomposeReshardingsPass::runPass(Fusion* fusion) {
  decomposeRowParallelLinearWithBias(fusion);

  rFactorLoopSplits(fusion);

  // whereToReshard selects whether insertReshardingSetsAfter or
  // insertReshardingSetsBefore is used.
  insertReshardingSetsAfter(fusion);
  insertReshardingSetsBefore(fusion);

  // Validate
  for (Expr* e : fusion->exprs()) {
    if (isResharding(e)) {
      getCommunicationInfo(e);
    }
  }
}

} // namespace nvfuser::preseg_passes

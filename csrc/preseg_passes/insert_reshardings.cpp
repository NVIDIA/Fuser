// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/insert_reshardings.h>

#include <device_lower/utils.h>
#include <fusion.h>
#include <host_ir/lower.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <ops/alias.h>

namespace nvfuser::preseg_passes {
namespace {
// TODO: We can either reshard the inputs of a resharding expression or
// the outputs. Currently, we reshard the outputs when there is only one
// input, otherwise we reshard the inputs. This heuristic should be smarter
// and attempt to minimize communication.
// We do no support resharding multi-output expressions. Fusions may contain
// multi-output expressions if they don't require resharding.
bool shouldReshardAfter(Expr* expr) {
  return expr->inputs().size() == 1 && expr->outputs().size() == 1;
}

void insertReshardingSetsBefore(Fusion* fusion) {
  // Remove this after we refactor this as a pre-segmenter pass.
  FusionGuard fg(fusion);
  for (Expr* expr : fusion->exprs()) {
    if (HostIrLower::canLower(expr, /*ignore_inner_resharding=*/true) ||
        shouldReshardAfter(expr)) {
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
    if (HostIrLower::canLower(expr, /*ignore_inner_resharding=*/true) ||
        !shouldReshardAfter(expr)) {
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
      shardAllLike(input, {output}, allParallelTypes());
    }
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
    }
  }
}

} // namespace

void InsertReshardingsPass::runPass(Fusion* fusion) {
  rFactorLoopSplits(fusion);

  // shouldReshardAfter selects whether insertReshardingSetsAfter or
  // insertReshardingSetsBefore is used.
  insertReshardingSetsAfter(fusion);
  insertReshardingSetsBefore(fusion);
}

} // namespace nvfuser::preseg_passes

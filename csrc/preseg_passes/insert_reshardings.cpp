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
#include <scheduler/utils.h>

namespace nvfuser::preseg_passes {
namespace {
// TODO: We can either reshard the inputs of a resharding expression or
// the outputs. Currently, we reshard the outputs when there is only one
// input, otherwise we reshard the inputs. This heuristic should be smarter
// and attempt to minimize communication.
// We do no support resharding multi-output expressions. Fusions may contain
// multi-output expressions if they don't require resharding.
bool shouldReshardAfter(Expr* expr) {
  return true;
}

void propagateParallelization(TensorView* ref, std::vector<TensorView*> tvs) {
  shardAllLike(ref, tvs);
  if (!tvs.empty()) {
    scheduler_utils::parallelizeAllLike(ref, tvs, {ParallelType::Stream});
  }
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
    propagateParallelization(output, new_inputs);
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
      propagateParallelization(output, {new_output});
      propagateParallelization(input, {output});

      // TODO: put this in a separate pass to propagate Stream parallelization

      // Find the stream parallelized axis in new_output
      IterDomain* stream_axis = nullptr;
      for (auto* id : new_output->getLoopDomain()) {
        if (id->getParallelType() == ParallelType::Stream) {
          stream_axis = id;
          break;
        }
      }
      if (stream_axis == nullptr) {
        return;
      }

      // Step 1: build the mapping of DIDx axis
      // collect the set of all loop domain axis between output and new_output that are DID-x parallelized.
      // Collect all DIDx parallelized axes from output and new_output
  
      // Build exact mapping model between output and new_output
      IdModel id_model(fusion);
      id_model.buildExactGraph();

      // Collect all DIDx parallelized axes from output and new_output
      std::unordered_set<IterDomain*> didx_axes;
      for (auto* tv : {output, new_output}) {
        auto loop_domain = tv->getLoopDomain();
        std::copy_if(
            loop_domain.begin(),
            loop_domain.end(), 
            std::inserter(didx_axes, didx_axes.begin()),
            [](IterDomain* id) { return id->getParallelType() == ParallelType::DIDx; });
      }

      // Helper to check if an axis maps to any DIDx axis
      auto is_mapped_to_didx = [&](IterDomain* id) {
        return std::any_of(
            didx_axes.begin(),
            didx_axes.end(),
            [&](IterDomain* didx_id) {
              return id_model.idGraph(IdMappingMode::EXACT)
                  .disjointValSets()
                  .strictAreMapped(id, didx_id);
            });
      };

      // Step 2:
      if (!is_mapped_to_didx(stream_axis)) {
        return;
      }

      // Step 3:
      // Find first Serial axis in output that maps to DIDx
      for (auto* id : output->getLoopDomain()) {
        if (id->getParallelType() == ParallelType::Serial && is_mapped_to_didx(id)) {
          id->parallelize(ParallelType::Stream);
          break;
        }
      }
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

      if (!loop_id->isParallelized()) {
        // rFactor non-parallelized IDs so they get reduced locally.
        rfactor_axes.push_back(i);
      }
    }

    if (!rfactor_axes.empty()) {
      tv->rFactor(rfactor_axes);
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

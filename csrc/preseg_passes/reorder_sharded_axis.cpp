// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/reorder_sharded_axis.h>

#include <device_lower/utils.h>
#include <fusion.h>
#include <host_ir/lower.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <id_model/id_model.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

void ReorderShardedAxisPass::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);

  const std::vector<Expr*>& exprs = fusion->exprs();
  IdModel id_model(fusion);
  id_model.buildExactGraph();
  ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    // if (HostIrLower::canLower(expr)) {
    //   continue;
    // }
    NVF_ERROR(
        ir_utils::isTvOp(expr),
        "Non-tv op is not supported: ",
        expr->toString());
    NVF_ERROR(
        expr->outputs().size() == 1,
        "Resharding operations can only have one output: ",
        expr->toString());
    NVF_ERROR(
        expr->inputs().size() == 1,
        "Resharding operations can have only one input: ",
        expr->toString());
    auto* output = expr->outputs().at(0)->as<TensorView>();
    auto* input = expr->inputs().at(0)->as<TensorView>();
    auto [shard_additions, shard_deletions] = getShardingChanges(input, output);
    NVF_ERROR(
        shard_additions.size() + shard_deletions.size() <= 1,
        "Resharding expr can only support one axis: ",
        expr->toString())

    // For gather operations i.e. ID goes from sharded to unsharded
    // this will rematerialize a sharded axis.
    // ProcessGroup expects contiguous tensors.
    // Update input to push the rematerialized axis to the front -> collective
    // -> permute the rematerialized axis to the proper location
    // Example: [i0 DIDx(i1)] -> [i0 i1]
    // Rewritten to: [i0 DIDx(i1)] -> [DIDx(i1) i0] -> [i1 i0] -> [i0 i1]
    // Note: there are no reduction based collectives that
    // materializes an axis so expr is guaranteed to be a set.
    if (!shard_deletions.empty() && isInnerResharding(expr)) {
      IterDomain* shard_deleted_id = shard_deletions[0];
      int64_t sharding_axis = input->domain()->rootPosOf(shard_deleted_id);

      TensorView* input_permute = permute(input, {{sharding_axis, 0}});
      TensorView* output_permute = set(input_permute);
      TensorView* new_output = permute(output_permute, {{0, sharding_axis}});
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);

      // Propagate shardings from input and manually apply sharding deletions.
      shardAllLike(
          input,
          {input_permute, output_permute, new_output},
          deviceAndStreamParallelTypes());
      output_permute->axis(0)->parallelize(ParallelType::Serial);
      new_output->axis(sharding_axis)->parallelize(ParallelType::Serial);
      output_permute->setDeviceMesh(output->getDeviceMesh());
      new_output->setDeviceMesh(output->getDeviceMesh());

      // Set allocation domain for comm in/out
      input_permute->setAllocationDomain(input_permute->getLoopDomain(), true);
      output_permute->setAllocationDomain(
          output_permute->getLoopDomain(), true);
    }
    // For scatter operations i.e. ID goes from unsharded to sharded
    // Update input to push the scattered axis to the front -> collective ->
    // permute the sharded axis to the proper location.
    // Scatter example: [i0 i1] -> [i0 DIDx(i1)]
    // Rewritten to [i0 i1] -> [i1 i0] -> [DIDx(i1) i0] -> [i0 DIDx(i1)]
    // Reduce Scatter example: [i0 DIDx(i1) i2] -> [i0 r1 DIDx(i2)]
    // Rewritten to: [i0 DIDx(i1) i2] -> [i2 i0 DIDx(i1)] ->
    //                    [DIDx(i2) i0 r1] -> [i0 DIDx(i2)]
    // Note that reduction axis shifts from axis=1 to axis=2.
    else if (!shard_additions.empty() && isInnerResharding(expr)) {
      IterDomain* shard_added_id = shard_additions[0];
      int sharding_axis =
          static_cast<int>(output->domain()->rootPosOf(shard_added_id));

      TensorView* input_permute = permute(input, {{sharding_axis, 0}});
      TensorView* output_permute = nullptr;
      // Calculate the number of reduction axes before the sharding axis.
      // After permuting the sharding axis to the front, the reduction axis
      // will be offset by this amount.
      auto reduction_axis = output->getReductionAxis();
      int num_reduction_axes_before_sharding_axis =
          (reduction_axis.has_value() &&
           sharding_axis > static_cast<int>(reduction_axis.value()))
          ? 1
          : 0;
      if (expr->isA<ReductionOp>()) {
        auto num_reduction_dims =
            output->domain()->nDims() - output->domain()->noReductions().size();
        NVF_ERROR(
            num_reduction_dims == 1,
            "Cannot support reducing multiple reduction axes ",
            expr->toString())
        int reduction_axis_after_permute =
            static_cast<int>(reduction_axis.value()) +
            num_reduction_axes_before_sharding_axis;
        auto* red_expr = dynamic_cast<ReductionOp*>(expr);
        output_permute = reductionOp(
            red_expr->getReductionOpType(),
            {reduction_axis_after_permute},
            red_expr->init(),
            input_permute);
      } else {
        output_permute = set(input_permute);
      }
      int sharding_axis_after_permute =
          sharding_axis - num_reduction_axes_before_sharding_axis;
      // Note this is a no-op and is moving a device parallel axis back
      TensorView* new_output =
          permute(output_permute, {{0, sharding_axis_after_permute}});
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);

      output_permute->axis(0)->parallelize(shard_added_id->getParallelType());
      new_output->axis(sharding_axis_after_permute)
          ->parallelize(shard_added_id->getParallelType());
      // `output_permute` and `new_output` have inherited mesh from `input`. We
      // need to change them to `output`'s mesh so communication is only
      // between `input_permute` and `output_permute`.
      output_permute->setDeviceMesh(output->getDeviceMesh());
      new_output->setDeviceMesh(output->getDeviceMesh());

      // Set allocation domain for comm in/out
      input_permute->setAllocationDomain(input_permute->getLoopDomain(), true);
      output_permute->setAllocationDomain(
          output_permute->getLoopDomain(), true);
    }

    // 1. Get the resharding id pair between producer and consumer.
    std::optional<std::pair<IterDomain*, IterDomain*>> resharding_id_pair = getReshardingIdPair(input, output, exact_graph);
    if (!resharding_id_pair.has_value()) {
      continue;
    }

    auto [p_loop_id, c_loop_id] = resharding_id_pair.value();

    // 2. These ids may not be present in the logical domain. Find the logical id that p_id was derived from.
    auto p_logical_dom = input->getLogicalDomain();
    auto p_inputs_in_logical = getInputsInTargetDomain(p_loop_id, p_logical_dom);
    NVF_ERROR(p_inputs_in_logical.size() == 1, "Expected exactly one input in logical domain");
    auto p_logical_id = p_inputs_in_logical.at(0);
    auto p_logical_idx = std::distance(p_logical_dom.begin(), std::find(p_logical_dom.begin(), p_logical_dom.end(), p_logical_id));
    
    auto propagateTransform = [] (TensorView* ref, std::vector<TensorView*> tvs) -> void {
      TransformPropagator propagator(ref);
      SetSelector selector({tvs.begin(), tvs.end()});
      MaxLogicalDomainInfoSpanningTree(ref, &selector).traverse(&propagator);
    };

    if (p_loop_id->isDeviceDim() && axisIndex(input->getMaybeAllocationDomain(), p_logical_id) > 0){
      TensorView* inp_copy = set(input); // Allocates sharded axis at the front
      TensorView* out_copy = set(inp_copy); // Communication op
      TensorView* new_output = set(out_copy); // Reverts the sharded axis to the original position

      propagateTransform(input, {inp_copy});
      shardAllLike(input, {inp_copy});

      propagateTransform(output, {out_copy, new_output});
      shardAllLike(output, {out_copy, new_output});

      for (auto tv : {inp_copy, out_copy}) {
        tv->setAllocationDomain(TensorDomain::orderedAs(tv->getMaybeAllocationDomain(), {{p_logical_idx, 0}}), true);
      }
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);
    }
  }
}

} // namespace nvfuser::preseg_passes

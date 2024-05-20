// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <compute_at_map.h>
#include <device_lower/utils.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/utils.h>

#include <c10/util/irange.h>

namespace nvfuser {

NVF_API bool distributedEnabled() {
#ifdef NVFUSER_DISTRIBUTED
  return true;
#else
  return false;
#endif
}

namespace {

std::unordered_set<IterDomain*> getShardedIterDomains(TensorView* tv) {
  std::unordered_set<IterDomain*> sharded_ids;
  std::copy_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      std::inserter(sharded_ids, sharded_ids.begin()),
      [](auto id) { return id->isDeviceDim(); });
  return sharded_ids;
}

// Returns whether a IterDomain in a TensorView is the outermost
// allocated IterDomain in the TensorView.
bool isOutermostAllocatedId(TensorView* tv, IterDomain* id) {
  for (auto i : tv->getLeafDomain()) {
    if (i == id) {
      return true;
    }
    if (!i->isDeviceDim() && !i->isReduction() && !i->isBroadcast()) {
      return false;
    }
  }
  NVF_ERROR(
      false, "Id", id->toString(), " is not in TensorView ", tv->toString());
  return false;
}

// For a resharding expression, either a set or reduce, returns root IDs
// that change sharding.
// (1) sharded root IterDomains that are added by the expression
// i.e. sharded IterDomains that are present in the output, but not the input.
// (2) sharded root IterDomains that are removed by the expression
// i.e. sharded IterDomains that are present in the input, but not the output.
// TODO: Analyze leaf domain for unsharded/sharded IDs and return their
// parent root IDs.
std::pair<std::vector<IterDomain*>, std::vector<IterDomain*>> getShardingChanges(
    Expr* expr) {
  NVF_ERROR(
      ir_utils::isTvOp(expr), "Expression must be a TvOp ", expr->toString());
  NVF_ERROR(
      expr->outputs().size() == 1,
      "Resharding expression can only have one output");
  NVF_ERROR(
      expr->inputs().size() == 1,
      "Resharding expression can have only one input");
  auto output = expr->outputs().at(0)->as<TensorView>();
  auto input = expr->inputs().at(0)->as<TensorView>();

  std::vector<IterDomain*> shard_additions;
  std::vector<IterDomain*> shard_deletions;
  auto rootmap = PairwiseRootDomainMap(input, output).mapBroadcast(false);
  const auto c2p_map = rootmap.mapConsumerToProducer();

  for (IterDomain* out_root : output->getRootDomain()) {
    IterDomain* in_root = c2p_map.at(out_root);
    // Ignore sharded broadcast domains and
    // sharded reductions on the output
    // ex. DIDx(i0) -> r(i0) or DIDx(i0) -> r(DIDx(i0))
    // since they don't affect allocation.
    if (in_root->isDeviceDim() && !in_root->isBroadcast() &&
        !out_root->isDeviceDim() && !out_root->isReduction()) {
      shard_deletions.push_back(in_root);
    } else if (
        !in_root->isDeviceDim() && out_root->isDeviceDim() &&
        !out_root->isBroadcast()) {
      shard_additions.push_back(out_root);
    } else if (in_root->isDeviceDim() && out_root->isDeviceDim()) {
      NVF_ERROR(
          in_root->getParallelType() == out_root->getParallelType(),
          expr->toString(),
          " reshards ",
          in_root->toString(),
          " to ",
          out_root->toString(),
          " which is not supported");
    }
  }
  return std::make_pair(shard_additions, shard_deletions);
}

} // namespace

bool isSharded(TensorView* tv) {
  bool is_sharded = false;
  auto rids = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  auto ids = TensorDomain::noReductions(tv->getLeafDomain());
  for (auto i : c10::irange(ids.size())) {
    // Only one axis can be sharded on DIDx.
    NVF_ERROR(
        !(is_sharded && ids[i]->isDeviceDim()),
        "Multiple IterDomains parallelized on DIDx in TensorView ",
        tv->toString());

    if (ids[i]->isDeviceDim()) {
      // Currently do not support split/merge on a device dimension.
      NVF_ERROR(
          std::find(rids.begin(), rids.end(), ids[i]) != rids.end(),
          "Cannot parallelize DIDx on a split/merge axis ",
          ids[i]->toString());
      is_sharded = true;
    }
  }
  return is_sharded;
}

int64_t numDeviceDims(TensorView* tv) {
  return std::count_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      [](IterDomain* id) { return id->isDeviceDim(); });
}

bool haveDifferentShardings(TensorView* producer, TensorView* consumer) {
  // exit early in the unsharded case for performance
  if (!producer->hasDeviceMesh() && !consumer->hasDeviceMesh()) {
    return false;
  }
  // If device mesh are different, the Expr is resharding
  if (!(producer->getDeviceMesh() == consumer->getDeviceMesh())) {
    return true;
  }
  // Create a map between producer's and consumer's IterDomains. We iterate
  // over producer's iterdomain and compare sharding type with consumer's
  // iterdomain
  const auto p2c_map =
      PairwiseRootDomainMap(producer, consumer).mapProducerToConsumer();
  for (auto p_id :
       TensorDomain::noReductions(producer->getMaybeRFactorDomain())) {
    auto p2c_map_it = p2c_map.find(p_id);
    NVF_ERROR(
        p2c_map_it != p2c_map.end(),
        "the producer ",
        producer,
        " has a dimension ",
        p_id,
        " that is not mapped to its consumer ",
        consumer);
    auto c_id = p2c_map_it->second;
    if (p_id->getParallelType() != c_id->getParallelType() &&
        (p_id->isDeviceDim() || c_id->isDeviceDim())) {
      // Mismatch found
      return true;
    }
  }
  return false;
}

bool isResharding(Expr* expr) {
  // we don't use getTvsWithDifferentSharding because it creates a computeAtMap,
  // which is too costly
  for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      // exit early in the unsharded case for performance
      if (haveDifferentShardings(input, output)) {
        return true;
      }
    }
  }
  return false;
}

bool isInnerResharding(Expr* expr) {
  NVF_ERROR(
      ir_utils::isTvOp(expr),
      "Non-tv op is not supported : ",
      expr->toString());
  NVF_ERROR(
      expr->outputs().size() == 1,
      "Resharding operations can only have one output");
  NVF_ERROR(
      expr->inputs().size() == 1,
      "Resharding operations can have only one input");
  auto output = expr->outputs().at(0)->as<TensorView>();
  auto input = expr->inputs().at(0)->as<TensorView>();
  auto [shard_additions, shard_deletions] = getShardingChanges(expr);
  NVF_ERROR(
      shard_additions.size() + shard_deletions.size() <= 1,
      "Resharding expr can only support one axis")

  if (!shard_deletions.empty()) {
    return !isOutermostAllocatedId(input, shard_deletions[0]);
  } else if (!shard_additions.empty()) {
    return !isOutermostAllocatedId(output, shard_additions[0]);
  }
  return false;
}

namespace {

void shardAllLike(TensorView* ref, std::vector<TensorView*> tvs) {
  for (auto tv : tvs) {
    tv->setDeviceMesh(ref->getDeviceMesh());
  }
  if (!tvs.empty()) {
    scheduler_utils::parallelizeAllLike(
        ref, tvs, {ParallelType::DIDx, ParallelType::Serial});
  }
}

// TODO: We can either reshard the inputs of a resharding expression or
// the outputs. Currently, we reshard the outputs when there is only one
// input, otherwise we reshard the inputs. This heuristic should be smarter
// and attempt to minimize communication.
bool shouldReshardAfter(Expr* expr) {
  return expr->inputs().size() == 1;
}

void insertReshardingBefore(Fusion* fusion) {
  // Remove this after we refactor this as a pre-segmenter pass.
  FusionGuard fg(fusion);
  for (auto expr : fusion->exprs()) {
    if (isLowerableToCommunication(expr) || shouldReshardAfter(expr)) {
      continue;
    }
    NVF_ERROR(
        ir_utils::isTvOp(expr),
        "Non-tv op is not supported yet: ",
        expr->toString());
    NVF_ERROR(
        expr->outputs().size() == 1,
        "multi-output expressions are not supported");

    auto output = expr->outputs().at(0)->as<TensorView>();
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
    shardAllLike(output, new_inputs);
  }
}

void insertReshardingsAfter(Fusion* fusion) {
  // Remove this after we refactor this as a pre-segmenter pass.
  FusionGuard fg(fusion);
  // Iterate backwards over fusion expressions. Reshard after will
  // replace expressions that occur downstream from the current expression.
  // This will ensure we don't process an expression that has been deleted.
  auto exprs = fusion->exprs();
  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (isLowerableToCommunication(expr) || !shouldReshardAfter(expr)) {
      continue;
    }
    NVF_ERROR(
        ir_utils::isTvOp(expr),
        "Non-tv op is not supported yet: ",
        expr->toString());
    NVF_ERROR(
        expr->outputs().size() == 1,
        "multi-output expressions are not supported");

    auto output = expr->outputs().at(0)->as<TensorView>();
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
      shardAllLike(output, {new_output});
      shardAllLike(input, {output});
    }
  }
}

void setShardedAllocationDomain(TensorView* tv) {
  if (!tv->hasAllocation()) {
    tv->setAllocationDomain(tv->getLeafDomain(), true);
  }
}

} // namespace

void propagateShardings(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    auto inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    auto outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    TensorView* input_with_mesh = nullptr;
    for (auto tv : inputs) {
      NVF_CHECK(
          tv->hasDeviceMesh(),
          "Expression inputs should be assigned a DeviceMesh ",
          expr->toString());
      if (input_with_mesh == nullptr) {
        input_with_mesh = tv;
      }
    }

    std::vector<TensorView*> outputs_without_mesh;
    for (auto tv : outputs) {
      if (!tv->hasDeviceMesh()) {
        outputs_without_mesh.push_back(tv);
      }
    }
    shardAllLike(input_with_mesh, outputs_without_mesh);
  }
}

void insertReshardings(Fusion* fusion) {
  // shouldReshardAfter selects whether insertReshardingAfter or
  // insertReshardingBefore is used.
  insertReshardingsAfter(fusion);
  insertReshardingBefore(fusion);
}

void insertShardedAxisReordering(Fusion* fusion) {
  auto exprs = fusion->exprs();
  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (!isResharding(expr)) {
      continue;
    }
    NVF_ERROR(
        ir_utils::isTvOp(expr),
        "Non-tv op is not supported : ",
        expr->toString());
    NVF_ERROR(
        expr->outputs().size() == 1,
        "Resharding operations can only have one output",
        expr->toString());
    NVF_ERROR(
        expr->inputs().size() == 1,
        "Resharding operations can have only one input",
        expr->toString());
    auto output = expr->outputs().at(0)->as<TensorView>();
    auto input = expr->inputs().at(0)->as<TensorView>();
    auto [shard_additions, shard_deletions] = getShardingChanges(expr);
    NVF_ERROR(
        shard_additions.size() + shard_deletions.size() <= 1,
        "Resharding expr can only support one axis ",
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
      shardAllLike(input, {input_permute, output_permute, new_output});
      output_permute->axis(0)->parallelize(ParallelType::Serial);
      new_output->axis(sharding_axis)->parallelize(ParallelType::Serial);
      output_permute->setDeviceMesh(output->getDeviceMesh());
      new_output->setDeviceMesh(output->getDeviceMesh());
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
      auto shard_added_id = shard_additions[0];
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
        auto red_expr = dynamic_cast<ReductionOp*>(expr);
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

      // Propagate shardings from input and manually apply sharding additions.
      shardAllLike(input, {input_permute, output_permute, new_output});
      output_permute->axis(0)->parallelize(shard_added_id->getParallelType());
      new_output->axis(sharding_axis_after_permute)
          ->parallelize(shard_added_id->getParallelType());
      output_permute->setDeviceMesh(output->getDeviceMesh());
      new_output->setDeviceMesh(output->getDeviceMesh());
    }
  }
}

void setShardedAllocationDomain(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    if (isResharding(expr)) {
      for (auto tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
        for (auto c : tv->getContiguity()) {
          if (c.has_value()) {
            NVF_CHECK(
                c.value(),
                "Resharding expression input must be contiguous ",
                expr);
          }
        }
        setShardedAllocationDomain(tv);
      }
      for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        setShardedAllocationDomain(tv);
      }
    }
  }
}

int64_t requestedNumberOfDevices(Fusion* fusion) {
  DeviceIdxType max_index = 0;
  for (auto tv : ir_utils::allTvs(fusion)) {
    if (tv->hasDeviceMesh()) {
      for (auto d_id : tv->getDeviceMesh().vector()) {
        max_index = std::max(max_index, d_id);
      }
    }
  }
  return static_cast<int64_t>(max_index + 1);
}

void unshard(TensorView* tv) {
  for (IterDomain* id : tv->getLeafDomain()) {
    if (id->isDeviceDim()) {
      id->parallelize(ParallelType::Serial);
    }
  }
  tv->setDeviceMesh(DeviceMesh());
}

void unshard(Fusion* fusion) {
  for (auto tv : ir_utils::allTvs(fusion)) {
    unshard(tv);
  }
}

std::set<DeviceIdxType> involvedDevices(Expr* expr) {
  std::set<DeviceIdxType> ret;
  for (const auto& tvs : {expr->inputs(), expr->outputs()}) {
    for (auto val : tvs) {
      NVF_ERROR(val->isA<TensorView>(), "Val is not a TensorView");
      auto tv = val->as<TensorView>();
      NVF_ERROR(tv->hasDeviceMesh(), "the TensorView has no device mesh");
      auto& mesh = tv->getDeviceMesh().vector();
      std::copy(mesh.begin(), mesh.end(), std::inserter(ret, ret.end()));
    }
  }
  return ret;
}

int64_t getShardedAxis(TensorView* tv) {
  auto ids = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i]->getParallelType() == ParallelType::DIDx) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

} // namespace nvfuser

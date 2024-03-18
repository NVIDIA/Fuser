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
namespace {
const std::unordered_set<IterDomain*> getShardedIterDomains(TensorView* tv) {
  std::unordered_set<IterDomain*> sharded_ids;
  std::copy_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      std::inserter(sharded_ids, sharded_ids.begin()),
      [](auto id) { return id->isDeviceDim(); });
  return sharded_ids;
}

// For a resharding expression, either a set or reduce, returns
// (1) sharded IterDomains that are added by the expression
// i.e. sharded IterDomains that are present in the output, but not the input.
// (2) sharded IterDomains that are removed by the expression
// i.e. sharded IterDomains that are present in the input, but not the output.
// TODO: Analysis is on the root domain since sharding is not supported
// on split/merged axes. Update to leaf domain analysis.
std::pair<std::vector<IterDomain*>, std::vector<IterDomain*>> shardMap(
    Expr* expr) {
  NVF_ERROR(
      expr->outputs().size() == 1,
      "Resharding operations can only have one output");
  NVF_ERROR(
      expr->inputs().size() == 1,
      "Resharding operations can have only one input");
  auto output = expr->outputs().at(0)->as<TensorView>();
  auto input = expr->inputs().at(0)->as<TensorView>();

  auto sharded_ids_input = getShardedIterDomains(input);
  auto sharded_ids_output = getShardedIterDomains(output);
  std::vector<IterDomain*> shard_additions;
  std::vector<IterDomain*> shard_deletions;
  auto rootmap = PairwiseRootDomainMap(input, output);

  const auto c2p_map = rootmap.mapConsumerToProducer(&sharded_ids_output);
  for (auto [id1, id2] : c2p_map) {
    if (!id1->sameAs(id2)) {
      shard_additions.push_back(id1);
    }
  }

  const auto p2c_map = rootmap.mapProducerToConsumer(&sharded_ids_input);
  for (auto [id1, id2] : p2c_map) {
    // TODO: temporarily don't push back reductions.
    // DIDx(i0) -> r(i0) should be represented as DIDx(i0)->r(DIDx(i0))
    if (!id1->sameAs(id2) && !id2->isReduction()) {
      shard_deletions.push_back(id1);
    }
  }
  return std::make_pair(shard_additions, shard_deletions);
}

// Returns whether a IterDomain in a TensorView can be read/written
// contiguously. when a IterDomain's sharding is flipped. This occurs when the
// IterDomain being unsharded/sharded is the outermost in the allocation domain.
// e.g. [DIDx(i0), r1, i2, i3]
// i0 => true, outermost axis
// i1 => true, reduction axis are not allocated
// i2 => true, reduction and device axis are not allocated
// i3 => false, i2 is allocated.
bool isContiguousShard(TensorView* tv, IterDomain* changed_id) {
  bool not_allocated = true;
  for (auto id : tv->getLeafDomain()) {
    if (id == changed_id) {
      return not_allocated;
    }
    not_allocated = not_allocated && (id->isDeviceDim() || id->isReduction());
  }
  NVF_ERROR(
      false,
      "Id",
      changed_id->toString(),
      " is not in TensorView ",
      tv->toString());
  return false;
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

template <typename TvIterator>
std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    TvIterator tvs) {
  std::unordered_set<TensorView*> ret;
  // isSharded asserts that there are no split/merge and that only the outmost
  // dimension is possibly sharded
  isSharded(ref);
  const auto& reference_dom = ref->getLeafDomain();
  FusionGuard fg(ref->fusion());
  auto ca_map = ComputeAtMap(FusionGuard::getCurFusion());
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;
  for (auto id : reference_dom) {
    auto ca_id =
        ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
    concrete_to_reference_map[ca_id] = id;
  }

  for (TensorView* tv : tvs) {
    isSharded(tv);
    if (!(ref->getDeviceMesh().vector() == tv->getDeviceMesh().vector())) {
      ret.insert(tv);
      continue;
    }
    for (auto id : tv->getLeafDomain()) {
      auto ca_id =
          ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
      if (concrete_to_reference_map.count(ca_id) > 0) {
        auto ref_id = concrete_to_reference_map.at(ca_id);
        if ((ref_id->isDeviceDim() || id->isDeviceDim()) &&
            ref_id->getParallelType() != id->getParallelType()) {
          ret.insert(tv);
          break;
        }
      }
    }
  }
  return ret;
}

bool isResharding(Expr* expr) {
  std::unordered_set<TensorView*> tvs;
  for (auto tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
    tvs.insert(tv);
  }
  for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
    tvs.insert(tv);
  }
  if (tvs.empty()) {
    return false;
  }
  auto tv_ref = *tvs.begin();
  tvs.erase(tv_ref);
  return !getTvsWithDifferentSharding(tv_ref, tvs).empty();
}

namespace {

void shardAllLike(TensorView* ref, std::vector<TensorView*> tvs) {
  for (auto tv : tvs) {
    tv->setDeviceMesh(ref->getDeviceMesh());
  }
  if (!tvs.empty()) {
    scheduler_utils::parallelizeAllLike(ref, tvs, {ParallelType::DIDx});
  }
}
} // namespace

void insertReshardings(Fusion* fusion) {
  auto exprs = fusion->exprs();
  for (auto expr : exprs) {
    if (isLowerableToCommunication(expr)) {
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
    std::vector<TensorView*> new_inputs;
    for (auto input : getTvsWithDifferentSharding(
             output, ir_utils::filterByType<TensorView>(expr->inputs()))) {
      // TODO: reuse cacheAfter?
      // TODO: here we should add a mechanism to potentially reuse the inserted
      // resharding accross all the consumer of the resharded tensor. This way
      // we could avoid wasteful resharding set insertion.
      TensorView* new_input = set(input);
      new_inputs.push_back(new_input);
      expr = ir_utils::replaceValInExprInputs(expr, input, new_input);
    }
    shardAllLike(output, new_inputs);
  }
}

void insertShardedAxisReordering(Fusion* fusion) {
  auto exprs = fusion->exprs();
  std::vector<Expr*> reshard_exprs;
  for (auto expr : exprs) {
    if (isResharding(expr)) {
      reshard_exprs.push_back(expr);
    }
  }
  for (auto expr : reshard_exprs) {
    NVF_ERROR(
        ir_utils::isTvOp(expr),
        "Non-tv op is not supported yet: ",
        expr->toString());
    NVF_ERROR(
        expr->outputs().size() == 1,
        "Resharding operations can only have one output");
    NVF_ERROR(
        expr->inputs().size() == 1,
        "Resharding operations can have only one input");
    auto output = expr->outputs().at(0)->as<TensorView>();
    auto input = expr->inputs().at(0)->as<TensorView>();
    auto [shard_additions, shard_deletions] = shardMap(expr);
    NVF_ERROR(
        shard_additions.size() + shard_deletions.size() <= 1,
        "Resharding expr can only support one axis")

    // For gather operations i.e. rematerializing an axis
    // communication libraries can only read/write contiguous tensors.
    // Update expr's input to push the rematerialized axis to the front
    // execute the expr then permute the rematerizlied axis to the proper
    // location. Originally: [input] -> set -> [output] Replace with [input] ->
    // permute -> [inpute_permute] -> set -> [output_permute] -> [output] Note:
    // there are no reduction based collectives that materialize an axis so expr
    // is guaranteed to be a set.
    if (!shard_deletions.empty()) {
      auto id = shard_deletions[0];
      int idx = static_cast<int>(input->domain()->posOf(id));
      if (isContiguousShard(input, id)) {
        continue;
      }
      auto ptype = id->getParallelType();

      // Note this first permute is a no-op. Moving a sharded id
      // has no affect on the underlying memory.
      TensorView* input_permute = permute(input, {{idx, 0}});
      input_permute->setDeviceMesh(input->getDeviceMesh());
      input_permute->axis(0)->parallelize(ptype);

      TensorView* output_permute = set(input_permute);
      output_permute->setDeviceMesh(output->getDeviceMesh());
      TensorView* new_output = permute(output_permute, {{0, idx}});
      new_output->setDeviceMesh(output->getDeviceMesh());

      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);
      fusion->removeExpr(expr);
    }
    // For scatter operations i.e. sharding an axis
    // Update expr's input to push the scattered axis to the front
    // execute the expr then permute the scatter axis to the proper location.
    // Originally: [input] -> set/reduce -> [output]
    // Replace with [input] -> permute -> [inpute_permute] -> set/reduce ->
    // [output_permute] -> permute-> [output]
    else if (!shard_additions.empty()) {
      auto id = shard_additions[0];
      int idx = static_cast<int>(output->domain()->posOf(id));
      if (isContiguousShard(output, id)) {
        continue;
      }
      auto ptype = id->getParallelType();

      TensorView* input_permute = permute(input, {{idx, 0}});
      input_permute->setDeviceMesh(input->getDeviceMesh());

      TensorView* output_permute = nullptr;
      // For reduce scatter, determine if the reduction axis shifted to the
      // right by 1.
      auto red_axis = output->getReductionAxis();
      int offset = (red_axis.has_value() && idx > red_axis.value()) ? 1 : 0;
      if (expr->isA<ReductionOp>()) {
        int raxis = static_cast<int>(red_axis.value()) + offset;
        input_permute->axis(raxis)->parallelize(ptype);
        auto red_expr = dynamic_cast<ReductionOp*>(expr);
        output_permute = reductionOp(
            red_expr->getReductionOpType(),
            {raxis},
            red_expr->init(),
            input_permute);
      } else {
        // Note this is a no-op and is moving a device parallel axis back
        output_permute = set(input_permute);
      }

      output_permute->axis(0)->parallelize(ptype);
      output_permute->setDeviceMesh(output->getDeviceMesh());
      int sharded_idx = idx - offset;
      TensorView* new_output = permute(output_permute, {{0, sharded_idx}});
      new_output->axis(sharded_idx)->parallelize(ptype);
      new_output->setDeviceMesh(output->getDeviceMesh());

      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);
      fusion->removeExpr(expr);
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
  tv->setDeviceMesh({});
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

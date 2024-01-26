// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <compute_at_map.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>

#include <c10/util/irange.h>

namespace nvfuser {

bool isSharded(TensorView* tv) {
  std::vector<bool> is_sharded;
  for (IterDomain* id : TensorDomain::noReductions(tv->getLeafDomain())) {
    is_sharded.push_back(id->isDeviceDim());
  }
  // Currently, only the most external dim is allowed to be sharded
  NVF_ERROR(tv->getMaybeRFactorDomain() == tv->getLeafDomain());
  for (auto i : c10::irange(1, is_sharded.size())) {
    NVF_ERROR(
        !is_sharded.at(i),
        "only the outmost dimension can be device-parallelized");
  }
  return is_sharded.empty() ? false : is_sharded.at(0);
}

namespace {

std::vector<IterDomain*> getShardedIterDomains(TensorView* tv) {
  std::vector<IterDomain*> sharded_ids;
  std::copy_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      std::back_inserter(sharded_ids),
      [](auto id) { return id->isDeviceDim(); });
  return sharded_ids;
}

} // namespace

std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    std::unordered_set<TensorView*> tvs) {
  std::unordered_set<TensorView*> ret;

  const auto& reference_dom = ref->getLeafDomain();
  FusionGuard fg(ref->fusion());
  auto ca_map = ComputeAtMap(FusionGuard::getCurFusion());
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;
  for (auto id : reference_dom) {
    auto ca_id =
        ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
    concrete_to_reference_map[ca_id] = id;
  }

  for (auto tv : tvs) {
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
  scheduler_utils::parallelizeAllLike(ref, tvs, {ParallelType::DIDx});
}

void reshardBefore(Expr* expr, Fusion* fusion) {
  NVF_ERROR(
      expr->outputs().size() == 1,
      "multi-output expressions are not supported");
  NVF_ERROR(
      expr->outputs().at(0)->isA<TensorView>(),
      "the expression's output is not a TensorView");
  TensorView* output = expr->outputs().at(0)->as<TensorView>();
  std::unordered_set<TensorView*> inputs;
  std::transform(
      expr->inputs().begin(),
      expr->inputs().end(),
      std::inserter(inputs, inputs.end()),
      [](Val* val) {
        NVF_ERROR(
            val->isA<TensorView>(),
            "the expression's input is not a TensorView");
        return val->as<TensorView>();
      });
  std::vector<TensorView*> new_inputs;
  // if the expr is not resharding, the following for loop is empty
  for (auto input : getTvsWithDifferentSharding(output, inputs)) {
    // TODO: reuse cacheAfter?
    // TODO: here we should add a mechanism to potentially reuse the inserted
    // resharding accross all the consumer of the resharded tensor. This way we
    // could avoid wasteful resharding set insertion.
    TensorView* new_input = set(input);
    expr = ir_utils::replaceValInExprInputs(expr, input, new_input);
    new_inputs.push_back(new_input);
  }
  if (!new_inputs.empty()) {
    shardAllLike(output, new_inputs);
  }
}

} // namespace

void insertReshardings(Fusion* fusion) {
  auto exprs = fusion->exprs();
  for (auto expr : exprs) {
    if (!isLowerableToCommunication(expr)) {
      // if the expr is not resharding, reshardBefore will not modify it
      reshardBefore(expr, fusion);
    }
  }
}

} // namespace nvfuser

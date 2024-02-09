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
#include <scheduler/utils.h>

#include <c10/util/irange.h>

namespace nvfuser {

bool isSharded(TensorView* tv) {
  std::vector<bool> is_sharded;
  for (IterDomain* id : TensorDomain::noReductions(tv->getLeafDomain())) {
    is_sharded.push_back(id->isDeviceDim());
  }
  // Currently, only the most external dim is allowed to be sharded and we don't
  // allow split/merge
  NVF_ERROR(tv->getMaybeRFactorDomain() == tv->getLeafDomain());
  for (auto i : c10::irange(1, is_sharded.size())) {
    NVF_ERROR(
        !is_sharded.at(i),
        "only the outmost dimension can be device-parallelized",
        "but axis ", i, " is sharded in tv ", tv->toString());
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
  scheduler_utils::parallelizeAllLike(ref, tvs, {ParallelType::DIDx});
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

int64_t requestedNumberOfDevices(Fusion* fusion) {
  std::set<DeviceIdxType> device_indices;
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->vals())) {
    if (tv->hasDeviceMesh()) {
      std::copy(
          tv->getDeviceMesh().vector().begin(),
          tv->getDeviceMesh().vector().end(),
          std::inserter(device_indices, device_indices.begin()));
    }
  }
  return static_cast<int64_t>(device_indices.size());
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
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->vals())) {
    unshard(tv);
  }
}

std::set<DeviceIdxType> involvedDevices(Expr* expr) {
  std::set<DeviceIdxType> ret;
  for (const auto& tvs : {expr->inputs(), expr->outputs()}) {
    for (auto tv : ir_utils::filterByType<TensorView>(tvs)) {
      NVF_ERROR(tv->hasDeviceMesh(), "the TensorView has no device mesh");
      auto& mesh = tv->getDeviceMesh().vector();
      std::copy(mesh.begin(), mesh.end(), std::inserter(ret, ret.end()));
    }
  }
  return ret;
}

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/utils.h>
#include <ir/internal_base_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
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
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      std::inserter(sharded_ids, sharded_ids.begin()),
      [](auto id) { return id->isDeviceDim(); });
  return sharded_ids;
}

// Returns whether a IterDomain in a TensorView is the outermost
// allocated IterDomain in the TensorView.
bool isOutermostAllocatedId(TensorView* tv, IterDomain* id) {
  for (auto i : tv->getLoopDomain()) {
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

} // namespace

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
  auto rootmap = PairwiseLogicalDomainMap(input, output).mapBroadcast(false);
  const auto c2p_map = rootmap.mapConsumerToProducer();

  for (IterDomain* out_root : output->getMaybeRootDomain()) {
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

bool isSharded(const TensorView* tv) {
  bool is_sharded = false;
  const auto& logical_ids = TensorDomain::noReductions(tv->getLogicalDomain());
  const auto& loop_ids = TensorDomain::noReductions(tv->getLoopDomain());
  for (auto i : c10::irange(loop_ids.size())) {
    if (!loop_ids[i]->isDeviceDim()) {
      continue;
    }

    // Only one axis can be sharded on DIDx.
    NVF_ERROR(
        !is_sharded,
        "Multiple IterDomains parallelized on DIDx in TensorView ",
        tv);

    // Currently do not support split/merge on a device dimension.
    NVF_ERROR(
        std::find(logical_ids.begin(), logical_ids.end(), loop_ids[i]) !=
            logical_ids.end(),
        "Cannot parallelize DIDx on a split/merge axis ",
        loop_ids[i]);

    is_sharded = true;
  }
  return is_sharded;
}

int64_t numDeviceDims(const TensorView* tv) {
  return std::count_if(
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      [](IterDomain* id) { return id->isDeviceDim(); });
}

bool haveDifferentShardings(
    const TensorView* producer,
    const TensorView* consumer) {
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
  const std::unordered_map<IterDomain*, IterDomain*>& p2c =
      PairwiseLogicalDomainMap(producer, consumer).mapProducerToConsumer();
  for (auto p_id : TensorDomain::noReductions(producer->getLogicalDomain())) {
    const auto i = p2c.find(p_id);
    if (i == p2c.end()) {
      // This happens e.g. when `p_id` is squeezed. Even if `p_id` is
      // parallelized on DID, the squeezed dimension is size-1 and doesn't
      // trigger resharding.
      continue;
    }

    auto c_id = i->second;
    if (p_id->getParallelType() != c_id->getParallelType() &&
        (p_id->isDeviceDim() || c_id->isDeviceDim())) {
      // Mismatch found
      return true;
    }
  }
  return false;
}

bool isResharding(const Expr* expr) {
  if (!ir_utils::isTvOp(expr)) {
    return false;
  }

  // Reduction over a sharded dimension. 
  if (expr->isA<ReductionOp>()) {
    // auto in = expr->as<ReductionOp>()->in()->as<TensorView>();
    auto out = expr->as<ReductionOp>()->out()->as<TensorView>();
    std::vector<IterDomain*> reduction_axis;
    std::copy_if(
        out->getLogicalDomain().begin(),
        out->getLogicalDomain().end(),
        std::back_inserter(reduction_axis),
        [](IterDomain* id) { return id->isReduction(); });
    if (reduction_axis[0]->isDeviceDim()) {
      return true;
    }
  }

  // We don't use getTvsWithDifferentSharding because it creates a computeAtMap,
  // which is too costly
  for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
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

void shardAllLike(TensorView* ref, std::vector<TensorView*> tvs, const std::unordered_set<ParallelType>& parallel_types) {
  for (auto tv : tvs) {
    tv->setDeviceMesh(ref->getDeviceMesh());
  }
  if (!tvs.empty()) {
    scheduler_utils::parallelizeAllLike(
        ref, tvs, parallel_types);
  }
}

void shardBetween(
  const std::vector<Expr*>& from, const std::vector<Expr*>& to, TensorView* ref_tv,
  const std::unordered_set<ParallelType>& parallel_types) {
    std::vector<TensorView*> from_tvs;
    std::vector<TensorView*> to_tvs;
    for (auto expr : from) {
      for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        from_tvs.push_back(tv);
      }
    }

    for (auto expr : from) {
      for (auto tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
        to_tvs.push_back(tv);
      }
    }

    for (auto expr : to) {
      for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        to_tvs.push_back(tv);
      }
    }

  auto tvs_between = scheduler_utils::getAllTvsFrom(from_tvs, {to_tvs.begin(), to_tvs.end()});

   std::cout << "Ref tv" << ref_tv->toString() << std::endl;
  for (auto tv : from_tvs) {
    std::cout << "From tv: " << tv->toString() << std::endl;
  }

  for (auto tv : to_tvs) {
    std::cout << "To tv: " << tv->toString() << std::endl;
  }

  std::vector<TensorView*> tvs_to_shard = {tvs_between.begin(), tvs_between.end()};
 
  std::cout << "Sharding: ";
  for (auto tv : tvs_to_shard) {
    std::cout << tv->toString() << std::endl;
  }
  shardAllLike(ref_tv, tvs_to_shard, parallel_types);
}

void shardBetween(
    const std::vector<TensorView*>& ref_tvs,
    const std::vector<TensorView*>& boundary_tvs,
    const std::unordered_set<ParallelType>& parallel_types) {
  // std::unordered_set<Val*> s = {ref_tvs.begin(), ref_tvs.end()};
  // std::vector<Val*> v = {boundary_tvs.begin(), boundary_tvs.end()};
  // // This won't get things that are dangling

  // auto between_vals =
  //     DependencyCheck::getAllValsBetween(s, v);
  // auto tvs_between = ir_utils::filterByType<TensorView>(between_vals);
  // Note: we use getAllTvsFrom instead of getAllTVsBetween so that we can get all TVs reachable from t that don't
  // cross the boundary. This is because (1) expressions like rng_uniform create a fresh TV
  // that is not along a path from user visible TVs. (2) multi-output expressions may have output 
  // tensors that are not along a path to the global output which would also be excluded. Our sharding
  // propagation checks check all TVs in the fusion are assigned a device mesh regardless if they are
  // reachable. To keep the checks simple, we require all TVs are assigned a mesh if they exist in the
  // fusion, regardless if they are reachable. 
  std::unordered_set<TensorView*> to_tvs = {boundary_tvs.begin(), boundary_tvs.end()};
  for (auto tv : ref_tvs) {
    auto expr = tv->definition();
    if (expr == nullptr) {
      continue;
    }
    for (auto tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      to_tvs.insert(tv);
    }
  }
  auto tvs_between = scheduler_utils::getAllTvsFrom(ref_tvs, to_tvs);


  std::vector<TensorView*> tvs_to_shard= {tvs_between.begin(), tvs_between.end()};
  // std::copy_if(
  //     tvs_between.begin(),
  //     tvs_between.end(),
  //     std::back_inserter(tvs_to_shard),
  //     [](TensorView* tv) { return !tv->hasDeviceMesh(); });
  for (auto tv : ref_tvs) {
    std::cout << "Ref tv: " << tv->toString() << std::endl;
  }
  for (auto tv : to_tvs) {
    std::cout << "Boundary tv: " << tv->toString() << std::endl;
  }
  std::cout << "Sharding: ";
  for (auto tv : tvs_to_shard) {
    std::cout << tv->toString() << std::endl;
  }
  shardAllLike(ref_tvs[0], tvs_to_shard, parallel_types);
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
  for (IterDomain* id : tv->getLoopDomain()) {
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
  for (const auto& tvs :
       {ir_utils::filterByType<TensorView>(expr->inputs()),
        ir_utils::filterByType<TensorView>(expr->outputs())}) {
    for (auto* tv : tvs) {
      NVF_ERROR(
          tv->hasDeviceMesh(),
          "the TensorView has no device mesh: ",
          tv->toString());
      auto& mesh = tv->getDeviceMesh().vector();
      std::copy(mesh.begin(), mesh.end(), std::inserter(ret, ret.end()));
    }
  }
  return ret;
}

int64_t getShardedAxis(TensorView* tv) {
  auto ids = TensorDomain::noReductions(tv->getLogicalDomain());
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i]->getParallelType() == ParallelType::DIDx) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

void reorderDIDToFront(TensorView* tv) {
  // new position to old position
  std::unordered_map<int64_t, int64_t> order_map;
  int64_t current_pos = 0;

  for (auto pos : c10::irange(tv->nDims())) {
    if (tv->axis(pos)->isDeviceDim()) {
      order_map[current_pos] = pos;
      current_pos++;
    }
  }

  for (auto pos : c10::irange(tv->nDims())) {
    if (!tv->axis(pos)->isDeviceDim()) {
      order_map[current_pos] = pos;
      current_pos++;
    }
  }

  tv->reorder(order_map);
}

} // namespace nvfuser

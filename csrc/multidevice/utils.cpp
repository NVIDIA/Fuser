// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/utils.h>
#include <id_model/id_model.h>
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
  NVF_THROW("Id", id->toString(), " is not in TensorView ", tv->toString());
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
  for (IterDomain* id : TensorDomain::noReductions(tv->getLoopDomain())) {
    if (!id->isDeviceDim()) {
      continue;
    }

    // Only one axis can be sharded on DIDx.
    NVF_ERROR(
        !is_sharded,
        "Multiple IterDomains parallelized on DIDx in TensorView ",
        tv);
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
    const TensorView* consumer,
    const IdModel& id_model) {
  // cpu scalars are not required to have a mesh
  if (producer->isCpuScalar() || consumer->isCpuScalar()) {
    return false;
  }

  // exit early in the unsharded case for performance
  if (!producer->hasDeviceMesh() && !consumer->hasDeviceMesh()) {
    return false;
  }

  // If device mesh are different, the Expr is resharding
  if (producer->getDeviceMesh() != consumer->getDeviceMesh()) {
    return true;
  }

  const std::unordered_map<IterDomain*, IterDomain*>& p2c =
      PairwiseLogicalDomainMap(producer, consumer).mapProducerToConsumer();
  std::unordered_set<Val*> mapped_p_logical_ids;
  mapped_p_logical_ids.reserve(p2c.size());
  std::unordered_set<Val*> mapped_c_root_ids;
  mapped_c_root_ids.reserve(p2c.size());
  for (IterDomain* p_logical_id : producer->getLogicalDomain()) {
    const auto i = p2c.find(p_logical_id);
    if (i == p2c.end()) {
      // This happens e.g. when `p_logical_id` is squeezed. Even if
      // `p_logical_id` is parallelized on DID, the squeezed dimension is size-1
      // and doesn't trigger resharding.
      continue;
    }
    mapped_p_logical_ids.insert(p_logical_id);
    mapped_c_root_ids.insert(i->second);
  }

  auto map_parallel_type_to_id = [](const std::vector<IterDomain*>& loop_domain)
      -> std::unordered_map<ParallelType, IterDomain*> {
    std::unordered_map<ParallelType, IterDomain*> parallel_type_to_id;
    parallel_type_to_id.reserve(kParallelTypeDIDs.size());
    for (IterDomain* loop_id : loop_domain) {
      const ParallelType parallel_type = loop_id->getParallelType();
      if (!isParallelTypeDeviceDim(parallel_type)) {
        continue;
      }

      NVF_ERROR(
          parallel_type_to_id.try_emplace(parallel_type, loop_id).second,
          "Found multiple loop IterDomains with the same parallel type (",
          parallel_type,
          "): ",
          toDelimitedString(loop_domain));
    }
    return parallel_type_to_id;
  };
  std::unordered_map<ParallelType, IterDomain*> p_parallel_type_to_id =
      map_parallel_type_to_id(producer->getLoopDomain());
  std::unordered_map<ParallelType, IterDomain*> c_parallel_type_to_id =
      map_parallel_type_to_id(consumer->getLoopDomain());

  auto depends_on = [](IterDomain* id,
                       const std::unordered_set<Val*>& dependencies) -> bool {
    const auto inputs = IterVisitor::getInputsTo(
        {id}, {dependencies.begin(), dependencies.end()});
    return std::any_of(inputs.begin(), inputs.end(), [&](Val* input) -> bool {
      return dependencies.count(input);
    });
  };

  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::PERMISSIVE);
  for (const auto parallel_type : kParallelTypeDIDs) {
    IterDomain* p_loop_id = getOrDefault(p_parallel_type_to_id, parallel_type);
    if (p_loop_id != nullptr && !depends_on(p_loop_id, mapped_p_logical_ids)) {
      p_loop_id = nullptr;
    }

    IterDomain* c_loop_id = getOrDefault(c_parallel_type_to_id, parallel_type);
    if (c_loop_id != nullptr && !depends_on(c_loop_id, mapped_c_root_ids)) {
      c_loop_id = nullptr;
    }

    if ((p_loop_id == nullptr) != (c_loop_id == nullptr)) {
      return true;
    }

    if (p_loop_id != nullptr) {
      NVF_ERROR(c_loop_id != nullptr);
      if (!exact_graph.disjointValSets().strictAreMapped(
              p_loop_id, c_loop_id)) {
        return true;
      }
    }
  }
  return false;
}

bool isResharding(const Expr* expr) {
  if (!ir_utils::isTvOp(expr)) {
    return false;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  IdModel id_model({const_cast<Expr*>(expr)}, {}, false, false);
  id_model.buildPermissiveGraph();
  // We don't use getTvsWithDifferentSharding because it creates a computeAtMap,
  // which is too costly
  for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      // exit early in the unsharded case for performance
      if (haveDifferentShardings(input, output, id_model)) {
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

void shardAllLike(TensorView* ref, std::vector<TensorView*> tvs) {
  for (auto tv : tvs) {
    tv->setDeviceMesh(ref->getDeviceMesh());
  }
  if (!tvs.empty()) {
    scheduler_utils::parallelizeAllLike(
        ref, tvs, {ParallelType::DIDx, ParallelType::Serial});
  }
}

void shardBetween(
    const std::vector<Expr*>& from,
    const std::vector<Expr*>& to,
    TensorView* ref) {
  std::vector<TensorView*> from_tvs;
  std::vector<TensorView*> to_tvs;
  for (auto expr : from) {
    auto outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    std::copy(outputs.begin(), outputs.end(), std::back_inserter(from_tvs));
  }

  for (auto expr : to) {
    auto outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    std::copy(outputs.begin(), outputs.end(), std::back_inserter(to_tvs));
  }

  shardBetween(from_tvs, to_tvs, ref);
}

void shardBetween(
    const std::vector<TensorView*>& from,
    const std::vector<TensorView*>& to,
    TensorView* ref) {
  std::unordered_set<TensorView*> boundary = {to.begin(), to.end()};
  for (auto tv : from) {
    auto expr = tv->definition();
    if (expr == nullptr) {
      continue;
    }
    auto inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    std::copy(
        inputs.begin(), inputs.end(), std::inserter(boundary, boundary.end()));
  }

  std::unordered_set<TensorView*> all_tvs =
      scheduler_utils::getAllTvsFrom(from, boundary);
  shardAllLike(ref, {all_tvs.begin(), all_tvs.end()});

  // Remove DID parallelizations on reduction axes.
  for (auto* tv : all_tvs) {
    for (IterDomain* id : tv->getLoopDomain()) {
      if (id->isReduction() && id->isDeviceDim()) {
        id->parallelize(ParallelType::Serial);
      }
    }
  }
}

int64_t requestedNumberOfDevices(Fusion* fusion) {
  DeviceIdxType max_index = 0;
  for (auto tv : fusion->allTvs()) {
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
  for (auto tv : fusion->allTvs()) {
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

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/utils.h>
#include <host_ir/lower.h>
#include <id_model/id_model.h>
#include <instrumentation.h>
#include <ir/internal_base_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
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

// Returns the position where an axis is allocated in a tv, skipping trivial
// dimensions (i.e. DID, reduction and broadcast). Returns -1 if id is not in
// tv's loop domain WAR: today we assume that the loop domain match with the
// actual allocation, but this will have to change in the future.
int64_t allocationIndex(TensorView* tv, IterDomain* id) {
  int64_t index = 0;
  for (auto* loop_id : tv->getLoopDomain()) {
    if (loop_id == id) {
      return index;
    }
    if (!loop_id->isDeviceDim() && !loop_id->isReduction() &&
        !loop_id->isBroadcast()) {
      index++;
    }
  }
  return -1;
}

} // namespace

std::pair<std::vector<IterDomain*>, std::vector<IterDomain*>> getShardingChanges(
    TensorView* producer,
    TensorView* consumer) {
  std::vector<IterDomain*> shard_additions;
  std::vector<IterDomain*> shard_deletions;
  auto rootmap =
      PairwiseLogicalDomainMap(producer, consumer).mapBroadcast(false);
  const auto c2p_map = rootmap.mapConsumerToProducer();

  for (IterDomain* out_root : consumer->getMaybeRootDomain()) {
    IterDomain* in_root = c2p_map.at(out_root);
    // Ignore sharded broadcast domains and
    // sharded reductions on the consumer
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
          " resharding ",
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
  for (IterDomain* alloc_id : tv->getMaybeAllocationDomain()) {
    if (!alloc_id->isDeviceDim()) {
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

namespace {
// Collect device-parallel IterDomains in `domain` and return them as a
// ParallelType-to-IterDomain map.
std::unordered_map<ParallelType, IterDomain*> mapDeviceParallelTypeToId(
    const std::vector<IterDomain*>& domain) {
  std::unordered_map<ParallelType, IterDomain*> parallel_type_to_id;
  parallel_type_to_id.reserve(kParallelTypeDIDs.size());
  for (IterDomain* id : domain) {
    const ParallelType parallel_type = id->getParallelType();
    if (!isParallelTypeDeviceDim(parallel_type)) {
      continue;
    }

    NVF_ERROR(
        parallel_type_to_id.try_emplace(parallel_type, id).second,
        "Found multiple loop IterDomains with the same parallel type (",
        parallel_type,
        "): ",
        toDelimitedString(domain));
  }
  return parallel_type_to_id;
}

std::unordered_map<IterDomain*, int64_t> mapIterDomainToTensorAxis(
    const std::vector<IterDomain*>& domain) {
  std::unordered_map<IterDomain*, int64_t> id_to_axis;
  int64_t axis = 0;
  for (auto* id : domain) {
    // Reduction IterDomains are not materialized as an at::Tensor axis.
    if (id->isReduction()) {
      continue;
    }
    id_to_axis[id] = axis;
    axis++;
  }
  return id_to_axis;
}

} // namespace

int64_t getShardedLogicalAxis(
    const TensorView* tv,
    const ParallelType parallel_type) {
  std::unordered_map<ParallelType, IterDomain*> parallel_type_to_id =
      mapDeviceParallelTypeToId(tv->getMaybeAllocationDomain());
  IterDomain* alloc_id = getOrDefault(parallel_type_to_id, parallel_type);
  if (alloc_id == nullptr) {
    return -1;
  }

  std::unordered_map<IterDomain*, int64_t> logical_id_to_axis =
      mapIterDomainToTensorAxis(tv->getLogicalDomain());
  IterDomain* id = alloc_id;
  while (logical_id_to_axis.count(id) == 0) {
    Expr* def = id->definition();
    NVF_ERROR(
        def != nullptr,
        "Failed to find a non-reduction logical IterDomain that produces ",
        alloc_id);
    if (auto* split = dynamic_cast<Split*>(def)) {
      // Returning just which tensor axis is sharded isn't sufficient to let
      // shardTensor, a user of this function, know how to shard the tensor.
      // For example,
      //
      //   t = makeContigConcreteTensor({6});
      //   t->split(0, 2, /*inner_split=*/true);
      //   t->axis(-1)->parallelize(DIDx);
      //   // [i{3}, iDIDx{2}]
      //
      // and the unsharded tensor is [0, 1, 2, 3, 4, 5], regardless of the
      // stride. The sharded tensor ought to be [0, 2, 4] for GPU 0 and [1, 3,
      // 5] for GPU 1. However, shardTensor as is will return [0, 1, 2] and [3,
      // 4, 5], assuming the axis is sharded outermost.
      //
      // One potential way to solve the general problem is to replay and rewind
      // the splits on the at::Tensor.  For example,
      //
      //   t = makeContigConcreteTensor({30});
      //   t->split(0, 5);
      //   t->split(0, 3);
      //   t->axis(0)->parallelize(Host);
      //   t->axis(1)->parallelize(DIDx);
      //   // [iHost{2}, iDIDx{3}, i{5}]
      //
      // Given an unsharded at::Tensor of shape [30], we'll first replay the
      // splits using `torch.view` to get a tensor of shape [2,3,5]. Then, we
      // `torch.slice` axis 1 for DIDx to get a tensor of shape [2,1,5]. Then,
      // we rewind the splits (and therefore apply merging) using
      // `torch.reshape` to get a sharded tensor of shape [10].
      NVF_ERROR(
          split->outer() == id,
          "Currently, we don't support DID on inner splits: ",
          split);
      id = split->in();
    } else if (auto* merge = dynamic_cast<Merge*>(def)) {
      // For example,
      //
      //   t = makeContigTensor(2);
      //   t->merge(0, 1);
      //   t->axis(0)->parallelize(DIDx);
      //
      // When `unshardedSizes` is given a local tensor of shape [1, 1], it's
      // unclear the global shape is [1, D] or [D, 1] or even [2, D/2], etc.
      NVF_THROW(
          "Failed to attribute the sharding to a single tensor axis and therefore bailed out: ",
          merge);
    } else {
      NVF_THROW(
          "Unexpected transforms from logical to a DID-parallel allocation IterDomain: ",
          def);
    }
  }

  return logical_id_to_axis.at(id);
}

at::Tensor shardTensor(
    at::Tensor tensor,
    const int64_t axis,
    const DeviceMesh& mesh,
    const DeviceIdxType device_id) {
  auto i = mesh.idxOf(device_id);
  auto extent = tensor.size(axis);
  auto nslices = mesh.size();
  NVF_CHECK(
      extent % nslices == 0, "Sharded axis must be evenly divisble by mesh");
  auto stride = extent / nslices;
  // TODO: returning slice 0 temporarily when device is not in the mesh.
  i = (i < 0) ? 0 : i;
  // The following slicing is problematic when DID is on an inner split (cf.
  // MultiDeviceTest.ShardTensor_InnerSplit). We currently disallow that and
  // it's enforced by getShardedLogicalAxis.
  return tensor.slice(axis, i * stride, (i + 1) * stride).contiguous();
}

std::vector<int64_t> unshardedSizes(
    const TensorView* tv,
    c10::IntArrayRef sizes) {
  std::vector<int64_t> unsharded_sizes = sizes.vec();
  for (ParallelType parallel_type : kParallelTypeDIDs) {
    const int64_t sharded_axis = getShardedLogicalAxis(tv, parallel_type);
    if (sharded_axis == -1) {
      continue;
    }
    unsharded_sizes.at(sharded_axis) *= tv->getDeviceMesh().size(parallel_type);
  }
  return unsharded_sizes;
}

int64_t numDeviceDims(const TensorView* tv) {
  return std::count_if(
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      [](IterDomain* id) { return id->isDeviceDim(); });
}

namespace {

std::vector<IterDomain*> getInputsInTargetDomain(
    IterDomain* loop_id,
    const std::vector<IterDomain*>& target_domain) {
  const std::vector<Val*> inputs_as_vals = IterVisitor::getInputsTo(
      {loop_id}, {target_domain.begin(), target_domain.end()});

  std::vector<IterDomain*> inputs_as_iter_domains;
  inputs_as_iter_domains.reserve(inputs_as_vals.size());
  std::transform(
      inputs_as_vals.begin(),
      inputs_as_vals.end(),
      std::back_inserter(inputs_as_iter_domains),
      [](Val* val) { return val->as<IterDomain>(); });
  return inputs_as_iter_domains;
}

bool overlaps(
    const std::vector<IterDomain*>& a,
    const std::unordered_set<IterDomain*>& b) {
  return std::any_of(
      a.begin(), a.end(), [&](IterDomain* id) { return b.count(id); });
}

} // namespace

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

  // The rest of this function tries to do the following: for each pair of
  // logical-domain-mapped IterDomains (i.e. those mapped by
  // PairwiseLogicalDomainMap), check if they are sharded consistently. If not,
  // returns true. For example,
  //
  //   a: iDIDx{M}, iK
  //   b: iK, iDIDy{N}
  //   c = matmul(a, b): iDIDx{M}, iDIDy{N}
  //
  // haveDifferentShardings(a, c) only cares about iM, which is
  // logical-domain-mapped, but not iK or iN, which are not
  // logical-domain-mapped.
  //
  // One challenge is that DID parallelization doesn't always
  // happen on the root/logical IterDomains. For example, a root/logical
  // IterDomain may be outer-split by the number of devices, and only the outer
  // split gets parallelized on DID.
  //
  //   logical: iM
  //   loop: iDIDx{D}, iM/D
  //
  // Therefore, we collect all the loop IterDomains that depend on the
  // logical-domain-mapped IterDomains, and check if they are DID-parallelized
  // consistently.
  const std::unordered_map<IterDomain*, IterDomain*>& p2c =
      PairwiseLogicalDomainMap(producer, consumer).mapProducerToConsumer();
  std::unordered_set<IterDomain*> mapped_p_logical_ids;
  mapped_p_logical_ids.reserve(p2c.size());
  std::unordered_set<IterDomain*> mapped_c_root_ids;
  mapped_c_root_ids.reserve(p2c.size());
  for (IterDomain* p_logical_id : producer->getLogicalDomain()) {
    const auto i = p2c.find(p_logical_id);
    if (i == p2c.end()) {
      // This happens e.g. when `p_logical_id` is squeezed or is a product of a
      // reduction. Even if `p_logical_id` is parallelized on DID, the
      // dimension is size-1 and doesn't trigger resharding.
      continue;
    }
    mapped_p_logical_ids.insert(p_logical_id);
    mapped_c_root_ids.insert(i->second);
  }

  // In practice, only loop IterDomains can be parallelized, and no two loop
  // IterDomains in a TensorView can have the same parallel type. Therefore, we
  // do the check in reverse order for efficiency and simplicity:
  // 1. For each DID parallel type, find the loop IterDomain in producer and the
  // one in consumer that have the type.
  // 2. Find what IterDomains they come from in producer's logical or
  // consumer's root domain. If that input IterDomain is not
  // logical-domain-mapped, treat the loop IterDomain as not existing -- it is
  // parallelized but just not a concern for this producer-consumer pair.
  // 3. Check if the two loop IterDomains are almost-exactly mapped in the
  // IdModel.
  std::unordered_map<ParallelType, IterDomain*> p_parallel_type_to_id =
      mapDeviceParallelTypeToId(producer->getLoopDomain());
  std::unordered_map<ParallelType, IterDomain*> c_parallel_type_to_id =
      mapDeviceParallelTypeToId(consumer->getLoopDomain());

  for (const auto parallel_type : kParallelTypeDIDs) {
    IterDomain* p_loop_id = getOrDefault(p_parallel_type_to_id, parallel_type);
    if (p_loop_id != nullptr) {
      auto p_inputs =
          getInputsInTargetDomain(p_loop_id, producer->getLogicalDomain());
      if (!overlaps(p_inputs, mapped_p_logical_ids)) {
        p_loop_id = nullptr;
      }
    }

    IterDomain* c_loop_id = getOrDefault(c_parallel_type_to_id, parallel_type);
    if (c_loop_id != nullptr) {
      auto c_inputs =
          getInputsInTargetDomain(c_loop_id, consumer->getMaybeRootDomain());
      if (!overlaps(c_inputs, mapped_c_root_ids)) {
        c_loop_id = nullptr;
      }
    }

    auto is_mapped_in_id_model =
        [](IterDomain* a, IterDomain* b, const IdModel& id_model) -> bool {
      if (a == nullptr && b == nullptr) {
        return true;
      }

      if (a == nullptr || b == nullptr) {
        return false;
      }

      // Going between bDIDx{1} and iDIDx{N} doesn't trigger resharding, but
      // would be flagged by ALMOSTEXACT as a false positive.
      if (id_model.idGraph(IdMappingMode::BROADCAST)
              .disjointValSets()
              .strictAreMapped(a, b)) {
        return true;
      }

      // Check ALMOSTEXACT so iDIDx{N}*b{1} and iDIDx{N} are mapped.
      return id_model.idGraph(IdMappingMode::ALMOSTEXACT)
          .disjointValSets()
          .strictAreMapped(a, b);
    };

    if (!is_mapped_in_id_model(p_loop_id, c_loop_id, id_model)) {
      return true;
    }
  }

  return false;
}

bool isResharding(const Expr* expr) {
  FUSER_PERF_SCOPE("isResharding");

  if (!ir_utils::isTvOp(expr)) {
    return false;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  IdModel id_model({const_cast<Expr*>(expr)}, {}, false, false);
  id_model.buildAlmostExactGraph();
  id_model.buildBroadcastGraph();
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

  for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      auto [shard_additions, shard_deletions] =
          getShardingChanges(input, output);
      NVF_ERROR(
          shard_additions.size() + shard_deletions.size() <= 1,
          "Resharding expr can only support one axis")
      if ((!shard_deletions.empty() &&
           allocationIndex(input, shard_deletions.at(0)) > 0) ||
          (!shard_additions.empty() &&
           allocationIndex(output, shard_additions.at(0)) > 0)) {
        return true;
      }
    }
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
      if (tv->hasDeviceMesh()) {
        auto& mesh = tv->getDeviceMesh().vector();
        std::copy(mesh.begin(), mesh.end(), std::inserter(ret, ret.end()));
      } else {
        ret.insert(0);
      }
    }
  }
  return ret;
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

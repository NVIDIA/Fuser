// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/utils.h>
#include <expr_simplifier.h>
#include <host_ir/lower.h>
#include <instrumentation.h>
#include <ir/container.h>
#include <ir/internal_base_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <statement_guard.h>
#include <transform_replay.h>

namespace nvfuser {

NVF_API bool distributedEnabled() {
#ifdef NVFUSER_DISTRIBUTED
  return true;
#else
  return false;
#endif
}

int64_t allocationIndex(TensorView* tv, IterDomain* id) {
  int64_t index = 0;
  for (auto* alloc_id : tv->getMaybeAllocationDomain()) {
    if (alloc_id == id) {
      return index;
    }
    if (alloc_id->isDeviceDim() || alloc_id->isReduction() || alloc_id->isBroadcast()) {
      continue;
    }
    index++;
  }
  return -1;
}

} // namespace

bool isTvContiguous(const TensorView* tv) {
  // Reduction and broadcast axis do not have a contiguity value.
  return std::all_of(
      tv->getContiguity().begin(),
      tv->getContiguity().end(),
      [](std::optional<bool> c) { return c.value_or(true); });
}

std::optional<std::pair<IterDomain*, IterDomain*>> getReshardingIdPair(Expr* expr, const ComputeAtMap& ca_map) {
  NVF_ERROR(expr->isOneOf<ReductionOp, LoadStoreOp>(), "Expected a reduction or load/store operator.");

  auto producer = expr->inputs()[0]->as<TensorView>();
  auto consumer = expr->outputs()[0]->as<TensorView>();

std::optional<std::pair<IterDomain*, IterDomain*>> getReshardingIdPair(TensorView* producer, TensorView* consumer, ValGraph& graph) {
  auto p_loop_domain = producer->getLoopDomain();
  auto c_loop_domain = consumer->getLoopDomain();
  auto p2c_map = graph.buildMapBetween(
            p_loop_domain, c_loop_domain);

  std::vector<std::pair<IterDomain*, IterDomain*>> resharding_id_pairs;

  bool has_sharding_changes = false;

  IterDomain* resharded_p_id = nullptr;
  IterDomain* resharded_c_id = nullptr;

  for (auto [p_val, c_vals] : p2c_map) {
    auto p_id = p_val->as<IterDomain>();
    auto c_id = c_vals.front()->as<IterDomain>();

    if (!p_id->isDeviceDim() && !c_id->isDeviceDim()) {
      continue;
    }

    // No reordering for reduction and broadcast axes.
    if (p_id->isReduction() || p_id->isBroadcast()) {
      continue;
    }
    if (c_id->isReduction() || c_id->isBroadcast()) {
      continue;
    }

    if (p_id->isDeviceDim() && c_id->isDeviceDim()) {
      if (p_id->getParallelType() != c_id->getParallelType()) {
        NVF_THROW("Resharding from ", p_id->toString(), " to ", c_id->toString(), " is not supported.");
      }
    }

    NVF_ERROR(!has_sharding_changes, "Resharding expr can only support one axis: ", consumer->definition()->toString());
    has_sharding_changes = true;
    resharded_p_id = p_id;
    resharded_c_id = c_id;
  }

  if (!has_sharding_changes) {
    return std::nullopt;
  }
  
  return std::make_pair(resharded_p_id, resharded_c_id);
}

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

    // Reduction dimensions are not materialized in the concrete tensor, so we
    // don't consider rDIDx{i0} sharded. For example,
    //
    //   ```
    //   [iDIDx{i0}, iS{i1}] => [rDIDx{i0}, iS{i1}]
    //   ```
    //
    // is considered an allreduce and the output is replicated.
    if (alloc_id->isReduction()) {
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

    // rDIDx{i0}, usually a product of an Allreduce or a ReduceScatter, is
    // treated as replicated. This way `iDIDx{i0} => rDIDx{i0}` is considered
    // resharding.
    if (id->isReduction()) {
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
    if (id->isReduction()) {
      // Reduction IterDomains are not materialized as an at::Tensor axis.
      id_to_axis[id] = -1;
    } else {
      id_to_axis[id] = axis;
      axis++;
    }
  }
  return id_to_axis;
}

} // namespace

int64_t getShardedLogicalAxisFromDomain(
    const TensorView* tv,
    const ParallelType parallel_type,
    std::vector<IterDomain*> domain) {
  std::unordered_map<ParallelType, IterDomain*> parallel_type_to_id =
      mapDeviceParallelTypeToId(domain);
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

int64_t getShardedLogicalAxis(
    const TensorView* tv,
    const ParallelType parallel_type) {
  return getShardedLogicalAxisFromDomain(tv, parallel_type, tv->getMaybeAllocationDomain());
}

int64_t getShardedLoopAxis(
    const TensorView* tv,
    const ParallelType parallel_type) {
  NVF_ERROR(
      isParallelTypeDeviceDim(parallel_type),
      "Expect a DID but found: ",
      parallel_type);
  for (auto&& [index, loop_id] : enumerate(tv->getLoopDomain())) {
    if (loop_id->getParallelType() == parallel_type) {
      return index;
    }
  }
  return -1;
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
      [](IterDomain* id) { return id->isDeviceDim() && !id->isReduction(); });
}

namespace {
// Given a loop ID `id` and a source domain `sources`, returns the Val* that
// represents the index of that loop ID. `sources` is either the producer's
// logical or the consumer's root. The boolean returned indicates whether the
// loop ID depends on a producer logical ID or a consumer root ID that are
// mapped by PairwiseLogicalDomainMap. Recall that the caller only examines DIDs
// that originates from a mapped ID. `id_to_index` operates as a cache.
std::pair<Val*, bool> computeLoopIndex(
    IterDomain* id,
    const std::vector<IterDomain*>& sources,
    std::unordered_map<IterDomain*, std::pair<Val*, bool>>& id_to_index) {
  if (id == nullptr) {
    return {nullptr, false};
  }

  std::vector<Expr*> transforms =
      StmtSort::getExprsBetween({sources.begin(), sources.end()}, {id});
  for (Expr* transform : transforms) {
    if (std::all_of(
            transform->outputs().begin(),
            transform->outputs().end(),
            [&](Val* val) {
              return id_to_index.count(val->as<IterDomain>()) > 0;
            })) {
      continue;
    }

    if (auto* split = dynamic_cast<Split*>(transform)) {
      auto* in = split->in()->as<IterDomain>();
      auto* outer = split->outer()->as<IterDomain>();
      auto* inner = split->inner()->as<IterDomain>();

      const auto& in_info = id_to_index.at(in);
      id_to_index[outer] = {
          div(in_info.first, inner->extent()), in_info.second};
      id_to_index[inner] = {
          mod(in_info.first, inner->extent()), in_info.second};
    } else if (auto* merge = dynamic_cast<Merge*>(transform)) {
      auto* outer = merge->outer()->as<IterDomain>();
      auto* inner = merge->inner()->as<IterDomain>();
      auto* out = merge->out()->as<IterDomain>();

      const auto& outer_info = id_to_index.at(outer);
      const auto& inner_info = id_to_index.at(inner);
      id_to_index[out] = {
          add(mul(outer_info.first, inner->extent()), inner_info.first),
          outer_info.second || inner_info.second};
    } else {
      NVF_THROW("Unexpected transform: ", transform);
    }
  }

  return id_to_index.at(id);
}

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

} // namespace

bool haveDifferentShardings(
    const TensorView* producer,
    const TensorView* consumer) {
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

  // Special handling of SelectOp for a quick fix
  // TODO: work on a proper implementation
  if (consumer->definition()->isA<SelectOp>()) {
    auto* select_op = consumer->definition()->as<SelectOp>();
    NVF_ERROR(
        select_op->input(0) == producer, "SelectOp input 0 is not producer");
    // If we select into the sharded axis, the op is resharding because the
    // axis doesn't exist in the consumer and so becomes "replicated".
    //
    // tv0 = makeContigTensor(2); // [DIDx(4), 8] on mesh {0,1,2,3}
    // tv1 = select(tv0, /*axis=*/0, /*index=*/1); // [8] on mesh {0,1,2,3}
    //
    // The long term better solution would actually to "select" into the
    // DeviceMesh, e.g.,
    //
    // tv0 = makeContigTensor(2); // [DIDx(4), 8] on mesh {0,1,2,3}
    // tv1 = select(tv0, /*axis=*/0, /*index=*/1); // [8] on mesh {1}
    // But for achieving this with symbolic "index" we need to make DeviceMesh
    // symbolic.
    if (select_op->getIndexedID()->isDeviceDim()) {
      return true;
    }
    // If the sharded axis is not selected into, then we still need to check
    // that other axis do not get resharded.
    const std::unordered_map<IterDomain*, IterDomain*>& c2p =
        PairwiseLogicalDomainMap(producer, consumer)
            .mapBroadcast(false)
            .mapConsumerToProducer();
    return !std::all_of(
        consumer->getLoopDomain().begin(),
        consumer->getLoopDomain().end(),
        [&c2p](IterDomain* c_id) {
          auto p_id = c2p.at(c_id);
          return c_id->isDeviceDim() == p_id->isDeviceDim();
        });
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
  const std::unordered_map<IterDomain*, IterDomain*>& c2p =
      PairwiseLogicalDomainMap(producer, consumer)
          // We skip broadcast dimensions because they are replicated on all
          // devices regardless of DIDx. Even when the corresponding consumer
          // dimension is non-broadcast, they don't cause communication. If we
          // didn't skip them, we would need to modify the downstream code for
          // collecting assumptions of `index < extent`. Recall that
          // non-expanded broadcast dimensions have a fixed extent of 1.
          .mapBroadcast(false)
          .mapConsumerToProducer();

  auto c2p_values = std::views::values(c2p);
  std::unordered_set<IterDomain*> mapped_p_logical_ids(
      c2p_values.begin(), c2p_values.end());

  Fusion* fusion = producer->fusion();
  NVF_ERROR(
      fusion == consumer->fusion(),
      "The producer and consumer must be in the same fusion.");
  FusionGuard fg(fusion);
  StatementGuard sg(fusion);

  // The second element of the value pair indicates whether the IterDomain
  // depends on a mapped producer logical IterDomain or a mapped consumer root
  // IterDomain. Propagating this information is needed to solve the matmul
  // example above.
  std::unordered_map<IterDomain*, std::pair<Val*, bool>> id_to_index;
  std::vector<Val*> assumptions;
  assumptions.reserve(
      (producer->getLogicalDomain().size() +
       consumer->getMaybeRootDomain().size()) *
      2);

  auto create_index = [&](IterDomain* id, bool mapped) {
    auto* index = IrBuilder::create<Val>(DataType::Index);
    NVF_ERROR(id_to_index.emplace(id, std::make_pair(index, mapped)).second);
    assumptions.push_back(
        SimplifyingIrBuilder::leExpr(fusion->zeroVal(), index));
    assumptions.push_back(SimplifyingIrBuilder::ltExpr(index, id->extent()));
  };

  // Create indices for producer logical IDs and consumer root IDs. As an
  // optimization, we create indices only for those that DIDs depend on.
  std::unordered_map<ParallelType, IterDomain*> p_parallel_type_to_id =
      mapDeviceParallelTypeToId(producer->getLoopDomain());
  std::unordered_map<ParallelType, IterDomain*> c_parallel_type_to_id =
      mapDeviceParallelTypeToId(consumer->getLoopDomain());
  for (const auto parallel_type : kParallelTypeDIDs) {
    if (IterDomain* p_loop_id =
            getOrDefault(p_parallel_type_to_id, parallel_type)) {
      for (IterDomain* p_logical_id :
           getInputsInTargetDomain(p_loop_id, producer->getLogicalDomain())) {
        if (id_to_index.count(p_logical_id) > 0) {
          continue;
        }

        create_index(p_logical_id, mapped_p_logical_ids.count(p_logical_id));
      }
    }
  }

  for (const auto parallel_type : kParallelTypeDIDs) {
    if (IterDomain* c_loop_id =
            getOrDefault(c_parallel_type_to_id, parallel_type)) {
      for (IterDomain* c_root_id :
           getInputsInTargetDomain(c_loop_id, consumer->getMaybeRootDomain())) {
        if (id_to_index.count(c_root_id) > 0) {
          continue;
        }

        IterDomain* p_logical_id = getOrDefault(c2p, c_root_id);
        if (p_logical_id == nullptr) {
          create_index(c_root_id, /*mapped=*/false);
          continue;
        }

        auto i = id_to_index.find(p_logical_id);
        if (i == id_to_index.end()) {
          create_index(c_root_id, /*mapped=*/true);
          continue;
        }
        // Reuse the same index as the mapped producer logical ID. This is
        // necessary for proving is-non-resharding; otherwise we won't see any
        // connections between producer and consumer's loop indices.
        NVF_ERROR(id_to_index
                      .emplace(c_root_id, std::make_pair(i->second.first, true))
                      .second);
      }
    }
  }

  // For each parallel type, check whether the corresponding loop index in the
  // producer and that in the consumer are equivalent. If they can't be proven
  // to be equivalent, return is-resharding.
  for (const auto parallel_type : kParallelTypeDIDs) {
    IterDomain* p_id = getOrDefault(p_parallel_type_to_id, parallel_type);
    Val* p_index = nullptr;
    bool p_mapped = false;
    std::tie(p_index, p_mapped) =
        computeLoopIndex(p_id, producer->getLogicalDomain(), id_to_index);
    if (!p_mapped) {
      p_index = nullptr;
    }

    IterDomain* c_id = getOrDefault(c_parallel_type_to_id, parallel_type);
    Val* c_index = nullptr;
    bool c_mapped = false;
    std::tie(c_index, c_mapped) =
        computeLoopIndex(c_id, consumer->getMaybeRootDomain(), id_to_index);
    if (!c_mapped) {
      c_index = nullptr;
    }

    const bool is_equivalent = [&]() -> bool {
      if (p_index == nullptr && c_index == nullptr) {
        return true;
      }

      if (p_index == nullptr || c_index == nullptr) {
        return false;
      }

      return simplifyExpr(
                 SimplifyingIrBuilder::eqExpr(p_index, c_index),
                 /*variables=*/{},
                 assumptions)
          ->isTrue();
    }();

    if (!is_equivalent) {
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

  // We don't use getTvsWithDifferentSharding because it creates a computeAtMap,
  // which is too costly
  for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
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

std::unordered_set<ParallelType> deviceAndStreamParallelTypes() {
  static auto s = [&] {
    std::unordered_set<ParallelType> s(
        {kParallelTypeDIDs.begin(), kParallelTypeDIDs.end()});
    s.insert(ParallelType::Stream);
    return s;
  }();
  return s;
}

void shardAllLike(
    TensorView* ref,
    const std::vector<TensorView*>& tvs,
    const std::unordered_set<ParallelType>& parallel_types) {
  if (tvs.empty()) {
    return;
  }
  for (auto tv : tvs) {
    tv->setDeviceMesh(ref->getDeviceMesh());
  }
  scheduler_utils::parallelizeAllLike(ref, tvs, parallel_types);
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
  shardAllLike(
      ref, {all_tvs.begin(), all_tvs.end()}, deviceAndStreamParallelTypes());
}

int64_t requestedNumberOfDevices(Fusion* fusion) {
  DeviceIdxType max_index = 0;
  for (auto tv : fusion->allTvs()) {
    if (tv->hasDeviceMesh()) {
      max_index = std::max(max_index, tv->getDeviceMesh().maxDeviceId());
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

std::unordered_map<int64_t, int64_t> reorderDIDToFront(TensorView* tv) {
  // old position to new position
  std::unordered_map<int64_t, int64_t> order_map;
  int64_t current_pos = 0;

  for (auto pos : arange(tv->nDims())) {
    if (tv->axis(pos)->isDeviceDim()) {
      order_map[pos] = current_pos;
      current_pos++;
    }
  }

  tv->reorder(order_map);
  return order_map;
}

std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    const std::vector<TensorView*>& tvs) {
  std::unordered_set<TensorView*> ret;
  const auto& reference_dom = ref->getLoopDomain();
  FusionGuard fg(ref->fusion());
  auto ca_map = ComputeAtMap(FusionGuard::getCurFusion());
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;
  for (auto id : reference_dom) {
    auto ca_id =
        ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
    concrete_to_reference_map[ca_id] = id;
  }

  for (TensorView* tv : tvs) {
    if (ref->getDeviceMesh().vector() != tv->getDeviceMesh().vector()) {
      ret.insert(tv);
      continue;
    }
    for (auto id : tv->getLoopDomain()) {
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

void propagateDIDTransform(
    const TensorView* ref,
    const std::vector<TensorView*>& tvs,
    int64_t did_pos,
    PropagateDirection direction) {
  TensorDomain* replayed_domain = nullptr;
  for (TensorView* tv : tvs) {
    if (direction == PropagateDirection::kForward) {
      replayed_domain = TransformReplay::replayCasP(tv, ref, did_pos).first;
    } else {
      replayed_domain = TransformReplay::replayPasC(tv, ref, did_pos).first;
    }
    tv->setLoopDomain(replayed_domain->loop());
  }
}

} // namespace nvfuser

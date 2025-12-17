// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "multidevice/resharding.h"

#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include "device_lower/utils.h"
#include "expr_simplifier.h"
#include "fusion_guard.h"
#include "instrumentation.h"
#include "ir/builder.h"
#include "ir/container.h"
#include "ir/internal_base_nodes.h"
#include "ir/internal_nodes.h"
#include "ir/utils.h"
#include "logical_domain_map.h"
#include "multidevice/utils.h"
#include "ops/all_ops.h"
#include "statement_guard.h"
#include "transform_replay.h"
#include "type.h"

namespace nvfuser {

namespace {

const std::vector<IterDomain*>& getDomainOf(
    const TensorView* tv,
    DomainType domain_type) {
  switch (domain_type) {
    case DomainType::kRoot:
      return tv->getMaybeRootDomain();
    case DomainType::kLogical:
      return tv->getLogicalDomain();
    case DomainType::kLoop:
      return tv->getLoopDomain();
    case DomainType::kAllocation:
      return tv->getMaybeAllocationDomain();
  }
  std::unreachable();
}

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

} // namespace

bool haveDifferentShardings(
    const TensorView* producer,
    DomainType producer_domain_type,
    const TensorView* consumer,
    DomainType consumer_domain_type,
    const std::unordered_set<ParallelType>& parallel_types) {
  // cpu scalars are not parallelized
  if (producer->isCpuScalar() || consumer->isCpuScalar()) {
    return false;
  }

  // exit early in the unsharded case for performance if we are
  // not checking for `Stream`.
  if (!producer->hasDeviceMesh() && !consumer->hasDeviceMesh() &&
      !parallel_types.count(ParallelType::Stream)) {
    return false;
  }

  // If device mesh are different, the Expr is resharding if parallel_types
  // includes DIDs
  if (std::any_of(
          kParallelTypeDIDs.begin(),
          kParallelTypeDIDs.end(),
          [&](ParallelType pt) { return parallel_types.count(pt); }) &&
      producer->getDeviceMesh() != consumer->getDeviceMesh()) {
    return true;
  }

  const auto& producer_domain = getDomainOf(producer, producer_domain_type);
  const auto& consumer_domain = getDomainOf(consumer, consumer_domain_type);

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

    auto indexed_id_pt = select_op->getIndexedID()->getParallelType();
    if (parallel_types.count(indexed_id_pt)) {
      return true;
    }
    // If the sharded axis is not selected into, then we still need to check
    // that other axis do not get resharded.
    const std::unordered_map<IterDomain*, IterDomain*>& c2p =
        PairwiseLogicalDomainMap(producer, consumer)
            .mapBroadcast(false)
            .mapConsumerToProducer();
    return !std::all_of(
        consumer_domain.begin(),
        consumer_domain.end(),
        [&c2p, &parallel_types](IterDomain* c_id) {
          auto p_id = c2p.at(c_id);
          auto p_id_pt = p_id->getParallelType();
          auto c_id_pt = c_id->getParallelType();
          if (parallel_types.count(p_id_pt) || parallel_types.count(c_id_pt)) {
            return p_id_pt == c_id_pt;
          }
          return true;
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
  // optimization, we create indices only for those that parallel_types depend
  // on.
  std::unordered_map<ParallelType, IterDomain*> p_parallel_type_to_id =
      mapDeviceAndStreamParallelTypeToId(producer_domain);
  std::unordered_map<ParallelType, IterDomain*> c_parallel_type_to_id =
      mapDeviceAndStreamParallelTypeToId(consumer_domain);
  for (const auto parallel_type : parallel_types) {
    if (IterDomain* p_loop_id =
            getOrDefault(p_parallel_type_to_id, parallel_type)) {
      for (IterDomain* p_logical_id :
           getInputsInTargetDomain({p_loop_id}, producer->getLogicalDomain())) {
        if (id_to_index.count(p_logical_id) > 0) {
          continue;
        }

        create_index(p_logical_id, mapped_p_logical_ids.count(p_logical_id));
      }
    }
  }

  for (const auto parallel_type : parallel_types) {
    if (IterDomain* c_loop_id =
            getOrDefault(c_parallel_type_to_id, parallel_type)) {
      for (IterDomain* c_root_id : getInputsInTargetDomain(
               {c_loop_id}, consumer->getMaybeRootDomain())) {
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
  for (const auto parallel_type : parallel_types) {
    IterDomain* p_id = getOrDefault(p_parallel_type_to_id, parallel_type);
    Val* p_index = nullptr;
    bool p_mapped = false;
    std::tie(p_index, p_mapped) = computeLoopIndex(
        p_id, getDomainOf(producer, DomainType::kLogical), id_to_index);
    if (!p_mapped) {
      p_index = nullptr;
    }

    IterDomain* c_id = getOrDefault(c_parallel_type_to_id, parallel_type);
    Val* c_index = nullptr;
    bool c_mapped = false;
    std::tie(c_index, c_mapped) = computeLoopIndex(
        c_id, getDomainOf(consumer, DomainType::kRoot), id_to_index);
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

bool haveDifferentShardings(
    const TensorView* producer,
    const TensorView* consumer,
    const std::unordered_set<ParallelType>& parallel_types) {
  return haveDifferentShardings(
      producer, DomainType::kLoop, consumer, DomainType::kLoop, parallel_types);
}

bool isResharding(const Expr* expr) {
  FUSER_PERF_SCOPE("isResharding");

  if (expr == nullptr || !ir_utils::isTvOp(expr)) {
    return false;
  }

  // We don't use getTvsWithDifferentSharding because it creates a computeAtMap,
  // which is too costly
  for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      if (haveDifferentShardings(input, output, deviceParallelTypes())) {
        return true;
      }
    }
  }

  return false;
}

} // namespace nvfuser

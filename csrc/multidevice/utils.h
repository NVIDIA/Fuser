// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <multidevice/multidevice.h>
#include <visibility.h>

namespace nvfuser {

// Returns true iff nvFuser was compiled with distributed APIs enabled.
NVF_API bool distributedEnabled();

// For a resharding expression, either a set or reduce, returns root IDs
// that change sharding.
// (1) sharded root IterDomains that are added by the expression
// i.e. sharded IterDomains that are present in the output, but not the input.
// (2) sharded root IterDomains that are removed by the expression
// i.e. sharded IterDomains that are present in the input, but not the output.
// TODO: Analyze loop domain for unsharded/sharded IDs and return their
// parent root IDs.
std::pair<std::vector<IterDomain*>, std::vector<IterDomain*>> getShardingChanges(
    Expr* expr);

// Returns whether a TensorView has a non-reduction axis parallelized Didx
// Checks that the other non-reduction axis are not parallelized on Didx
bool isSharded(const TensorView*);

// Returns number of device dimensions in a TensorView's loop domain.
int64_t numDeviceDims(const TensorView*);

// Returns the subset of tvs which elements have the different multi-device
// sharding as ref
template <typename TvIterator>
std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    TvIterator tvs) {
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

// Returns whether an Expr embeds multi-device resharding
bool isResharding(const Expr* expr);

// Returns whether two tensors have different shardings. Expect a
// producer/consumer relationship between the arguments.
bool haveDifferentShardings(
    const TensorView* producer,
    const TensorView* consumer);

// Returns whether a resharding expr reshards an inner axis
bool isInnerResharding(Expr* expr);

// Shards all tensors in tvs like reference
void shardAllLike(TensorView* ref, std::vector<TensorView*> tvs);

// Shards all TVs between from and to AND between TVs created inside a fusion
// and to. This is required for (1) expressions like rng_uniform that create a
// TV inside a fusion that is not between a path from user visible TVs. (2)
// multi-output expressions may have output tensors that are not along a path to
// the fusion output which would not be reachable otherwise. (2) sharding
// propagation checks all TVs in the fusion are assigned a device mesh
// regardless if they are reachable. To keep the checks simple, we require all
// TVs are assigned a mesh if they exist in the fusion.
void shardBetween(
    const std::vector<TensorView*>& from,
    const std::vector<TensorView*>& to,
    TensorView* ref);
// Same as above but using the outputs of the from and to expressions
// to form the from and to TVs.
void shardBetween(
    const std::vector<Expr*>& from,
    const std::vector<Expr*>& to,
    TensorView* ref);

// Returns the devices involved in an expr
std::set<DeviceIdxType> involvedDevices(Expr* expr);

// Returns the number of device indices present accross all
// device meshes in the Fusion
int64_t requestedNumberOfDevices(Fusion*);

// remove the multi-device scheduling annotations
void unshard(Fusion*);
void unshard(TensorView*);

// Returns the index of the a sharded axis if none return -1.
// TODO: Assumes no merges/splits on sharded axis.
int64_t getShardedAxis(TensorView*);

// Reorders a TensorView so that the DID parallelized axis are in front.
void reorderDIDToFront(TensorView*);
} // namespace nvfuser

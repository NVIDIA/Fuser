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

// Returns whether a TensorView has a non-reduction axis parallelized Didx
// Checks that the other non-reduction axis are not parallelized on Didx
NVF_API bool isSharded(TensorView*);

// Returns number of device dimensions in a TensorView's leaf domain.
int64_t numDeviceDims(TensorView*);

// Returns the subset of tvs which elements have the different multi-device
// sharding as ref
template <typename TvIterator>
std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    TvIterator tvs) {
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

  for (TensorView* tv : tvs) {
    if (ref->getDeviceMesh().vector() != tv->getDeviceMesh().vector()) {
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

// Returns whether an Expr embeds multi-device resharding
bool isResharding(Expr* expr);

// Returns whether a resharding expr reshards an inner axis
bool isInnerResharding(Expr* expr);

// Returns the devices involved in an expr
std::set<DeviceIdxType> involvedDevices(Expr* expr);

// Returns the number of device indices present accross all
// device meshes in the Fusion
int64_t requestedNumberOfDevices(Fusion*);

// remove the multi-device scheduling annotations
void unshard(Fusion*);
void unshard(TensorView*);

// TODO: Re-implement a robust and smatert sharding propagation pass.
// Very simple sharding propagation pass that identifies tvs without
// a DeviceMesh and shards it like its first producer tv with a sharding.
// This assumes that all global inputs are sharded.
// This cannot be done when the Op is inserted into the fusion, because
// the multidevice shcheduling hasn't been applied.
void propagateShardings(Fusion* fusion);

// Runs through the fusion and inserts a resharding Set Op after
// any resharding Expr that is not directly lowerable to a series of
// communications
void insertReshardings(Fusion* fusion);

// This can only run after the insertResharding passes.
// Assumes all resharding ops are either a set or reduction.
// For each resharding operation that requires communication
// over a noncontiguous slices of the tensor, this pass
// inserts permutations necessary to push the device parallel axis
// to the front so that communication operations are contiguous.
void insertShardedAxisReordering(Fusion* fusion);

// Returns the index of the a sharded axis if none return -1.
// TODO: Assumes no merges/splits on sharded axis.
int64_t getShardedAxis(TensorView*);

} // namespace nvfuser

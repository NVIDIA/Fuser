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
#include <scheduler/utils.h>
#include <visibility.h>

namespace nvfuser {

// Returns whether a TensorView has a non-reduction axis parallelized Didx
// Checks that the other non-reduction axis are not parallelized on Didx
bool isSharded(const TensorView*);

// Returns number of device dimensions in a TensorView's loop domain.
int64_t numDeviceDims(const TensorView*);

std::unordered_set<IterDomain*> getInputsInTargetDomain(
    const std::vector<IterDomain*>& loop_ids,
    const std::vector<IterDomain*>& target_domain);

// Returns the subset of tvs which elements have the different multi-device
// sharding as ref
std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    const std::vector<TensorView*>& tvs);

// Returns whether an Expr embeds multi-device resharding
NVF_API bool isResharding(const Expr* expr);

// Returns whether two tensors have different shardings. Expect a
// producer/consumer relationship between the arguments.
bool haveDifferentShardings(
    const TensorView* producer,
    const TensorView* consumer,
    const std::unordered_set<ParallelType>& parallel_types);

// Returns a set that contains DIDs and Stream.
std::unordered_set<ParallelType> deviceAndStreamParallelTypes();

std::unordered_set<ParallelType> deviceParallelTypes();

// Collect device and stream parallelized IterDomains in `domain` and return
// them as a ParallelType-to-IterDomain map. Excludes reduction iterdomains.
std::unordered_map<ParallelType, IterDomain*> mapDeviceAndStreamParallelTypeToId(
    const std::vector<IterDomain*>& domain);

// Shards all tensors in tvs like reference.
// Accepts a set of parallel types to shard on.
// If empty, all DID parallel types are used.
void shardAllLike(
    TensorView* ref,
    const std::vector<TensorView*>& tvs,
    const std::unordered_set<ParallelType>& parallel_types);

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

// Returns the index of the sharded logical axis that produces the allocation
// IterDomain sharded on `parallel_type`. If `tv` isn't sharded on the parallel
// type, returns -1.
//
// This is used to correlate `tv` and its corresponding at::Tensor, e.g., by
// `unshardedSizes` and `shardTensor`. `at::Tensor::sizes` and
// `tv->getLogicalDomain()` map one-to-one modulo reduction. However, a size in
// `at::Tensor::sizes` is a factor of the corresponding logical IterDomain's
// extent if that IterDomain is sharded.
int64_t getShardedLogicalAxis(const TensorView* tv, ParallelType parallel_type);

// Returns the IterDomain that's parallelized on `parallel_type`.  If it's not
// found, returns nullptr. `parallel_type` decides which domain to look at.
// ParallelType::Stream looks at the allocation domain and DIDs look at the loop
// domain. Refer to the implementation for the reason.
IterDomain* getShardedIterDomain(
    const TensorView* tv,
    ParallelType parallel_type);

// Reorders a TensorView's loop domain so that parallelized IterDomains are in
// front, making the order most convenient for (inter-GPU and intra-GPU)
// schedulers.
//
// Returns a map of the old index to the new index.
std::unordered_map<int64_t, int64_t> reorderParallelizedToFront(TensorView*);

// Validate the expression is a valid DID split: expr is an outer split with
// device dim as the outer dimension.
bool isValidDeviceSplit(Expr* expr);

// Propagate sharding for the given parallel types from loop domain to
// allocation domain, refining contiguity as needed so allocation aliases the
// original storage layout.
void shardAllocationAsLoop(
    TensorView* tv,
    const std::unordered_set<ParallelType>& parallel_types);

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/ArrayRef.h>

#include <compute_at_map.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <multidevice/multidevice.h>
#include <scheduler/utils.h>
#include <visibility.h>

namespace nvfuser {

// Returns true iff nvFuser was compiled with distributed APIs enabled.
NVF_API bool distributedEnabled();

// Return true if the TensorView is contiguous.
NVF_API bool isTvContiguous(const TensorView* tv);

// Returns whether a TensorView has a non-reduction axis parallelized Didx
// Checks that the other non-reduction axis are not parallelized on Didx
NVF_API bool isSharded(const TensorView*);

// Returns number of device dimensions in a TensorView's loop domain.
NVF_API int64_t numDeviceDims(const TensorView*);

NVF_API std::unordered_set<IterDomain*> getInputsInTargetDomain(
    const std::vector<IterDomain*>& loop_ids,
    const std::vector<IterDomain*>& target_domain);

// Returns the subset of tvs which elements have the different multi-device
// sharding as ref
NVF_API std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    const std::vector<TensorView*>& tvs);

// Returns whether an Expr embeds multi-device resharding
NVF_API bool isResharding(const Expr* expr);

// Returns whether two tensors have different shardings. Expect a
// producer/consumer relationship between the arguments.
NVF_API bool haveDifferentShardings(
    const TensorView* producer,
    const TensorView* consumer);

// Returns a set that contains DIDs and Stream.
NVF_API std::unordered_set<ParallelType> deviceAndStreamParallelTypes();

// Collect device and stream parallelized IterDomains in `domain` and return
// them as a ParallelType-to-IterDomain map. Excludes reduction iterdomains.
NVF_API std::unordered_map<ParallelType, IterDomain*>
mapDeviceAndStreamParallelTypeToId(const std::vector<IterDomain*>& domain);

// Shards all tensors in tvs like reference.
// Accepts a set of parallel types to shard on.
// If empty, all DID parallel types are used.
NVF_API void shardAllLike(
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
NVF_API void shardBetween(
    const std::vector<TensorView*>& from,
    const std::vector<TensorView*>& to,
    TensorView* ref);
// Same as above but using the outputs of the from and to expressions
// to form the from and to TVs.
NVF_API void shardBetween(
    const std::vector<Expr*>& from,
    const std::vector<Expr*>& to,
    TensorView* ref);

// Returns the devices involved in an expr
NVF_API std::set<DeviceIdxType> involvedDevices(Expr* expr);

// Returns the number of device indices present accross all
// device meshes in the Fusion
NVF_API int64_t requestedNumberOfDevices(Fusion*);

// remove the multi-device scheduling annotations
NVF_API void unshard(Fusion*);
NVF_API void unshard(TensorView*);

// Returns the index of the sharded logical axis that produces the allocation
// IterDomain sharded on `parallel_type`. If `tv` isn't sharded on the parallel
// type, returns -1.
//
// This is used to correlate `tv` and its corresponding at::Tensor, e.g., by
// `unshardedSizes` and `shardTensor`. `at::Tensor::sizes` and
// `tv->getLogicalDomain()` map one-to-one modulo reduction. However, a size in
// `at::Tensor::sizes` is a factor of the corresponding logical IterDomain's
// extent if that IterDomain is sharded.
NVF_API int64_t
getShardedLogicalAxis(const TensorView* tv, ParallelType parallel_type);

// Returns the index of the loop axis that's parallelized on `parallel_type`.
// If it's not found, returns -1.
NVF_API int64_t
getShardedLoopAxis(const TensorView* tv, ParallelType parallel_type);

// Shards the input tensor along `axis`. How the tensor gets sliced along `axis`
// is determined by `mesh` and `device_id`. Returns the sharded tensor.
NVF_API at::Tensor shardTensor(
    at::Tensor tensor,
    int64_t axis,
    const DeviceMesh& mesh,
    DeviceIdxType device_id);

// Reorders a TensorView so that the DID parallelized axis are in front.
// Returns a map of the old index to the new index.
NVF_API std::unordered_map<int64_t, int64_t> reorderDIDToFront(TensorView*);

// Given a TensorView and the shape of a sharded tensor of which certain
// dimensions are partially allocated, returns the global shape that'll be used
// to bind to the TensorView's logical domain. This is to solve #3282 so we can
// bind a sharded tensor to a TensorView that has a DID-parallel loop domain.
//
// For example, when `tv` is
//   logical: iM, iN
//   allocation: iDIDx{D}, iN/D, iM
// and `sizes` is [2, 3], the returned shape will be [2, 3D]. This is because,
// according to the allocation domain, iM is fully allocated and iN is sharded
// and thus partially allocated.
//
// If the TensorView is not sharded, this function returns `sizes`.
//
// Limitations:
// - The function assumes that there are no Merges from logical to the
// DID-parallel IterDomains in allocation. Otherwise, it's unclear which logical
// dimension this DID-parallelization should be attributed to.
// - The function assumes that all Splits from logical to the DID-parallel
// IterDomains in allocation are even. This is because there are currently no
// ways to pass in the global shape.
//
// Despite these limitations, I took this approach as a shortcut to fix #3282,
// which blocked many other tasks. I'm however open to other better, long-term
// solutions. Some alternatives considered in #3282 are:
// - Try to bind `at::Tensor`s to allocation domains instead of logical. Many
// `*Op::evaluate` methods (e.g.
// https://github.com/NVIDIA/Fuser/blob/2415d904d1e9a5da7ca6fb1a55d3045bbd510341/csrc/ir/nodes.cpp#L4321-L4329)
// assume the input/output `at::Tensor`s have the same dimension order as the
// logical domain. Doing so would have to change them all.
// - Try to pass into FusionExecutorCache both logical (global) shapes and
// allocated (local) tensors for sharded TensorViews. The logical shapes would
// have to be passed through FusionKernelRuntime, FusionExecutor,
// ExpressionEvaluator, and so on, which is an API overhaul.
NVF_API std::vector<int64_t> unshardedSizes(
    const TensorView* tv,
    c10::IntArrayRef sizes);

} // namespace nvfuser

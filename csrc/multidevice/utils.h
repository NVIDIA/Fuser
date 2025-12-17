// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <iosfwd>

#include "fusion.h"
#include "ir/interface_nodes.h"

namespace nvfuser {

// Identifies which TensorView domain to inspect.
enum class DomainType : int {
  kRoot,
  kLogical,
  kLoop,
  kAllocation,
};

std::ostream& operator<<(std::ostream& os, DomainType domain_type);

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

// Collect device and stream parallelized IterDomains in `domain` and return
// them as a ParallelType-to-IterDomain map. Excludes reduction iterdomains.
std::unordered_map<ParallelType, IterDomain*> mapDeviceAndStreamParallelTypeToId(
    const std::vector<IterDomain*>& domain);

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

// Returns the IterDomain that's parallelized on `parallel_type` in the domain
// of type `domain_type`.
//
// The allocation domain for multidevice TensorViews is set during
// presegmentation, which happens after concretization. At that point fusion
// inputs still lack allocation domains, so callers must explicitly choose
// which domain to inspect. Use `domain_type` to pick loop vs. allocation (or
// root/logical) depending on the information you need.
// Returns the IterDomain that's parallelized on `parallel_type` within
// `domain_type`. If it's not found, returns nullptr.
IterDomain* getShardedIterDomain(
    const TensorView* tv,
    ParallelType parallel_type,
    DomainType domain_type);

// Reorders a TensorView's loop domain so that parallelized IterDomains are in
// front, making the order most convenient for (inter-GPU and intra-GPU)
// schedulers.
//
// Returns a map of the old index to the new index.
std::unordered_map<int64_t, int64_t> reorderParallelizedToFront(TensorView*);

// Validate the expression is a valid DID split: expr is an outer split with
// device dim as the outer dimension.
bool isValidDeviceSplit(Expr* expr);

// When the contracting dimension is sharded, each device has a partial
// matmul output and is followed by an allreduce. For loop split, this is
// represented as an rfactored reduction. For example, for matmul, the local
// logical domain after the rfactor is: i{DIDx}, i{M}, i{N}, r{K//d}.
// Unsqueeze the rfactored DID axis to correctly bind with the logical domain.
// See tests/python/test_multidevice.py/test_matmul_allreduce_loop_split
int64_t getRFactorDeviceDimensionIndex(const TensorView* tv);

} // namespace nvfuser

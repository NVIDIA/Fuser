// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <ir/interface_nodes.h>
#include <multidevice/multidevice.h>
#include <visibility.h>

namespace nvfuser {

// Returns whether a TensorView has a non-reduction axis parallelized Didx
// Checks that the other non-reduction axis are not parallelized on Didx
NVF_API bool isSharded(TensorView*);

// Returns the subset of tvs which elements have the same multi-device sharding
// as ref
template <typename TvIterator>
std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    TvIterator tvs);

// Returns whether an Expr embbeds multi-device resharding
bool isResharding(Expr* expr);

// Returns the devices involved in an expr
std::set<DeviceIdxType> involvedDevices(Expr* expr);

// Returns the number of device indices present accross all
// device meshes in the Fusion
int64_t requestedNumberOfDevices(Fusion*);

// remove the multi-device scheduling annotations
void unshard(Fusion*);
void unshard(TensorView*);

// Runs through the fusion and inserts a resharding Set Op before any resharding
// Expr that is not directly lowerable to a series of communications
// TODO: add an option to rather insert the Set AFTER the resharding Expr
void insertReshardings(Fusion* fusion);

// This can only run after the insertResharding pass.
// Expects all resharding ops are either a set or reduction.
// For each resharding operation that requires communication
// over a noncontiguous slices of the tensor, this pass
// inserts permutations necessary to push the device parallel axis
// to the front so that communication operations are contiguous.
void insertPermutes(Fusion* fusion);

// Returns the index of the axis with ParallelType
// if none return -1.
int64_t dimWithParallelType(
    TensorView*,
    ParallelType,
    bool withReductions = false);

// Returns whether a device's sharded tensors are contiguously stored
// with respect to its unsharded tensor
// i.e. [DIDx(i0), i1] tensors are contiguously stored in [i0, i1]
// [i0, DIDx(i1)] tensors are sharded.
bool isContiguousShard(TensorView*, TensorView*);

} // namespace nvfuser

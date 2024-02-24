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

// Returns whether a TensorView has its first non-reduction axis parallelized
// on Didx
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

// Runs through the fusion and inserts a resharding Set Op before any resharding
// Expr that is not directly lowerable to a series of communications
// TODO: add an option to rather insert the Set AFTER the resharding Expr
void insertReshardings(Fusion* fusion);

// Returns the unsharded tensor size given sharded size and tv
std::vector<int64_t> unshardedSize(
    TensorView* tv,
    c10::IntArrayRef sharded_sizes);

} // namespace nvfuser

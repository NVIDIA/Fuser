// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

#include "fusion.h"
#include "ir/interface_nodes.h"
#include "multidevice/device_mesh.h"
#include "multidevice/multidevice.h"
#include "visibility.h"

namespace nvfuser {

// Returns the number of device indices present across all device meshes in the
// Fusion.
int64_t requestedNumberOfDevices(Fusion*);

// Shards the input tensor along `axis`. How the tensor gets sliced along `axis`
// is determined by `mesh` and `device_id`. Returns the sharded tensor.
NVF_API at::Tensor shardTensor(
    at::Tensor tensor,
    int64_t axis,
    const DeviceMesh& mesh,
    DeviceIdxType device_id);

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
// Despite these limitations, this approach was taken as a shortcut to fix
// #3282. Some alternatives considered in #3282 are:
// - Try to bind `at::Tensor`s to allocation domains instead of logical. Many
// `*Op::evaluate` methods (e.g.
// https://github.com/NVIDIA/Fuser/blob/2415d904d1e9a5da7ca6fb1a55d3045bbd510341/csrc/ir/nodes.cpp#L4321-L4329)
// assume the input/output `at::Tensor`s have the same dimension order as the
// logical domain. Doing so would have to change them all.
// - Try to pass into FusionExecutorCache both logical (global) shapes and
// allocated (local) tensors for sharded TensorViews. The logical shapes would
// have to be passed through FusionKernelRuntime, FusionExecutor,
// ExpressionEvaluator, and so on, which is an API overhaul.
std::vector<int64_t> unshardedSizes(
    const TensorView* tv,
    c10::IntArrayRef sizes);

} // namespace nvfuser

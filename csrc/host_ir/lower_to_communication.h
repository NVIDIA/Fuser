// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

#include "ir/allocation_utils.h"
#include "ir/base_nodes.h"
#include "ir/interface_nodes.h"
#include "ir/internal_base_nodes.h"
#include "multidevice/communication.h"
#include "multidevice/multidevice.h"

namespace nvfuser {

struct CommunicationInfo {
  CommunicationType type;
  // Sharded logical IDs in producer/consumer.
  // For ReduceScatter, this is the scattered axis. Reduced axis is not stored.
  IterDomain* p_sharded_id;
  IterDomain* c_sharded_id;
};

// Returns whether the communication layout is compliant.
// ProcessGroup expects contiguous tensors and
// gathered/scattered axes to be outermost in allocation.
// This is only supported for load/store and reduction ops.
// Composite expressions that are communication + compute are not supported.
bool isCommunicationLayoutCompliant(Expr* expr);

// Given an Expr that's known to be a communication, returns the communication
// info: type and sharded IDs. We assume that the expr has been decomposed and
// represented a single communication. If multiple communications are present or
// 2D sharding, this function will raise an error.
CommunicationInfo getCommunicationInfo(Expr* expr);

// Given the input/output TensorView of a communication, returns its layout
// required by the communication backend (e.g. NCCL or UCC). `sharded_id` is the
// sharded IterDomain stored in a CommunicationInfo. We don't distinguish
// between input and output because the requirements so far are the same. But
// this may change in the future. The returned layout is guaranteed to be
// canonicalized, i.e., the allocation domain is a permutation of the logical
// domain.
Layout getCommunicationLayout(
    TensorView* tv,
    const CommunicationType type,
    IterDomain* sharded_id);

std::vector<Expr*> convertSingleOpToCommunication(
    Expr* c,
    DeviceIdxType my_device_idx,
    const CommunicatorBackend backend = CommunicatorBackend::kNccl);

} // namespace nvfuser

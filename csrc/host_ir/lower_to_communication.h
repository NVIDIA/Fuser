// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <host_ir/lower.h>
#include <ir/all_nodes.h>

namespace nvfuser {

struct CommunicationInfo {
  // Allgather/Gather/Scatter/ReduceScatter/Allreduce/Reduce
  CommunicationType type;
  // Sharded logical IDs in producer/consumer.
  // For ReduceScatter, this is the scattered axis. Reduced axis is not stored.
  IterDomain* p_sharded_id;
  IterDomain* c_sharded_id;
};

// Checks whether the allocation order of id in tv is compliant
// with NCCL/UCC requirements. Specifically, it checks that a gather/scatter
// axis is outermost in the allocation unless its local size is 1.
bool isAllocationOrderCompliant(TensorView* tv, IterDomain* id);

// Returns whether the communication layout is compliant.
// ProcessGroup expects contiguous tensors and
// gathered/scattered axes to be outermost in allocation.
// This is only supported for load/store and reduction ops.
// Composite expressions that are communication + compute are not supported.
bool isCommunicationLayoutCompliant(Expr* expr);

// Returns the communication info for the
// (All)Gather/Scatter/ReduceScatter/(All)Reduce communication that may require
// copying the input/output and reordering the allocation domain.
// We assume that the expr has been decomposed and represented a single
// communication. If multiple communications are present, this function will
// raise an error.
std::optional<CommunicationInfo> getCommunicationInfo(Expr* expr);

std::vector<Expr*> convertSingleOpToCommunication(
    Expr* c,
    DeviceIdxType my_device_idx,
    const HostIrLowerParams& params);

} // namespace nvfuser

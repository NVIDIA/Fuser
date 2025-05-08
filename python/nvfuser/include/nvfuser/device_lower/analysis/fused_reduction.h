// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>

namespace nvfuser {

//! Keep track of certain patterns of reductions.
//!
//! - Allreduce IterDomain: reduced and broadcast domain.
class FusedReductionInfo {
 public:
  void markAsAllreduce(IterDomain* id);

  bool isAllreduce(IterDomain* id) const;

 private:
  // Reduction IterDomains that are also broadcast
  std::unordered_set<IterDomain*> allreduce_ids_;
};

//! Detect reductions and broadcasts that are eligible for the fused
//! reduction kernel. When found, the predicate flags of the broadcast
//! is unset, which effectively makes the broadcast just a unary set
//! op.
//! TODO: Consider moving the warp-based fused reduction here.
void fuseReductionsAndBroadcasts(Fusion*);

} // namespace nvfuser

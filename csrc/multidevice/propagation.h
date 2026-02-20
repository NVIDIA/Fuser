// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <unordered_set>

#include "ir/interface_nodes.h"
#include "scheduler/utils.h"

namespace nvfuser {

int numParallelIterDomains(const TensorView* tv);

template <typename R>
TensorView* findMostParallelTensorView(const R& range) {
  TensorView* reference = nullptr;
  int max_parallel_count = -1;
  for (TensorView* tv : range) {
    auto parallel_count = numParallelIterDomains(tv);
    if (parallel_count > max_parallel_count) {
      max_parallel_count = parallel_count;
      reference = tv;
    }
  }
  return reference;
}

// Propagates the given device/stream ids from ref to target.
void shardLoopLike(
    const TensorView* ref,
    TensorView* tv,
    const std::unordered_set<ParallelType>& selected_parallel_types,
    PropagateDirection direction);

// Canonicalizes tv's loop domain for simplicity and working around schedulers'
// limitations. Many schedulers panic when seeing the input fusion segment
// contains non-DID loop splits. For example, an rFactor tensor may look like
// the following:
//
//                            r{k}
//                            /  \.
// [i{m}         i{n}    iDIDx{d}  r{k/d}]
//               /  \.
//            i{d} i{n/d}
//
// The split of i{n} is unnecessary because i{d} and i{n/d} are both
// ParallelType::Serial. This function replaces the two with i{n} in the loop
// domain.
void canonicalizeLoopDomain(TensorView* tv);

// Unparallelize tv's loop domain for the given parallel types
// and canonicalize it.
void unshard(
    TensorView* tv,
    const std::unordered_set<ParallelType>& parallel_types);

} // namespace nvfuser

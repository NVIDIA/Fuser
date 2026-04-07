// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <optional>
#include <unordered_set>

namespace nvfuser {

class IterDomain;
class TensorView;

namespace scheduler_tools {

// torch.repeat can be represented as:
//
// t0: [i0, i1]
// t1 = broadcast(t0) // [i0, b2, i1]
// t2 = expand(t1, {-1, 2, -1}); // [i0, b2(2), i1]
// t3 = reshape(t2, {i0, 2 * i1}); // [i0, 2*i1]
//
// It is especially important to recognize this pattern when it
// appears at the end of a pointwise fusion segment, where an output
// is used as the reference tensor of scheduling the segment. For
// example, if a segment has the above pattern at the end of
// the segment with t3 as the only output, the whole segment may be
// scheduled based on t3. That is quite common in RoPE, where Q, K and
// V tensors have different sizes but smaller tensors are commonly
// repeated at the end of its computation.
//
// This can be problematic since the whole segment is scheduled based
// on the repeated tensor whose size is larger than the rest of the
// tensors by the repetition factor. For example, if the it is
// repeated twice, we would launch threads and blocks that are
// required for the twice-larger tensor but most of the actual
// computations will actually only need half of them. In fact,
// depending on actual scheduling strategies, they may be just
// redundantly doing the same computations, which should be avoided if
// possible.
//
// getMaybeStaticRepeatInfo analyzes a given tensor and its producers
// to detect the above repeat pattern. The detected pattern is
// currently only used by the resize scheduler. It effectively factors
// out the repetition factor as an iter domain and moves it to the
// outermost position. The remaining iter domains are scheduled and
// propagated to the rest of the tensors.
//
// TODO: Consider generalizing this heuristics to the other
// schedulers.

// Some of the relevant iter domains of the output tensor of the
// reshape that realizes a repetition.
struct StaticRepeatInfo {
  // Tensor after a repeat. In the above example, this corresponds
  // to t3.
  TensorView* reshape_output_tv = nullptr;
  // Root ID that is repeated. In the above example, this corresponds
  // to i1.
  IterDomain* input_id = nullptr;
  // Root ID that is originally an expanded broadcast. In the above example,
  // this corresponds to b(2).
  IterDomain* factor_id = nullptr;
  // Logical repeated ID. In the above example, this corresponds
  // to 2*i1.
  IterDomain* output_id = nullptr;
};

// Check if the given tensor matches with the final reshape output
// tensor of the repetition pattern and return the relevant
// information about the detected pattern. Only a static repeat case
// is considered.
std::optional<StaticRepeatInfo> getMaybeStaticRepeatInfo(
    TensorView* maybe_repeat_out_tv);

} // namespace scheduler_tools

} // namespace nvfuser

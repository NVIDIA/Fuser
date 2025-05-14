// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// A pre-segmenter pass that moves gather operations ahead of producer
// unary pointwise ops such as cast. We move take_along_axis ahead of another op
// if that op is a unary pointwise op such as cast or neg, or it's a squeeze op.
// Following 2 examples demonstrated what this pass does:

// ┌───────┐                            ┌───────┐
// │ Expr0 │                            │ Expr0 │
// │       │                            │       ┼────────┐
// └──┬────┘                            └───┬───┘        │
//    │                                     │            │
// ┌──▼────┐                            ┌───▼───┐        │
// │ Cast  │                            │Gather │    ┌───▼───┐
// │       ├─────┐     ─────────►       │       │    │ Cast  │
// └───┬───┘     │                      └───┬───┘    │       │
//     │         │                          │        └───┬───┘
// ┌───▼───┐  ┌──▼────┐                 ┌───▼───┐        │
// │Gather │  │ Expr 2│                 │ Cast  │    ┌───▼───┐
// │       │  │       │                 │       │    │ Expr2 │
// └───┬───┘  └───────┘                 └───┬───┘    │       │
//     │                                    │        └───────┘
// ┌───▼───┐                            ┌───▼───┐
// │ Expr 1│                            │ Expr1 │
// │       │                            │       │
// └───────┘                            └───────┘

// Or in the case of a squeeze:
//            Expr 0
//              │                  │ Expr 1    Expr 0
//              ▼                  ▼             │  │
//            Squeeze             Unsqueeze      ▼  │
//              │  │                  ───────► Gathe└────►Squeeze
// Expr 1       ▼  └───► Expr3                   │           │
//   ───────► Gather                             ▼           ▼
//              ▼            ──────►           Squeeze    Expr 3
//            Expr 2                             ▼
//                                             Expr 2
// Please note Expr 1 is the def of index Tv and Squeeze was the def for
// Lookup Tv.

// The motivations for this pass are:
// 1. Moving the op after gather reduced the amount of memory this op has to
// work on. ( doesn't quite apply when there is an Expr 3 in the above example)
// 2. If Expr 3 and Gather are in different segments duplicating the op
// removes any temporary Tv that is communicated between these two segments.

// Current restrictions:
// We only handle gather operations of the type take_along_axis.
// Future work:
// TODO: We currently handle a fusion with a single take_along_axis. Multiple
// ops of the type should be handled.
// TODO: In the case that def of lookupTv is a squeeze - we only support a
// single dim being squeezed. We should add book keeping to handle mulitple
// dims.
class MoveGatherPass : public OptimizationPass<MoveGatherPass> {
  friend class OptimizationPass<MoveGatherPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "MoveGatherPass";
  }
};

} // namespace nvfuser::preseg_passes

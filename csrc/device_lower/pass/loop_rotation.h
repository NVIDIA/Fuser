// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <executor_params.h>
#include <ir/all_nodes.h>

namespace nvfuser {

// Note [Loop Rotation]
// Loop Rotation is an optimization pass to improve instruction scheduling. For
// a given loop, for example:
//   for (int i = 0; i < n; i++) {
//     line1(i);
//     line2(i);
//     line3(i);
//     line4(i);
//   }
// If we rotate one line up, then we get
//   if (0 < n) {
//     line1(0);
//   }
//   for (int i = 0; i < n; i++) {
//     line2(i);
//     line3(i);
//     line4(i);
//     if (i + 1 < n) {
//       line1(i + 1);
//     }
//   }
// Similarly, if we rotate two lines up, then we get
//   if (0 < n) {
//     line1(0);
//     line2(0);
//   }
//   for (int i = 0; i < n; i++) {
//     line3(i);
//     line4(i);
//     if (i + 1 < n) {
//       line1(i + 1);
//       line2(i + 1);
//     }
//   }
// In order to take advantage of this pass, the scheduler needs to specify which
// loop to rotate and the consumers whose allocation and computation will be
// rotated, and pass this information as compilation parameter. For example, if
// I have a fusion that will create the following loop structure:
//   for (int i = 0; i < id1.extent(); i++) {
//     float T1[5];
//     for (int j = 0; j < 5; j++) {
//       if (i < T0.size[0]) {
//         T1[j] = sin(T0[i, j]);
//       }
//     }
//     float T2[5];
//     for (int j = 0; j < 5; j++) {
//       T2[j] = cos(T1[j]);
//     }
//     float T3[5];
//     for (int j = 0; j < 5; j++) {
//       T3[j] = exp(T2[j]);
//     }
//     for (int j = 0; j < 5; j++) {
//       if (i < T4.size[0]) {
//         T4[i, j] = log(T3[j]);
//       }
//     }
//   }
// Then the scheduler could make a compilation parameter {id1, {T1, T2}} to the
// fusion, and this pass will transform the code as
//   float T1[5];
//   float T2[5];
//   if (0 < id1.extent()) {
//     for (int j = 0; j < 5; j++) {
//       if (0 < T0.size[0]) {
//         T1[j] = sin(T0[0, j]);
//       }
//     }
//     for (int j = 0; j < 5; j++) {
//       T2[j] = cos(T1[j]);
//     }
//   }
//   for (int i = 0; i < id1.extent(); i++) {
//     float T3[5];
//     for (int j = 0; j < 5; j++) {
//       T3[j] = exp(T2[j]);
//     }
//     for (int j = 0; j < 5; j++) {
//       if (i < T4.size[0]) {
//         T4[i, j] = log(T3[j]);
//       }
//     }
//     if (i + 1 < id1.extent()) {
//       for (int j = 0; j < 5; j++) {
//         if (i + 1 < T0.size[0]) {
//           T1[j] = sin(T0[i + 1, j]);
//         }
//       }
//       for (int j = 0; j < 5; j++) {
//         T2[j] = cos(T1[j]);
//       }
//     }
//   }
// Currently, because all our existing predicates should already cover
// out-of-bound access, so we are omitting the predicates to get a
// better-looking code:
//   float T1[5];
//   float T2[5];
//   for (int j = 0; j < 5; j++) {
//     if (0 < T0.size[0]) {
//       T1[j] = sin(T0[0, j]);
//     }
//   }
//   for (int j = 0; j < 5; j++) {
//     T2[j] = cos(T1[j]);
//   }
//   for (int i = 0; i < id1.extent(); i++) {
//     float T3[5];
//     for (int j = 0; j < 5; j++) {
//       T3[j] = exp(T2[j]);
//     }
//     for (int j = 0; j < 5; j++) {
//       if (i < T4.size[0]) {
//         T4[i, j] = log(T3[j]);
//       }
//     }
//     for (int j = 0; j < 5; j++) {
//       if (i + 1 < T0.size[0]) {
//         T1[j] = sin(T0[i + 1, j]);
//       }
//     }
//     for (int j = 0; j < 5; j++) {
//       T2[j] = cos(T1[j]);
//     }
//   }

// vector of (tv, dim, selection)
// For each entry in the vector, the selected tv/expr in loop tv->axis(dim)
// will be rotated
using LoopRotationParam = std::vector<
    std::tuple<TensorView*, int64_t, std::unordered_set<Statement*>>>;

std::vector<Expr*> rotateLoops(
    const std::vector<Expr*>& exprs,
    const LoopRotationParam& params);

} // namespace nvfuser

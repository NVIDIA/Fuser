// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>

#include <vector>

namespace nvfuser {

//! This is an experimental pass that subsumes the following previously-used passes:
//!
//!  - reuseMemoryAllocations
//!  - CircularBufferPass
//!  - insertRawThreadSynchronization
//!  - insertWarThreadSynchronization
//!  - insertWarAsyncWait
//!
//! These passes all used similar analyses and had circular dependencies because syncing, circular buffering, and memory reuse are intertwined topics.
std::vector<Expr*> circularBufferAndInsertSyncs(const std::vector<Expr*>& exprs);

} // namespace nvfuser

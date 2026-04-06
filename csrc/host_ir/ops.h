// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// This file contains user-facing functions to create host IR nodes. When using
// host_ir/ir.h directly, the user has to create both input TensorViews and
// output TensorViews and then call IrBuilder::create to create an Expr
// connecting them.  The functions here instead take input TensorViews and
// produce output TensorViews, so they are more convenient.  This separation is
// similar to csrc/ops vs csrc/ir.

#pragma once

#include "host_ir/ir.h"
#include "ir/interface_nodes.h"

namespace nvfuser::hir {

// Creates a ShardByStream without needing the destination TensorView. Returns
// the destination TensorView. `e` is the Expr from which we propagate
// `ParallelType::Stream` domain from. `source` must be either an input or an
// output of `e`. The destination TensorView will have a `ParallelType::Stream`
// domain that's consistent with `e` and an allocation domain that's a shard of
// `source`.
//
// Why is `e` unnecessary? I made a mistake previously to propagate `source`'s
// loop domain to `destination`. This broke
// test_stream.py::test_two_matmuls_not_inlinable because, when `source` is an
// input of `e`, `source`'s loop domain reflects its producing Expr rather than
// `e`.
// If `destination` cannot be sharded by `ParallelType::Stream`, returns
// nullptr. For e.g.: in decomposed allgather, we go from DIDx -> Stream.
// `destination` is already sharded on `DIDx`
TensorView* shardByStream(TensorView* source, Val* stream_index, Expr* e);

} // namespace nvfuser::hir

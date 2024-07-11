// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/id_model.h>
#include <id_model/utils.h>

namespace nvfuser {

// If the for-loop is double-buffered and not prologue, the loop
// index should be advanced by one except for the double-buffered
// tensor itself
Val* adjustProducerLoopIndexForCircularBuffering(
    const Expr* expr,
    const ForLoop* for_loop,
    const IdModel& id_model,
    Val* loop_index);

Val* adjustIndexToSwitchBuffer(
    TensorView* tv,
    bool as_consumer,
    const std::vector<ForLoop*>& for_loops,
    Val* idx);

std::optional<CircularBufferLoopStage> getCircularBufferLoopStage(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph);

} // nvfuser nvfuser

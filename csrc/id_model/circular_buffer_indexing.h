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

// For a circur-buffering expr, the producer loop index needs to be
// advanced by (#stages - 1) if it's the main loop. Return the offset
// if it's applicable. Otherwise, nullptr is returned.
Val* getLoopIndexOffsetForProducerOfCircularBuffer(
    const Expr* expr,
    const ForLoop* for_loop,
    const IdModel& id_model);

// Get the additional offset for a circular buffer. This offset will
// be added to the normal linear index. For example, if this is a
// double buffered tensor, the offset would look like "i % 2", where i
// is the loop index of the double-buffer loop.
Val* getCircularBufferOffset(
    TensorView* circular_buffer_tv,
    bool as_consumer,
    const std::vector<ForLoop*>& for_loops);

// Find the circular buffering stage of a given circular buffered tensor
std::optional<CircularBufferLoopStage> getCircularBufferLoopStage(
    const TensorView* circular_buffer_tv,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph);

} // namespace nvfuser

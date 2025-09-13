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

namespace nvfuser {

// Get the loop index of a given loop domain for circular buffer
// loops. nullptr is returned if not relevant.
//
// This is a WAR for circular buffering. TensorIndexer has a map of
// loop indices for all loop groups, however, it does not work with
// circular buffering. The loop graph is
// designed to represent each loop and each loop group is supposed
// to have a one-to-one relationship with each loop. However, for
// circular buffering, this assumption is broken as we are using
// the same iter domain for the prologue, main and epilogue
// loops. Ideally, those loops should have distinctive loop groups,
// but for now, here's a workaround to get a correct loop index
Val* getLoopIndexOfCircularBufferLoop(
    IterDomain* loop_id,
    const std::vector<kir::ForLoop*>& for_loops,
    const IdModel& id_model);

// For a circular-buffering expr, the producer loop index needs to be
// advanced by (#stages - 1) if it's the main loop. Return the offset
// if it's applicable. Otherwise, nullptr is returned.
Val* getLoopIndexOffsetForProducerOfCircularBuffer(
    const Expr* expr,
    const kir::ForLoop* for_loop,
    const IdModel& id_model);

// Get the additional offset for a circular buffer. This offset will
// be added to the normal linear index. For example, if this is a
// double buffered tensor, the offset would look like "i % 2", where i
// is the loop index of the double-buffer loop.
Val* getOffsetForCircularBufferTensor(
    TensorView* circular_buffer_tv,
    bool as_consumer,
    const std::vector<kir::ForLoop*>& for_loops);

// Find the circular buffering stage of a given circular buffered tensor
CircularBufferLoopStage getCircularBufferLoopStage(
    const TensorView* circular_buffer_tv,
    const std::vector<kir::ForLoop*>& for_loops,
    const ValGraph& loop_graph);

} // namespace nvfuser

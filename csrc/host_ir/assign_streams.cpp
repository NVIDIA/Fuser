// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "host_ir/assign_streams.h"

#include "host_ir/container.h"
#include "ir/builder.h"

namespace nvfuser::hir {

void AssignStreams::runPass(Fusion* fusion) {
  auto* hic = dynamic_cast<HostIrContainer*>(fusion);
  NVF_CHECK(hic != nullptr);

  // For each stream-parallel loop, insert to the beginning a SetCurrentStream
  // and a Synchronize (to the main stream). Right after the loop exits, insert
  // another loop that joins all the worker streams.

  for (auto it = hic->topLevel().exprs().begin();
       it != hic->topLevel().exprs().end();
       ++it) {
    auto* for_loop = dynamic_cast<hir::ForLoop*>(*it);
    if (!for_loop) {
      continue;
    }

    // FIXME: should have checked that the loop is stream-parallel

    auto* get_current_stream = IrBuilder::create<GetCurrentStream>();
    Stream* main_stream = get_current_stream->stream();
    hic->topLevel().insert(it, get_current_stream);

    // At the beginning of each iteration: set stream and synchronize with main
    // stream
    auto* worker_stream = IrBuilder::create<Stream>(for_loop->index());
    auto* set_stream = IrBuilder::create<SetCurrentStream>(worker_stream);
    auto* sync_main = IrBuilder::create<Synchronize>(main_stream);

    // Insert at the beginning of the loop body
    auto body_it =
        for_loop->body().insert(for_loop->body().exprs().begin(), set_stream);
    for_loop->body().insert(std::next(body_it), sync_main);

    // After the loop: create a joining loop to synchronize all worker streams
    auto* join_loop = IrBuilder::create<hir::ForLoop>(
        for_loop->index(), for_loop->start(), for_loop->stop());

    // In the joining loop: synchronize each worker stream
    auto* join_worker_stream = IrBuilder::create<Stream>(join_loop->index());
    auto* sync_worker = IrBuilder::create<Synchronize>(join_worker_stream);
    join_loop->body().push_back(sync_worker);

    // Insert join_loop after the current for_loop
    auto next_it = std::next(it);
    hic->topLevel().insert(next_it, join_loop);
  }
}

} // namespace nvfuser::hir

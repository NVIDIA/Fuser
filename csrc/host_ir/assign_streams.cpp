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
  FusionGuard fg(hic);

  for (auto it = hic->topLevel().exprs().begin();
       it != hic->topLevel().exprs().end();) {
    auto next_it = std::next(it);

    auto* for_loop = dynamic_cast<hir::ForLoop*>(*it);
    if (for_loop == nullptr) {
      it = next_it;
      continue;
    }

    // We should check that the loop is stream-parallel. This is not necessary
    // at this moment because all loops are stream-parallel. This is also hard
    // to do becauase hir::ForLoop doesn't point to the source IterDomain.

    auto* get_current_stream = IrBuilder::create<GetCurrentStream>();
    Stream* main_stream = get_current_stream->stream();
    hic->topLevel().insert(it, get_current_stream);

    // At the beginning of each iteration: set stream and synchronize with main
    // stream
    auto* worker_stream = IrBuilder::create<Stream>(for_loop->index());
    auto* set_stream = IrBuilder::create<SetCurrentStream>(worker_stream);
    auto* sync_main = IrBuilder::create<Synchronize>(main_stream);
    auto old_begin = for_loop->body().exprs().begin();
    for_loop->body().insert(old_begin, set_stream);
    for_loop->body().insert(old_begin, sync_main);

    // After the loop: create a joining loop to synchronize all worker streams
    auto* join_loop = IrBuilder::create<hir::ForLoop>(
        for_loop->index(), for_loop->start(), for_loop->stop());

    // In the joining loop: synchronize each worker stream
    auto* join_worker_stream = IrBuilder::create<Stream>(join_loop->index());
    auto* sync_worker = IrBuilder::create<Synchronize>(join_worker_stream);
    join_loop->body().push_back(sync_worker);

    // Insert join_loop after the current for_loop
    hic->topLevel().insert(next_it, join_loop);
    it = next_it;
  }
}

} // namespace nvfuser::hir

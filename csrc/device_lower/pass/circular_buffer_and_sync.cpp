// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/circular_buffer_and_sync.h>

#include <debug.h>
#include <device_lower/dependencies.h>
#include <device_lower/utils.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <options.h>

namespace nvfuser {

namespace {

// TODO: move this struct and utilities to lower_utils
// See https://github.com/NVIDIA/Fuser/pull/4810
enum class MemoryProxy { Generic, Async };

MemoryProxy getMemoryProxy(Expr* expr) {
  if (auto* mma = dynamic_cast<MmaOp*>(expr);
      (mma && (mma->isHopper() || mma->isBlackwell())) ||
      ir_utils::isCpAsyncOp(expr) || ir_utils::isCpAsyncBulk(expr)) {
    return MemoryProxy::Async;
  } else {
    return MemoryProxy::Generic;
  }
}

enum class CompletionSync { MBarrierExpectTX, CpAsync, Wgmma };

struct RawSyncInterval {
  Expr* producer;
  Expr* consumer;

  CompletionSync completion_sync_type;

  // If the producer expression requires fencing of any memory types before the
  // consumer due to mismatched proxies, list the mtypes of the dependencies
  // here. If it is empty, then no proxy is required.
  std::unordered_set<MemoryType> proxy_fence_mtypes;
};

struct WarSyncInterval {
  TensorView* tv;
  Expr* async_read;
  Expr* write;

  CompletionSync completion_sync_type;
};

// For warp specialization
struct WarpSpecializedRole {
  // For Hopper, we will have one async warp with index 0 and compute warps with
  // index -1.
  int64_t async_warp = -1L;
};

class SyncHelper {
 public:
  SyncHelper(const std::vector<Expr*> exprs) : exprs_(exprs), deps_(exprs_) {
    // Look up circular buffering info

    // Determine warp specialized roles

    // Find all producer-consumer RAW sync intervals (p2c data dependencies)

    // Find all WAR sync intervals (read must complete before overwriting in
    // circular buffer)

    // Resolve intervals. This means finding a set of planned syncs,

    // Add mbarriers for pingpong

    // Allocate mbarriers
  }

  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

 private:
  std::vector<Expr*> exprs_;
  DependencyMapper deps_;
};

} // namespace

std::vector<Expr*> circularBufferAndInsertSyncs(
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("circularBufferAndSync");

  SyncHelper helper(exprs);

  return helper.exprs();
}

} // namespace nvfuser

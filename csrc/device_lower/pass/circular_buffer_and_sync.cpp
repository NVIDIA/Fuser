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

class SyncHelper {
 public:
  SyncHelper(const std::vector<Expr*> exprs) : exprs_(exprs), deps_(exprs_) {}

  void run() {}

  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

 private:
  std::vector<Expr*> exprs_;
  DependencyMapper deps_;
};

}

std::vector<Expr*> circularBufferAndInsertSyncs(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("circularBufferAndSync");

  SyncHelper helper(exprs);

  helper.run();

  return exprs;
}

} // namespace nvfuser

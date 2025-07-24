// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/dependencies.h>

#include <debug.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <ops/arith.h>
#include <options.h>


namespace nvfuser {

DependencyMapper::DependencyMapper(kir::Kernel* kernel) {
  current_pos_ = 0;
  current_coords_ = {0};

  traverse(kernel->exprs());
}

void DependencyMapper::dispatch(Expr* expr) {
  // Record expr position


  // Increment current position and coords


  // Record reads
}

} // namespace nvfuser

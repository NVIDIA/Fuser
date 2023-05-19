// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>

namespace nvfuser {

//! Set up KernelPerformanceProfile of GpuLower when enabled, which
//! keeps track of expressions to profile. A new TensorView is added
//! for storing profiling results. The expression list is prepended
//! with an kir::Allocate node to allocate the TensorView profile
//! buffer. Note that any expression added after this pass will not be
//! profiled, so this pass should be called after all expressions are
//! lowered. KernelPerformanceProfile is copied to Kernel after
//! lowering.
std::vector<Expr*> instrumentKernel(const std::vector<Expr*>& exprs);

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ir/all_nodes.h>
#include <ops/all_ops.h>

namespace direct_bindings {

// Get string name for UnaryOp
std::string toString(const nvfuser::UnaryOp* uop);

// Get string name for BinaryOp
std::string toString(const nvfuser::BinaryOp* bop);

// Get string name for TernaryOp
std::string toString(const nvfuser::TernaryOp* bop);

// Get string name for ReductionOp
std::string toString(const nvfuser::ReductionOp* rop);

} // namespace direct_bindings

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ops/all_ops.h>

namespace nvfuser::python {

// Get string name for UnaryOp
std::string toString(const UnaryOp* uop);

// Get string name for BinaryOp
std::string toString(const BinaryOp* bop);

// Get string name for TernaryOp
std::string toString(const TernaryOp* bop);

// Get string name for ReductionOp
std::string toString(const ReductionOp* rop);

// Get string name for ScanOp
std::string toString(const ScanOp* sop);

} // namespace nvfuser::python

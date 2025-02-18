// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/base_nodes.h>

namespace nvfuser {

// A stupid and simple compiler that compiles a string into fusion IR. It is
// stupid because of the following limitations:
// - only support named scalars as variables
// - tokens must be separated by one and only one space, for example, i1+i2 and
//   i1  + i2 are all illegal, you have to write i1 + i2. Also note -5 is a
//   single negative integer constant, but - 5 is an expression neg(5)
// - poor error message
namespace stupid_simple_compiler {

// syntatic sugar to conveniently compile string into Val*
namespace ops {

Val* operator""_(const char* str, size_t);

} // namespace ops

} // namespace stupid_simple_compiler

} // namespace nvfuser

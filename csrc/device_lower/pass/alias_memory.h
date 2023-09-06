// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <dispatch.h>
#include <ir/all_nodes.h>

#include <vector>

namespace nvfuser {

//! Reuse Allocation nodes via pointer aliasing
//!
//! First pass finds candidate TensorViews
//! A candidate TensorView is anything in shared memory OR
//! in local memory with a static size larger than register_size_threshold
//!
//! Second pass finds appropriate input Allocate Node
//! among candidate TensorViews
//!
//! Alias Criteria:
//! If input is a candidate TensorView,
//!          input allocation has the same size as output allocation,
//!          thread bindings match,
//!          is not used after this op:
//! then alias output Allocate to input Allocate.
//!
std::vector<Expr*> reuseMemoryAllocations(const std::vector<Expr*>& exprs);

} // namespace nvfuser

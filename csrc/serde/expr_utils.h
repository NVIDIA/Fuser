// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once
#include <exceptions.h>
#include <kernel.h>
#include <serde/fusion_cache_generated.h>
#include <type.h>

namespace nvfuser::serde {

//! Add unique items to container
template <typename Container, typename T>
bool insertUniqueItem(Container& container, T v) {
  if (std::find(container.begin(), container.end(), v) == container.end()) {
    container.push_back(v);
    return true;
  }
  return false;
}

//! Add TensorView RootDomain IterDomain Extents for all kernel inputs
//! Common function between ExpressionBuilder and ExpressionSerializer to
//! ensure deterministic order in operation stack.
std::vector<nvfuser::Val*> gatherSymbolicValues(kir::Kernel* kernel);

} // namespace nvfuser::serde

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/builder_passkey.h>
#include <ir/container.h>

namespace nvfuser {

// Helper for pure composition pattern
// Checks if container OR parent is of specified type
template <typename T>
bool IrBuilderPasskey::isInContainerType() const {
  if (ir_container_->isA<T>()) {
    return true;
  }
  auto* parent = ir_container_->parent();
  return parent && parent->isA<T>();
}

// Helper for checking multiple types
template <typename T1, typename T2>
bool IrBuilderPasskey::isInContainerType() const {
  if (ir_container_->isOneOf<T1, T2>()) {
    return true;
  }
  auto* parent = ir_container_->parent();
  return parent && parent->isOneOf<T1, T2>();
}

} // namespace nvfuser

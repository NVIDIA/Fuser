// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/container.h>

#include <ir/cloner.h>
#include <ir/storage.h>

namespace nvfuser {

// Forward declaration - Fusion inherits from impl::IrContainer
class Fusion;

namespace impl {

IrContainer::IrContainer() : ir_storage_(std::make_unique<IrStorage>()) {
  ir_storage_->parent_ = static_cast<Fusion*>(this);
}

IrContainer::~IrContainer() {}

} // namespace impl
} // namespace nvfuser

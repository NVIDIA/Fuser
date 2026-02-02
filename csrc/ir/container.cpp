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

IrContainer::IrContainer() : ir_storage_(std::make_unique<IrStorage>()) {
  ir_storage()->parent_ = this;
}

IrContainer::~IrContainer() {}

void IrContainer::swap(IrContainer& a, IrContainer& b) noexcept {
  // We need to be careful to call IrStorage swap not unique_ptr swap, which
  // will only swap the ptrs NOT the contents.
  IrStorage::swap(*(a.ir_storage()), *(b.ir_storage()));

  // Fix parent pointers after swapping containers
  // After swap, each IrContainer owns a different IrStorage, so we must update
  // the parent backpointers in those containers to point to their new owners
  if (a.ir_storage_) {
    a.ir_storage()->parent_ = &a;
    // Also update all Statement ir_container_ pointers to point to new owner
    for (auto val : a.vals()) {
      val->ir_container_ = &a;
    }
    for (auto expr : a.deterministic_exprs()) {
      expr->ir_container_ = &a;
    }
  }
  if (b.ir_storage_) {
    b.ir_storage()->parent_ = &b;
    // Also update all Statement ir_container_ pointers to point to new owner
    for (auto val : b.vals()) {
      val->ir_container_ = &b;
    }
    for (auto expr : b.deterministic_exprs()) {
      expr->ir_container_ = &b;
    }
  }
}

IrCloner IrContainer::copy(const IrContainer* from, IrContainer* to) {
  auto ir_cloner = IrStorage::copy(from->ir_storage(), to->ir_storage());

  return ir_cloner;
}
} // namespace nvfuser

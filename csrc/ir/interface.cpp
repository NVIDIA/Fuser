// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/container.h>
#include <ir/interface.h>

namespace nvfuser {

// Default constructor - creates new IrContainer
IrContainer::IrContainer() : ir_storage_(std::make_unique<IrStorage>()) {
  ir_storage()->setParent(this);
}

// Copy constructor - clones the container
IrContainer::IrContainer(const IrContainer& other)
    : ir_storage_(std::make_unique<IrStorage>(*other.ir_storage_)) {
  ir_storage()->setParent(this);
}

// Move constructor
IrContainer::IrContainer(IrContainer&& other) noexcept
    : ir_storage_(std::move(other.ir_storage_)) {
  ir_storage()->setParent(this);
}

// Destructor - releases container without deleting if not owned
IrContainer::~IrContainer() {
  // if (ir_storage_) {
  //   ir_storage_.release(); // Don't delete the container
  // }
  // ir_storage()->~IrContainer();
}

// Copy assignment using copy-and-swap idiom
IrContainer& IrContainer::operator=(const IrContainer& other) {
  if (this != &other) {
    IrContainer temp(other);

    // swap handles parent reset.
    swap(*this, temp);
  }
  return *this;
}

// Move assignment
IrContainer& IrContainer::operator=(IrContainer&& other) noexcept {
  if (this != &other) {
    ir_storage_ = std::move(other.ir_storage_);

    // Update parent pointer to point to the new owner
    if (ir_storage_) {
      ir_storage()->setParent(this);
    }
  }

  return *this;
}

// Swap function - enables efficient copy-and-swap idiom
void swap(IrContainer& a, IrContainer& b) noexcept {
  using std::swap;
  swap(a.ir_storage_, b.ir_storage_);

  // Fix parent pointers after swapping containers
  // After swap, each IrContainer owns a different IrStorage, so we must update
  // the parent backpointers in those containers to point to their new owners
  if (a.ir_storage_) {
    a.ir_storage()->setParent(&a);
  }
  if (b.ir_storage_) {
    b.ir_storage()->setParent(&b);
  }
}

IrCloner IrContainer::copy(const IrContainer* from, IrContainer* to) {
  auto ir_cloner = IrStorage::copy(from->ir_storage(), to->ir_storage());

  // Ensure ir container for statements is updated.
  for (auto val : to->vals()) {
    val->ir_container_ = to;
  }
  for (auto expr : to->deterministic_exprs()) {
    expr->ir_container_ = to;
  }

  return ir_cloner;
}
} // namespace nvfuser

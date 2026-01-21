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
IrContainer::IrContainer() : container_(std::make_unique<IrContainer>()) {
  container()->setParent(this);
}

// Copy constructor - clones the container
IrContainer::IrContainer(const IrContainer& other)
    : container_(std::make_unique<IrContainer>(*other.container_)) {
  container()->setParent(this);
}

// Move constructor
IrContainer::IrContainer(IrContainer&& other) noexcept
    : container_(std::move(other.container_)) {
  container()->setParent(this);
}

// Destructor - releases container without deleting if not owned
IrContainer::~IrContainer() {
  // if (container_) {
  //   container_.release(); // Don't delete the container
  // }
  // container()->~IrContainer();
}

// Copy assignment using copy-and-swap idiom
IrContainer& IrContainer::operator=(const IrContainer& other) {
  if (this != &other) {
    IrContainer temp(other);
    swap(*this, temp);
  }
  return *this;
}

// Move assignment
IrContainer& IrContainer::operator=(IrContainer&& other) noexcept {
  if (this != &other) {
    container_ = std::move(other.container_);

    // Update parent pointer to point to the new owner
    if (container_) {
      container()->setParent(this);
    }
  }

  return *this;
}

// Swap function - enables efficient copy-and-swap idiom
void swap(IrContainer& a, IrContainer& b) noexcept {
  using std::swap;
  swap(a.container_, b.container_);

  // Fix parent pointers after swapping containers
  // After swap, each IrContainer owns a different IrStorage, so we must update
  // the parent backpointers in those containers to point to their new owners
  if (a.container_) {
    a.container()->setParent(&a);
  }
  if (b.container_) {
    b.container()->setParent(&b);
  }
}

} // namespace nvfuser

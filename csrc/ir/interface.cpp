// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/interface.h>
#include <ir/container.h>

namespace nvfuser {

// Default constructor - creates new IrContainer
IrInterface::IrInterface() : container_(std::make_unique<IrContainer>()) {
  container_->setParent(this);
}

// Constructor with existing container
IrInterface::IrInterface(std::unique_ptr<IrContainer> container)
    : container_(std::move(container)) {
  container_->setParent(this);
}

// Special constructor for Stage 2 dual inheritance (temporary)
// Wraps an existing IrContainer without taking ownership
IrInterface::IrInterface(IrContainer* existing_container, bool take_ownership)
    : container_(existing_container), owns_container_(take_ownership) {
  if (existing_container) {
    existing_container->setParent(this);
  }
}

// Copy constructor - clones the container
IrInterface::IrInterface(const IrInterface& other)
    : container_(std::make_unique<IrContainer>(*other.container_)),
      owns_container_(true) {  // Cloned container is always owned
  container_->setParent(this);
}

// Move constructor
IrInterface::IrInterface(IrInterface&& other) noexcept
    : container_(std::move(other.container_)),
      owns_container_(other.owns_container_) {
  other.owns_container_ = true;  // Reset moved-from state

  // Update parent pointer to point to the new owner
  if (container_) {
    container_->setParent(this);
  }
}

// Destructor - releases container without deleting if not owned
IrInterface::~IrInterface() {
  if (!owns_container_ && container_) {
    container_.release();  // Don't delete the container
  }
}

// Copy assignment using copy-and-swap idiom
IrInterface& IrInterface::operator=(const IrInterface& other) {
  if (this != &other) {
    IrInterface temp(other);
    swap(*this, temp);
  }
  return *this;
}

// Move assignment
IrInterface& IrInterface::operator=(IrInterface&& other) noexcept {
  container_ = std::move(other.container_);
  owns_container_ = other.owns_container_;
  other.owns_container_ = true;

  // Update parent pointer to point to the new owner
  if (container_) {
    container_->setParent(this);
  }

  return *this;
}

// Swap function - enables efficient copy-and-swap idiom
void swap(IrInterface& a, IrInterface& b) noexcept {
  using std::swap;
  swap(a.container_, b.container_);
  swap(a.owns_container_, b.owns_container_);

  // Fix parent pointers after swapping containers
  // After swap, each IrInterface owns a different container, so we must update
  // the parent backpointers in those containers to point to their new owners
  if (a.container_) {
    a.container_->setParent(&a);
  }
  if (b.container_) {
    b.container_->setParent(&b);
  }
}

} // namespace nvfuser

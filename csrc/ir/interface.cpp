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
IrInterface::IrInterface() : container_(std::make_unique<IrContainer>()) {}

// Constructor with existing container
IrInterface::IrInterface(std::unique_ptr<IrContainer> container)
    : container_(std::move(container)) {}

// Copy constructor - clones the container
IrInterface::IrInterface(const IrInterface& other)
    : container_(std::make_unique<IrContainer>(*other.container_)) {}

// Move constructor
IrInterface::IrInterface(IrInterface&& other) noexcept
    : container_(std::move(other.container_)) {}

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
  return *this;
}

// Swap function - enables efficient copy-and-swap idiom
void swap(IrInterface& a, IrInterface& b) noexcept {
  using std::swap;
  swap(a.container_, b.container_);
}

} // namespace nvfuser

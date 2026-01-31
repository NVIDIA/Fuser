// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <memory>

#include <ir/base_nodes.h>
#include <ir/storage.h>
#include <visibility.h>

namespace nvfuser {

// Forward declaration of impl namespace
namespace impl {
class IrContainer;
}

class Fusion;
class IrBuilderPasskey;
class ExprPasskey;
class OptOutMutator;

// Passkey for container to register names with statements
class IrContainerPasskey {
  friend class impl::IrContainer;
  friend class IrStorage;

 private:
  explicit IrContainerPasskey() = default;
};

namespace impl {

class NVF_API IrContainer : public PolymorphicBase {
 protected:
  // Constructors
  explicit IrContainer();

  IrContainer(const IrContainer& other) = delete;
  IrContainer(IrContainer&& other) noexcept = delete;
  IrContainer& operator=(const IrContainer& other) = delete;
  IrContainer& operator=(IrContainer&& other) noexcept = delete;

  ~IrContainer() override;

 protected:
  std::unique_ptr<IrStorage> ir_storage_;
};

} // namespace impl
} // namespace nvfuser

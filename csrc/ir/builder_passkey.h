// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class IrContainer;

// Passkey for builder to register properties with statements, and to call
// functions in IrContainer
class IrBuilderPasskey {
  friend class IrBuilder;

 public:
  // TODO: Collapse ir_container and Kernel once Kernel inherits from
  // IrContainer
  IrContainer* const ir_container_ = nullptr;

 private:
  explicit IrBuilderPasskey(IrContainer* ir_container)
      : ir_container_(ir_container) {}
};

} // namespace nvfuser

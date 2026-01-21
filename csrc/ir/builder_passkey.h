// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class IrInterface;

// Passkey for builder to register properties with statements, and to call
// functions in IrContainer
class IrBuilderPasskey {
  friend class IrBuilder;

 public:
  IrInterface* const ir_interface_ = nullptr;

 private:
  explicit IrBuilderPasskey(IrInterface* ir_interface)
      : ir_interface_(ir_interface) {}
};

} // namespace nvfuser

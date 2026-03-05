// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class Fusion;

// Passkey for builder to register properties with statements, and to call
// functions in IrContainer (now via Fusion)
class IrBuilderPasskey {
  friend class IrBuilder;

 public:
  Fusion* const ir_container_ = nullptr;

 private:
  explicit IrBuilderPasskey(Fusion* ir_container)
      : ir_container_(ir_container) {}
};

} // namespace nvfuser

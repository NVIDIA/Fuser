// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <host_ir/container.h>
#include <ir/base_nodes.h>
#include <multidevice/communication.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

class HostIrLower {
 public:
  static bool canLower(Expr* expr);

  // Lower a sharded Expr into a series of Communication.
  static std::vector<Expr*> lower(Expr* c);

  static std::unique_ptr<hir::HostIrContainer> lower(
      std::unique_ptr<Fusion> fusion,
      int64_t my_device_index);
};

} // namespace nvfuser

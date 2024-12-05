// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/base_nodes.h>
#include <multidevice/communication.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

class HostIrLower {
 public:
  // Returns whether we support transforming a given expression into a series
  // of communication.
  static bool canLower(Expr* expr);

  // Lower a sharded Expr into a series of Communication.
  static std::vector<Expr*> lower(Expr* c);
};

} // namespace nvfuser

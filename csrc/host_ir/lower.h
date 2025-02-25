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

struct HostIrLowerParams {
  CommunicatorBackend communicator_backend = CommunicatorBackend::kNccl;
};

class HostIrLower {
 public:
  explicit HostIrLower(const HostIrLowerParams& params = HostIrLowerParams())
      : params_(params) {}

  // The flag `ignore_inner_resharding` is useful because the preseg passes
  // `InsertReshardingsPass` and `ReorderShardedAxisPass` want different
  // behaviors
  static bool canLower(Expr* expr, bool ignore_inner_resharding = false);

  // Lower a sharded Expr into a series of Communication.
  std::vector<Expr*> lower(Expr* c);

  std::unique_ptr<hir::HostIrContainer> lower(
      std::unique_ptr<Fusion> fusion,
      int64_t my_device_index);

 private:
  std::vector<Expr*> lowerToCollectiveBasedPipelinedGemmComm(Expr* expr);
  const HostIrLowerParams params_;
};

} // namespace nvfuser

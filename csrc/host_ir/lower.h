// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion_segmenter.h>
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
  std::vector<Expr*> lower(Expr* c, DeviceIdxType my_device_index);

  std::unique_ptr<hir::HostIrContainer> lower(
      std::unique_ptr<Fusion> fusion,
      DeviceIdxType my_device_index);

  static bool isLowerableAsStandaloneHostOp(Expr* expr);

  static bool shouldMergeSegmentedGroups(
      SegmentedGroup* group1,
      SegmentedGroup* group2);

 private:
  const HostIrLowerParams params_;
};

namespace hir_pass {

std::vector<Expr*> convertSingleOpToCommunication(
    Expr* c,
    DeviceIdxType my_device_idx,
    const HostIrLowerParams& params);

} // namespace hir_pass

} // namespace nvfuser

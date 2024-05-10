// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <val_graph.h>

namespace nvfuser {

class IdModel;
struct StatefulInliningInfo;

class LoopPromotionMapBuilder {
 public:
  // Build a map of loop groups to IterDomains that represent actual
  // loops. The map is built based on the broadcast resolution with
  // root domains between inlined producer and consumer tensors.
  static std::unordered_map<ValGroup, IterDomain*> get(
      IdModel& id_model,
      const StatefulInliningInfo& inlining_info);

 private:
  LoopPromotionMapBuilder(
      IdModel& id_model,
      const StatefulInliningInfo& inlining_info);

  void build();

 private:
  IdModel& id_model_;
  const StatefulInliningInfo& inlining_info_;
  std::unordered_map<ValGroup, IterDomain*> loop_promotion_map_;
};

} // namespace nvfuser

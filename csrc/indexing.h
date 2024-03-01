// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <id_model/id_model.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <type.h>

#include <unordered_map>

namespace nvfuser {

class Indexing {
 public:
  Indexing(const IdModel& id_model);

  Val* getIndex(TensorView* tv, Expr* expr);

 private:
  void buildLoopIndexMap();

 private:
  const IdModel& id_model_;
  std::unordered_map<ValGroup, Val*> loop_index_map_;
};

} // namespace nvfuser

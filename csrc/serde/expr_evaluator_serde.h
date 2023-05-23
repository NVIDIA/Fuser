// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ir/all_nodes.h>
#include <kernel.h>
#include <vector>

namespace nvfuser::serde {

class ExpressionSerde {
 public:
  ExpressionSerde() = default;

  flatbuffers::Offset<serde::NaiveValueGenerator> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      kir::Kernel* kernel);

 private:
  std::vector<Val*> all_values_;
  std::unordered_set<std::string> named_scalar_values_;
  std::unordered_set<int64_t> const_values_;
  std::unordered_set<Val*> symbolic_values_;
  std::vector<Val*> derived_values_;
};

} // namespace nvfuser::serde

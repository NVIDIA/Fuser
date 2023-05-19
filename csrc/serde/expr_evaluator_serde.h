// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <vector>
#include <ir_all_nodes.h>
#include <kernel.h>

namespace nvfuser::serde {

class ExpressionSerde {
 public:
  ExpressionSerde() = default;

  void bind(kir::Kernel* kernel);

  void bind(TensorView* tv, bool is_input = false);

  void bind(std::vector<IterDomain*> domain, bool is_input = false) {
    for (auto d : domain) {
        bind(d->extent(), is_input);
    }
  }

  void bind(Val* v, bool is_input = false) {
    if (is_input) {
        input_values_.push_back(v);
    }
    all_values_.push_back(v);
  }

  void generate();

 private:
  void bindInputs(kir::Kernel* kernel);

  std::vector<Val*> all_values_; 
  std::vector<Val*> input_values_; 
};

} // namespace nvfuser::serde

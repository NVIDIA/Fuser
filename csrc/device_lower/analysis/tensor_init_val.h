// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dispatch.h>

#include <unordered_map>

namespace nvfuser {

class Fusion;
class Val;
class TensorView;

// Certain ops need inputs and outputs to be initialized with some
// values. This class holds a map of default values of tensors.
//
// For example, to lower the ArgsortOp, the current implementation
// requires the input tensor to be initialized with the maximum value
// if it's a descending sort.
class TensorInitVal : public OptOutDispatch {
 public:
  TensorInitVal(Fusion* fusion);

  Val* get(TensorView* tv) const;

 private:
  void handle(ArgsortOp* aop) final;

  void handle(ScanOp* sop) final;

  void handle(TopKOp* top) final;

  void registerInitVal(TensorView* tv, Val* val);

 private:
  std::unordered_map<TensorView*, Val*> init_val_map_;
};

} // namespace nvfuser

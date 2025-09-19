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

class TensorDefaultVal : public OptOutDispatch {
 public:
  TensorDefaultVal(Fusion* fusion);
  
  Val* get(TensorView* tv) const;

 private:
  void handle(ArgsortOp* aop) final;

  void registerDefaultVal(TensorView* tv, Val* val);

 private:
  std::unordered_map<TensorView*, Val*> default_val_map_;
};

} // namespace nvfuser

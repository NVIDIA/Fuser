// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/analysis/default_val.h>
#include <fusion.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <ops/utils.h>

namespace nvfuser {

TensorDefaultVal::TensorDefaultVal(Fusion* fusion) {
  for (auto expr: fusion->exprs()) {
    dispatch(expr);
  }
}

void TensorDefaultVal::handle(ArgsortOp* aop) {
  // It is already validated that the input is exclusively used by
  // this argsort op, so it's free to initialize it for this op
  auto inp_tv = ir_utils::getTvInput(aop);
  
  Val* default_val = nullptr;
  if (aop->isDescending()) {
    default_val = ops::getMinimumValue(inp_tv->dtype());
  } else {
    default_val = ops::getMaximumValue(inp_tv->dtype());
  }

  registerDefaultVal(inp_tv, default_val);
}

void TensorDefaultVal::registerDefaultVal(TensorView* tv, Val* val) {
  auto inserted = default_val_map_.emplace(tv, val).second;
  if (!inserted) {
    NVF_ERROR(default_val_map_[tv]->sameAs(val),
              "Duplicate setting of default val for ", tv->toString(),
              ". ", default_val_map_[tv]->toString(), " vs ",
              val->toString());
  }
}

Val* TensorDefaultVal::get(TensorView* tv) const {
  auto it = default_val_map_.find(tv);
  if (it != default_val_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

} // namespace nvfuser

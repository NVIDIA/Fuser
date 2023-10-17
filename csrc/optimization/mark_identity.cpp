// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>
#include <vector>

#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <optimization/mark_identity.h>

namespace nvfuser::optimization {

namespace {

bool isContiguous(const TensorView& tv) {
  NVF_ERROR(tv.nDims() == tv.getContiguity().size());
  for (size_t i = 0; i < tv.nDims(); i++) {
    if (!tv.axis(i)->isBroadcast() && !tv.getContiguity()[i]) {
      return false;
    }
  }
  return true;
}

void findAliasesOfSource(
    const TensorView* source,
    AliasAnalysisResult& alias_to_source) {
  std::queue<const TensorView*> q;
  if (!source->hasAllocation() && isContiguous(*source)) {
    q.push(source);
  }

  while (!q.empty()) {
    const TensorView* in_tv = q.front();
    q.pop();

    for (Expr* use : in_tv->uses()) {
      if (!use->isA<ViewOp>()) {
        continue;
      }

      Val* out = use->output(0);
      TensorView* out_tv = dynamic_cast<TensorView*>(out);
      if (out_tv == nullptr) {
        continue;
      }

      if (!out_tv->hasAllocation() && isContiguous(*out_tv)) {
        q.push(out_tv);
        alias_to_source[out_tv] = in_tv;
      }
    }
  }
}

} // namespace

AliasAnalysisResult findAliases(const Fusion& fusion) {
  fusion.print();

  AliasAnalysisResult alias_to_source;
  for (const Val* in : fusion.inputs()) {
    if (const TensorView* in_tv = dynamic_cast<const TensorView*>(in)) {
      findAliasesOfSource(in_tv, alias_to_source);
    }
  }
  return alias_to_source;
}

} // namespace nvfuser::optimization

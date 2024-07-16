// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/propagate_shardings.h>

#include <vector>

#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>

namespace nvfuser::preseg_passes {

void PropagateShardingsPass::runPass(Fusion* fusion) {
  for (Expr* expr : fusion->exprs()) {
    const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    const auto& outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    if (inputs.empty()) {
      continue;
    }
    TensorView* input_with_mesh = nullptr;
    for (TensorView* tv : inputs) {
      NVF_CHECK(
          tv->hasDeviceMesh(),
          "Tensor ",
          tv->toString(),
          " should be assigned a DeviceMesh");
      if (input_with_mesh == nullptr) {
        input_with_mesh = tv;
      }
    }

    std::vector<TensorView*> outputs_without_mesh;
    for (TensorView* tv : outputs) {
      if (!tv->hasDeviceMesh()) {
        outputs_without_mesh.push_back(tv);
      }
    }
    shardAllLike(input_with_mesh, outputs_without_mesh);
  }
}

} // namespace nvfuser::preseg_passes

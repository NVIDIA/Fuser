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
#include <ir/iostream.h>
#include <ir/utils.h>
#include <multidevice/utils.h>

namespace nvfuser::preseg_passes {

namespace {
void validateMeshes(Fusion* fusion) {
  // Validate that meshes are assigned to all TensorViews or none.
  TensorView* tv_with_mesh = nullptr;
  TensorView* tv_without_mesh = nullptr;
  for (TensorView* tv : fusion->allTvs()) {
    auto update_if_null = [](TensorView*& lhs, TensorView* rhs) {
      if (lhs == nullptr) {
        lhs = rhs;
      }
    };

    if (tv->isCpuScalar()) {
      continue;
    }

    if (tv->hasDeviceMesh()) {
      update_if_null(tv_with_mesh, tv);
    } else {
      update_if_null(tv_without_mesh, tv);
    }
  }
  NVF_CHECK(
      tv_with_mesh == nullptr || tv_without_mesh == nullptr,
      "Found ",
      tv_with_mesh,
      " assigned a mesh and ",
      tv_without_mesh,
      " not.");
}
} // namespace

void PropagateShardingsPass::runPass(Fusion* fusion) {
  const std::vector<Expr*>& exprs = fusion->exprs();
  for (Expr* expr : exprs) {
    const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    auto i = std::find_if(
        inputs.begin(), inputs.end(), std::mem_fn(&TensorView::hasDeviceMesh));
    if (i == inputs.end()) {
      continue;
    }
    TensorView* input_with_mesh = *i;

    // Note: Tvs without a mesh are assumed to have no manual sharding
    // annotation and are sharded like the first producer Tv.
    const auto& outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    std::vector<TensorView*> outputs_without_mesh;
    for (auto* tv : outputs) {
      if (!tv->hasDeviceMesh()) {
        outputs_without_mesh.push_back(tv);
      }
    }
    shardAllLike(input_with_mesh, outputs_without_mesh);
  }

  // Back-propagate device meshes. This makes sure all TensorViews have a mesh
  // if any of them has one. This is needed in addition to the forward
  // propagation for ops that don't take any TensorView operands, e.g.,
  // `uniform` used in dropout. See MultiDeviceTest.BackpropMeshes for an
  // example.
  for (auto i_expr = exprs.rbegin(); i_expr != exprs.rend(); i_expr++) {
    Expr* expr = *i_expr;
    const auto& outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    auto i_output = std::find_if(
        outputs.begin(),
        outputs.end(),
        std::mem_fn(&TensorView::hasDeviceMesh));
    if (i_output == outputs.end()) {
      continue;
    }
    TensorView* output_with_mesh = *i_output;

    const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto* tv : inputs) {
      if (!tv->hasDeviceMesh()) {
        tv->setDeviceMesh(output_with_mesh->getDeviceMesh());
      }
    }
  }

  validateMeshes(fusion);
}

} // namespace nvfuser::preseg_passes

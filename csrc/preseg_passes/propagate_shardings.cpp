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
  auto num_device_parallel_dimensions = [](const TensorView* tv) -> int64_t {
    return std::count_if(
        tv->getLoopDomain().begin(),
        tv->getLoopDomain().end(),
        std::mem_fn(&IterDomain::isDeviceDim));
  };

  const std::vector<Expr*>& exprs = fusion->exprs();
  for (Expr* expr : exprs) {
    const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    // Pick the "most parallel" input tensor as the reference. This is useful
    // for propagating tensor parallelism from weights to MLP's intermediate
    // tensors. For example,
    //
    //   x: [b, s, h]; replicated.
    //   w0: [h, 4*h]; column-wise sharded.
    //   w1: [4*h, h]; row-wise sharded.
    //   y = matmul(x, w0)
    //   z = matmul(y, w1)
    //
    // With the above heuristic, `y` can be automatically sharded column-wise.
    TensorView* ref_input = nullptr;
    auto max_num_dids = std::numeric_limits<int64_t>::min();
    for (auto* input : inputs) {
      if (!input->hasDeviceMesh()) {
        continue;
      }
      int64_t num_dids = num_device_parallel_dimensions(input);
      if (num_dids > max_num_dids) {
        max_num_dids = num_dids;
        ref_input = input;
      }
    }
    if (ref_input == nullptr) {
      continue;
    }

    // Note: Tvs without a mesh are assumed to have no manual sharding
    // annotation and are sharded like the first producer Tv.
    const auto& outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    std::vector<TensorView*> outputs_without_mesh;
    for (auto* tv : outputs) {
      if (!tv->hasDeviceMesh()) {
        outputs_without_mesh.push_back(tv);
      }
    }
    shardAllLike(ref_input, outputs_without_mesh);
  }

  // shardAllLike, which calls parallelAllLke, tries to DID-parallelize
  // reduction dimensions. For example,
  //
  //   [iDID{i1}, i2] -> (Reduce) -> [r{i1}, i2] -> (Pointwise) -> [i2]
  //
  // becomes
  //
  //   [iDID{i1}, i2] -> (Reduce) -> [rDID{i1}, i2] -> (Pointwise) -> [i2]
  //
  // This implies that the reduction result only exists on the "home" device.
  // `lower_communication` can't lower such a reduction today. lowerToReduce
  // is closest but it uses the output device mesh to indicate the home device.
  // Also, an extra broadcast will be needed to replicate the reduction result
  // to all devices for the pointwise op.
  //
  // Therefore, instead, we remove the DID from reduction dimensions and
  // therefore reset them to Serial. This way,
  // the above becomes
  //
  //   [iDID{i1}, i2] -> (Reduce) -> [r{i1}, i2] -> (Pointwise) -> [i2]
  //
  // where the reduction will be lowered to an Allreduce.
  //
  // Alternatively, @naoyam proposed to represent an allreduce as a reduce
  // followed by a broadcasting set.
  //
  //   [iDID{i1}, i2] -> (Reduce) -> [rDID{i1}, i2] -> (Set) [i2] -> (Pointwise)
  //   -> [i2]
  //
  // This will make the semantics similar to other parallel types and therefore
  // we can better leverage existing parallelization utilities. We have yet to
  // pursue this because of implementation difficulty -- `lower_communication`
  // would need to match the reduce-set pattern.
  for (TensorView* tv : fusion->allTvs()) {
    for (IterDomain* id : tv->getLoopDomain()) {
      if (id->isReduction() && id->isDeviceDim()) {
        id->parallelize(ParallelType::Serial);
      }
    }
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

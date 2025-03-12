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
#include <transform_replay.h>

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

std::pair<std::unordered_set<IterDomain*>, std::unordered_set<IterDomain*>> getReshapedIds(ViewOp* view_op, const std::unordered_map<IterDomain*, IterDomain*>& c2p) {
  std::unordered_set<IterDomain*> p_reshaped_ids; // Reshaped logical IDs
  std::unordered_set<IterDomain*> c_reshaped_ids; // Reshaped root IDs

  TensorView* consumer = view_op->out();
  std::vector<IterDomain*> c_root_domain = consumer->getMaybeRootDomain();
  
  for (auto id : consumer->getLogicalDomain()) {
    if (id->isRFactorProduct() && id->definition() &&
        !id->definition()->isA<Resize>()) {
      auto root_ids = getInputsInTargetDomain(id, c_root_domain);
      for (auto root_id : root_ids) {
        c_reshaped_ids.insert(root_id);
      }
    }
  }
  // Get the logical iterdomains in the producer that are reshaped.
  std::unordered_set<IterDomain*> p_reshape_ids;
  for (auto id : c_reshaped_ids) {
    if (auto p_id = c2p.find(id); p_id != c2p.end()) {
      p_reshaped_ids.insert(p_id->second);
    }
  }
  return std::make_pair(p_reshaped_ids, c_reshaped_ids);
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

    if (outputs_without_mesh.empty()) {
      continue;
    }

    // This restricts the transform propagation to the DID axis.
    int64_t did_pos = reorderDIDToFront(ref_input);

    if (ViewOp* view_op = dynamic_cast<ViewOp*>(expr)) {
      // This implementation asserts that only one sharding is applied on the reshaped ids.
      // Inner split is not supported.
      // The cases are:
      // 1. Split reshape: [h] -> [a, h/a]. Sharding on h is applied to a in consumer.
      // 2. Merge reshape: [a, h/a] -> [h]. Sharding on a is applied to h in consumer.
      // An improvement is to support mult-levels of sharding (not a real case in practice) if they are all outer splits.
      // For example: [h] -> [cp, h/cp] -> [cp, tp, h/(cp*tp)]

      TensorView* producer = view_op->in();
      TensorView* consumer = view_op->out();

      const std::unordered_map<IterDomain*, IterDomain*>& c2p = PairwiseLogicalDomainMap(producer, consumer).mapConsumerToProducer();
      auto [p_reshaped_ids, c_reshaped_ids] = getReshapedIds(view_op, c2p);
      
      auto p_loop_domain = producer->getLoopDomain();
      auto c_loop_domain = consumer->getLoopDomain();

      for (auto idx: c10::irange(did_pos)) {
        auto p_transforms = DependencyCheck::getAllExprsBetween(
            {p_reshaped_ids.begin(), p_reshaped_ids.end()}, {p_loop_domain.at(idx)});
        if (p_transforms.empty()) {
          // Sharding is not on reshaped ids. We will use the TransformPropagator.
          continue;
        }

        NVF_ERROR(p_transforms.size() == 1 && p_transforms.back()->isA<Split>(), "Expected only a single DID split on reshaped ids.");
        auto* p_did_split = p_transforms.front()->as<Split>();

        auto reshape_transform = DependencyCheck::getAllExprsBetween(
            {c_reshaped_ids.begin(), c_reshaped_ids.end()}, {consumer->getLogicalDomain().begin(), consumer->getLogicalDomain().end()});

        NVF_ERROR((reshape_transform.size() == 1 && reshape_transform.front()->isOneOf<Split, Merge>()), "Expected a split or merge transform between root and logical reshaped ids.");

        if (reshape_transform.front()->isA<Merge>()){
          // Check that the sharding is on the outer reshaped id. If it is on inner reshaped id (h/a for merge reshape), for non-resharding, the consumer should be inner split which is not supported.
          auto* outer_id = reshape_transform.front()->as<Merge>()->outer();
          auto* producer_outer_id = c2p.at(outer_id);
          NVF_ERROR(p_did_split->in() == producer_outer_id, "Expected the sharding to be on the outer reshaped id.");
        }

        auto* c_sharded_id = reshape_transform.front()->isA<Split>() ? reshape_transform.front()->as<Split>()->outer() : reshape_transform.front()->as<Merge>()->out();

        int64_t sharded_axis = std::distance(
          c_loop_domain.begin(),
          std::find(c_loop_domain.begin(),
                    c_loop_domain.end(),
                    c_sharded_id));
        
        Val* split_factor = p_did_split->factor();
        consumer->split(sharded_axis, split_factor, /*inner_split=*/false);
        consumer->axis(sharded_axis)->parallelize(ParallelType::DIDx);

        // Move this did_pos to the end in producer to avoid using TransformPropagator on it.
        // producer->reorder({{idx, -1}});
      }
      did_pos = did_pos - 1;
    }
    
    // Propagate the DID loop split to the outputs without mesh.
    TransformPropagator propagator(ref_input, did_pos);
    SetSelector selector(
        {outputs_without_mesh.begin(), outputs_without_mesh.end()});
    MaxLogicalDomainInfoSpanningTree(ref_input, &selector)
        .traverse(&propagator);
    
    // Apply parallelization on the outputs without mesh.
    shardAllLike(ref_input, outputs_without_mesh);
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

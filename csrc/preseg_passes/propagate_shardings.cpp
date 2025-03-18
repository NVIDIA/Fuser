// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/propagate_shardings.h>

#include <vector>
#include "type.h"

#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <scheduler/utils.h>
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

std::pair<std::unordered_set<IterDomain*>, std::unordered_set<IterDomain*>>
getReshapedIds(
    ViewOp* view_op,
    const std::unordered_map<IterDomain*, IterDomain*>& c2p) {
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

  for (auto id : c_reshaped_ids) {
    if (auto p_id = c2p.find(id); p_id != c2p.end()) {
      p_reshaped_ids.insert(p_id->second);
    }
  }
  return std::make_pair(p_reshaped_ids, c_reshaped_ids);
}

int64_t num_device_dims(TensorView* tv) {
  return std::count_if(
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      std::mem_fn(&IterDomain::isDeviceDim));
}

// Order the inputs of the expression based on their priority.
// For linear op, we use weights and bias before input.
// For matmul op, we use weights before input.
// For other ops, we sort the inputs by the number of device dimensions in
// descending order.
std::vector<TensorView*> getOrderedReferenceInputs(Expr* expr) {
  const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
  if (LinearOp* linear_op = dynamic_cast<LinearOp*>(expr)) {
    // Use weights and bias before input.
    return {linear_op->inB(), linear_op->bias(), linear_op->inA()};
  }

  if (MatmulOp* matmul_op = dynamic_cast<MatmulOp*>(expr)) {
    // Use weights before input.
    return {matmul_op->inB(), matmul_op->inA()};
  }

  // Sort inputs by number of device dimensions in descending order
  std::vector<TensorView*> sorted_inputs(inputs.begin(), inputs.end());
  std::sort(
      sorted_inputs.begin(),
      sorted_inputs.end(),
      [&](TensorView* a, TensorView* b) {
        return num_device_dims(a) > num_device_dims(b);
      });

  return sorted_inputs;
}

std::vector<TensorView*> getOutputsWithoutMesh(Expr* expr) {
  const auto& outputs = ir_utils::filterByType<TensorView>(expr->outputs());
  std::vector<TensorView*> outputs_without_mesh;
  std::copy_if(
      outputs.begin(),
      outputs.end(),
      std::back_inserter(outputs_without_mesh),
      [](TensorView* tv) { return !tv->hasDeviceMesh(); });
  return outputs_without_mesh;
}

class PropagateShardingsSelector : public SetSelector {
 private:
  bool allow_c2p_;
  bool allow_p2c_;

 public:
  explicit PropagateShardingsSelector(
      const std::unordered_set<TensorView*>& selected_tvs,
      bool allow_c2p = true,
      bool allow_p2c = true)
      : SetSelector(selected_tvs),
        allow_c2p_(allow_c2p),
        allow_p2c_(allow_p2c) {}

  bool allowC2P(TensorView* from, TensorView* to) override {
    return allow_c2p_ && SetSelector::allowC2P(from, to);
  }

  bool allowP2C(TensorView* from, TensorView* to) override {
    return allow_p2c_ && SetSelector::allowP2C(from, to);
  }
};

void splitLike(
    TensorView* tv,
    int64_t axis,
    Split* ref_split,
    bool allow_inner_split = false) {
  auto split_factor = ref_split->factor();
  auto inner_split = ref_split->innerSplit();
  NVF_ERROR(!inner_split || allow_inner_split, "Inner split is not supported.");
  tv->split(axis, split_factor, /*inner_split=*/inner_split);
}

// Returns the number of DID axis on reshaped ids that were propagated to the
// consumer.
int64_t handleViewOp(ViewOp* view_op, int64_t num_device_dims) {
  // This implementation asserts that only one sharding is applied on the
  // reshaped ids. Inner split is not supported. The cases are:
  // 1. Split reshape: [h] -> [a, h/a]. Sharding on h is applied to a in
  // consumer.
  // 2. Merge reshape: [a, h/a] -> [h]. Sharding on a is applied to h in
  // consumer.
  // 3. Multiple splits or merge reshapes: [x, y, z] -> [xyz]. Sharding on x and
  // xyz. Similarly for the corresponding split reshape.
  // 4. Independent splits or merge reshapes: [w, x, y, z] -> [wx, yz]. Sharding
  // is on w and y. In the consumer, it is applied to wx and yz. An improvement
  // is to support mult-levels of sharding (not a real case in practice) if they
  // are all outer splits. For example: For the reshape [h] -> [a, h/a] where
  // the h is sharded twice: [h] -> [cp, h/cp] -> [cp, tp, h/(cp*tp)]

  TensorView* producer = view_op->in();
  TensorView* consumer = view_op->out();

  const std::unordered_map<IterDomain*, IterDomain*>& c2p =
      PairwiseLogicalDomainMap(producer, consumer).mapConsumerToProducer();
  const std::unordered_map<IterDomain*, IterDomain*>& p2c =
      PairwiseLogicalDomainMap(producer, consumer).mapProducerToConsumer();
  auto [p_logical_reshaped_ids, c_root_reshaped_ids] =
      getReshapedIds(view_op, c2p);

  auto p_loop_domain = producer->getLoopDomain();
  auto c_loop_domain = consumer->getLoopDomain();
  auto c_logical_domain = consumer->getLogicalDomain();

  // Track number of DID axis on reshaped ids that were propagated to the
  // consumer. These will not be included in TransformPropagator.
  int64_t num_reshape_shardings = 0;

  for (auto idx : c10::irange(num_device_dims)) {
    IterDomain* p_did = p_loop_domain.at(idx);
    NVF_ERROR(p_did->isDeviceDim());

    auto p_transforms = DependencyCheck::getAllExprsBetween(
        {p_logical_reshaped_ids.begin(), p_logical_reshaped_ids.end()},
        {p_loop_domain.at(idx)});

    if (p_transforms.empty()) {
      // This did axis is not on reshaped ids. We will use the
      // TransformPropagator.
      continue;
    }

    if (p_transforms.size() > 1) {
      // This reshape has been transformed.
      // We will attempt to use TransformPropagator for this did axis.
      continue;
    }

    NVF_ERROR(
        p_transforms.front()->isA<Split>(),
        "Expected a split transform producing the did axis.");
    NVF_ERROR(
        TensorDomain::sameAs(c_logical_domain, c_loop_domain),
        "Sharding a previously transformed reshape is not supported.");

    num_reshape_shardings++;

    // Find the producer logical id that is sharded.
    // We expect the outermost reshaped id to be sharded and follow the
    // outermost path traversing the transforms
    auto* p_did_split = p_did->definition()->as<Split>();
    IterDomain* p_logical_did = p_did_split->in();

    // Find the mapping of the corresponding producer logical id in consumer
    // root.
    IterDomain* c_root_did = p2c.at(p_logical_did);

    // Get the reshape transforms corresponding to this root id.
    // We use the c_root_did to only find the reshape IDs related to this did.
    auto reshape_transforms = DependencyCheck::getAllExprsBetween(
        {c_root_did},
        {consumer->getLogicalDomain().begin(),
         consumer->getLogicalDomain().end()});

    // Obtain the logical axis sharded in the consumer.
    IterDomain* c_logical_did = c_root_did;
    for (auto transform : reshape_transforms) {
      if (transform->isA<Split>()) {
        c_logical_did = transform->as<Split>()->outer();
      }
      if (transform->isA<Merge>()) {
        NVF_ERROR(
            c_logical_did == transform->as<Merge>()->outer(),
            "Expected the sharding to be on the outer reshaped id.");
        c_logical_did = transform->as<Merge>()->out();
      }
    }

    int64_t sharded_axis = std::distance(
        c_loop_domain.begin(),
        std::find(c_loop_domain.begin(), c_loop_domain.end(), c_logical_did));

    splitLike(consumer, sharded_axis, p_did_split);
    consumer->axis(sharded_axis)->parallelize(p_did->getParallelType());

    // Move this did_pos to the end in producer to avoid using
    // TransformPropagator on it.
    producer->reorder({{idx, -1}});
  }
  return num_reshape_shardings;
}

void reorderAllAsLogicalMap(std::vector<TensorView*> tvs) {
  for (auto tv : tvs) {
    tv->reorder(scheduler_utils::domainReorderAsLogicalMap(tv));
  }
}

} // namespace

void PropagateShardingsPass::runPass(Fusion* fusion) {
  const std::vector<Expr*>& exprs = fusion->exprs();

  for (Expr* expr : exprs) {
    // Note: Tvs without a mesh are assumed to have no manual sharding
    // annotation and are sharded like the first producer Tv.
    const auto& outputs_without_mesh = getOutputsWithoutMesh(expr);
    if (outputs_without_mesh.empty()) {
      continue;
    }

    const auto& reference_inputs = getOrderedReferenceInputs(expr);

    std::unordered_set<ParallelType> output_parallel_types;

    // Propagate shardings from reference inputs in order.
    for (auto* ref_input : reference_inputs) {
      // Skip if the input has no device dimensions or is nullptr.
      if (ref_input == nullptr || num_device_dims(ref_input) == 0) {
        continue;
      }

      // This restricts the transform propagation to the DID axis.
      int64_t num_device_dims = reorderDIDToFront(ref_input);

      for (auto idx : c10::irange(num_device_dims)) {
        if (output_parallel_types.count(
                ref_input->axis(idx)->getParallelType())) {
          // Do not propagate parallel types already seen on the output.
          ref_input->reorder({{idx, -1}});
          num_device_dims--;
        }
      }

      if (ViewOp* view_op = dynamic_cast<ViewOp*>(expr)) {
        int64_t num_reshape_shardings = handleViewOp(view_op, num_device_dims);
        num_device_dims = num_device_dims - num_reshape_shardings;
      }

      // Propagate the DID loop split to the outputs without mesh.
      TransformPropagator propagator(ref_input, num_device_dims);
      PropagateShardingsSelector selector(
          {outputs_without_mesh.begin(), outputs_without_mesh.end()},
          /*allow_c2p=*/false,
          /*allow_p2c=*/true);
      MaxLogicalDomainInfoSpanningTree(ref_input, &selector)
          .traverse(&propagator);

      // Apply parallelization on the outputs without mesh.
      shardAllLike(ref_input, outputs_without_mesh);
      // Reorder the loop as logical domain since the transform propagator may
      // have reordered the iterdomains in loop domain. For example: Consider
      // linear op: in = [b, m, k] weight = [DIDx(d), n/d, k] After
      // transformation, the loop domain of linear output is [DIDx(d), n/d, b,
      // m, r{k}] Since we later set the allocation domain to be loop domain, we
      // reorder the loop domain as logical domain.
      reorderAllAsLogicalMap(outputs_without_mesh);
      for (auto idx : c10::irange(num_device_dims)) {
        output_parallel_types.insert(ref_input->axis(idx)->getParallelType());
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

    // All outputs of an expression are uniformly sharded so we pick the first
    // one.
    // TODO: Do we need to worry about the case where the outputs are not
    // uniformly sharded? The relevant exprs are Welford and SDPA.
    TensorView* ref_output = *i_output;
    int64_t did_pos = reorderDIDToFront(ref_output);

    const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    std::vector<TensorView*> unsharded_inputs;
    std::copy_if(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(unsharded_inputs),
        [](TensorView* tv) {
          return !tv->hasDeviceMesh() || num_device_dims(tv) == 0;
        });

    // Note: We do not have to manually shard for reshape here.
    // TransformPropagator can handle reshapes when going from consumer to
    // producer.
    TransformPropagator propagator(ref_output, did_pos);
    PropagateShardingsSelector selector(
        {unsharded_inputs.begin(), unsharded_inputs.end()},
        /*allow_c2p=*/true,
        /*allow_p2c=*/false);
    MaxLogicalDomainInfoSpanningTree(ref_output, &selector)
        .traverse(&propagator);
    shardAllLike(ref_output, unsharded_inputs, /*parallelize_inputs=*/true);
  }

  validateMeshes(fusion);
}

} // namespace nvfuser::preseg_passes

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

int64_t numDeviceDims(TensorView* tv) {
  return std::count_if(
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      std::mem_fn(&IterDomain::isDeviceDim));
}

// Sort the given tvs by the number of device dimensions in descending order.
// Break ties by the total number of dimensions.
// Only includes TensorViews that have a device mesh.

template <typename Range>
std::vector<TensorView*> filterTvsWithMesh(const Range& tvs) {
  std::vector<TensorView*> tvs_with_mesh;
  std::copy_if(
      tvs.begin(),
      tvs.end(),
      std::back_inserter(tvs_with_mesh),
      [](TensorView* tv) { return tv != nullptr && tv->hasDeviceMesh(); });
  return tvs_with_mesh;
}

template <typename Range>
std::vector<TensorView*> sortTvsByDeviceDims(const Range& tvs) {
  // Filter out TVs without a device mesh
  std::vector<TensorView*> tvs_with_mesh = filterTvsWithMesh(tvs);

  // Then sort the filtered TVs
  std::sort(tvs_with_mesh.begin(), tvs_with_mesh.end(), [](auto a, auto b) {
    int64_t a_device_dims = numDeviceDims(a);
    int64_t b_device_dims = numDeviceDims(b);
    if (a_device_dims != b_device_dims) {
      return a_device_dims >= b_device_dims;
    }
    // Break ties by the total number of dimensions
    return a->nDims() >= b->nDims();
  });

  return tvs_with_mesh;
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
    return filterTvsWithMesh(std::vector<TensorView*>(
        {linear_op->inB(), linear_op->bias(), linear_op->inA()}));
  }

  if (MatmulOp* matmul_op = dynamic_cast<MatmulOp*>(expr)) {
    // Use weights before input.
    return filterTvsWithMesh(
        std::vector<TensorView*>({matmul_op->inB(), matmul_op->inA()}));
  }

  // Sort inputs by number of device dimensions in descending order
  std::vector<TensorView*> sorted_inputs = sortTvsByDeviceDims(inputs);

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
int64_t shardViewOp(
    ViewOp* view_op,
    std::unordered_map<int64_t, int64_t>& new2old) {
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

  // A more general approach maybe to "undo" the reshape (reverse transforms
  // from root to logical domain), followed by simplification of the consumer
  // loop domain to move DID upwards.

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
  int64_t num_device_dims = new2old.size();

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

    // TODO: Check for divisibility of the consumer axis by the split factor.
    splitLike(consumer, sharded_axis, p_did_split);
    consumer->axis(sharded_axis)->parallelize(p_did->getParallelType());

    // Move this did_pos behind the non-propagated DID axis to avoid using
    // TransformPropagator on it.
    producer->reorder({{idx, num_device_dims - 1}});
    new2old[idx] = num_device_dims - 1;
    num_device_dims--;
  }

  return num_reshape_shardings;
}

void reorderLoopAsAllocation(std::vector<TensorView*> tvs) {
  // Use maybeAllocationDomain to transform
  // Transform using exprs between logical and loop and get the map.
  for (auto tv : tvs) {
    auto alloc_dom = tv->getMaybeAllocationDomain();
    std::vector<Expr*> transform_exprs = DependencyCheck::getAllExprsBetween(
        {alloc_dom.begin(), alloc_dom.end()},
        {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
    NVF_ERROR(
        std::all_of(
            transform_exprs.begin(),
            transform_exprs.end(),
            [](Expr* expr) { return expr->isA<Split>(); }),
        "Expected all transform exprs to be a split between logical and loop domain during sharding propagation.");
    auto reorder_map = scheduler_utils::createReorderMapUnderTransforms(
        /*ids_to_reorder=*/tv->getLoopDomain(),
        /*ids_to_transform=*/alloc_dom,
        /*transform_exprs=*/transform_exprs);
    tv->reorder(reorder_map);
  }
}

// Reorder the DID axis to the front only if it does not have a parallel type
// already seen on the output (existing_parallel_types).
// Returns a map from the new position to the old position to undo the
// reordering later.
std::unordered_map<int64_t, int64_t> selectiveReorderDIDToFront(
    TensorView* tv,
    std::unordered_set<ParallelType> existing_parallel_types) {
  std::unordered_map<int64_t, int64_t> old2new;
  std::unordered_map<int64_t, int64_t> new2old;
  int64_t current_pos = 0;

  for (auto pos : c10::irange(tv->nDims())) {
    if (tv->axis(pos)->isDeviceDim() &&
        !existing_parallel_types.count(tv->axis(pos)->getParallelType())) {
      old2new[pos] = current_pos;
      new2old[current_pos] = pos;
      current_pos++;
    }
  }

  tv->reorder(old2new);
  return new2old;
}

// Updates the set of parallel types seen on the output.
void updateOutputParallelTypes(
    TensorView* tv,
    std::unordered_set<ParallelType>& output_parallel_types) {
  for (auto id : tv->getLoopDomain()) {
    if (id->isDeviceDim()) {
      output_parallel_types.insert(id->getParallelType());
    }
  }
}

} // namespace

// This presegmentation pass propagates shardings from fusion inputs to
// downstream tensorviews.
// 1. Forward propagating DID loop splits and parallelization from inputs to
// outputs that don't have a mesh using TransformPropagator
// 2. Reshape is handled manually since the DID loop split transforms conflict
// with the reshape root-to-logical transforms if using TransformPropagator
// 3. Back-propagating device meshes to ensure all TensorViews have consistent
// meshes. This also splits and parallelizes unsharded inputs based on outputs.
// See `MultiDevicePresegPassesTest.ResidualAdd` for an example.
// 4. Reorders the loop domain as the allocation order. Ideally, loop domain
// should follow logical domain and allocation domain should follow any stride
// order specified/inferred. However, we currently require loop domain to be the
// same as allocation domain.
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

    if (reference_inputs.empty()) {
      continue;
    }

    std::unordered_set<ParallelType> output_parallel_types;

    // Propagate shardings from reference inputs in order.
    for (auto* ref_input : reference_inputs) {
      // Skip if the input has no device mesh or is nullptr.
      NVF_ERROR(
          ref_input != nullptr && ref_input->hasDeviceMesh(),
          "Reference input ",
          ref_input,
          " has no device mesh.");

      // Reorder the DID axis to the front only if it does not have a parallel
      // type already seen on the output.
      std::unordered_map<int64_t, int64_t> new2old =
          selectiveReorderDIDToFront(ref_input, output_parallel_types);

      // This restricts the transform propagation to the DID axis.
      int64_t num_device_dims = new2old.size();

      if (ViewOp* view_op = dynamic_cast<ViewOp*>(expr)) {
        // Propagation of reshape will return how many DID axis were propagated.
        // They are reordered behind non-propagated DID axis and the new2old map
        // is updated.
        int64_t num_reshape_shardings = shardViewOp(view_op, new2old);
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

      updateOutputParallelTypes(ref_input, output_parallel_types);

      // Undo the reordering of the DID axis so it is in the correct order
      // again.
      ref_input->reorder(new2old);
    }

    // Reorder the loop domain since the transform propagator may
    // have reordered the iterdomains in loop domain. For example: Consider
    // linear op: in = [b, m, k] weight = [DIDx(d), n/d, k] After
    // transformation, the loop domain of linear output is [DIDx(d), n/d, b,
    // m, r{k}]. Since, we set allocation to be the same as loop, we reorder it
    // as allocation domain in the interim. Ideally, this should follow logical
    // domain and DIDx axis at the front. The allocation domain should follow
    // any stride order specified/inferred.
    reorderLoopAsAllocation(outputs_without_mesh);
  }

  // Back-propagate device meshes. This makes sure all TensorViews have a mesh
  // if any of them has one. This is needed in addition to the forward
  // propagation for ops that don't take any TensorView operands, e.g.,
  // `uniform` used in dropout. See MultiDeviceTest.BackpropMeshes for an
  // example. For non-fusion inputs, we also propagate shardings from outputs to
  // inputs. See MultiDevicePresegPassesTest.ResidualAdd for an example.
  for (auto i_expr = exprs.rbegin(); i_expr != exprs.rend(); i_expr++) {
    Expr* expr = *i_expr;

    const auto& outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    std::vector<TensorView*> sorted_outputs = sortTvsByDeviceDims(outputs);
    // All outputs of an expression (Welford, SDPA) should be uniformly sharded.
    // We pick the most parallel output as the reference.
    // This is to avoid picking seed/offset tvs in SDPA.

    if (sorted_outputs.empty()) {
      continue;
    }

    TensorView* ref_output = sorted_outputs.front();
    NVF_ERROR(
        ref_output != nullptr && ref_output->hasDeviceMesh(),
        "Reference output ",
        ref_output,
        " has no device mesh.");

    // For fusion inputs, only check if they have a device mesh. We do not
    // modify their sharding. For non-fusion inputs, check both device mesh and
    // device dims.
    const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    std::vector<TensorView*> unsharded_inputs;
    for (auto* tv : inputs) {
      if (tv->isFusionInput()) {
        if (!tv->hasDeviceMesh()) {
          tv->setDeviceMesh(ref_output->getDeviceMesh());
        }
        continue;
      }
      if (!tv->hasDeviceMesh() || numDeviceDims(tv) == 0) {
        unsharded_inputs.push_back(tv);
      }
    }

    if (unsharded_inputs.empty()) {
      continue;
    }

    std::unordered_map<int64_t, int64_t> new2old =
        selectiveReorderDIDToFront(ref_output, {});
    int64_t did_pos = new2old.size();

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
    shardAllLike(ref_output, unsharded_inputs);

    ref_output->reorder(new2old);
    reorderLoopAsAllocation(unsharded_inputs);
  }

  validateMeshes(fusion);
}

} // namespace nvfuser::preseg_passes

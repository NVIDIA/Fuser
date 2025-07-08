// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <vector>

#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <preseg_passes/propagate_shardings.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {
std::unordered_set<IterDomain*> getReshapedIds(
    ViewOp* view_op,
    const std::unordered_map<IterDomain*, IterDomain*>& c2p) {
  std::unordered_set<IterDomain*>
      p_reshaped_ids; // Reshaped producer logical IDs

  TensorView* consumer = view_op->out();
  std::vector<IterDomain*> c_root_domain = consumer->getMaybeRootDomain();

  for (auto id : consumer->getLogicalDomain()) {
    if (id->isRFactorProduct() && id->definition() &&
        !id->definition()->isA<Resize>()) {
      auto root_ids = getInputsInTargetDomain(id, c_root_domain);
      for (auto root_id : root_ids) {
        p_reshaped_ids.insert(c2p.at(root_id));
      }
    }
  }

  return p_reshaped_ids;
}

// Returns true if the given iterdomain is divisible by the split factor of the
// reference split.
bool isSplitDivisible(IterDomain* id, Split* ref_split) {
  if (!ref_split->factor()->isConstInt()) {
    return false;
  }
  auto split_factor = ref_split->factor()->evaluate().as<int64_t>();
  if (split_factor == 1) {
    return true;
  }

  if (!id->extent()->isConstInt()) {
    return false;
  }
  auto id_extent = id->extent()->evaluate().as<int64_t>();
  return id_extent % split_factor == 0;
}

void splitLike(TensorView* tv, int64_t axis, Split* ref_split) {
  NVF_ERROR(
      isSplitDivisible(tv->axis(axis), ref_split),
      "Require the sharded ID to be divisible by the split factor. Got: ",
      tv->axis(axis),
      " and split factor: ",
      ref_split->factor());
  tv->outer_split(axis, ref_split->factor());
}

int64_t getLoopAxis(TensorView* tv, IterDomain* id) {
  auto it =
      std::find(tv->getLoopDomain().begin(), tv->getLoopDomain().end(), id);
  NVF_ERROR(
      it != tv->getLoopDomain().end(),
      "Expected the id ",
      id,
      " to be in the loop domain: ",
      tv->getLoopDomain());
  return std::distance(tv->getLoopDomain().begin(), it);
}

// did_pos is the original number of device dimensions present
// in the producer loop domain and reordered to the front.
// Returns the new did_pos, the number of DID axis that were not propagated
// since they were not on reshaped ids.
int64_t shardViewOp(ViewOp* view_op, int64_t did_pos) {
  // This implementation asserts that only one sharding is applied on the
  // reshaped ids. Inner split is not supported. The cases are:
  // 1. Split reshape: [h] -> [a, h/a]. Sharding on h is applied to a in
  // consumer.
  // 2. Merge reshape: [a, h/a] -> [h]. Sharding on a is applied to h in
  // consumer.
  // 3. Multiple splits or merge reshapes: [x, y, z] -> [xyz]. Sharding on x and
  // xyz. Similarly for the corresponding split reshape.
  // 4. Independent splits or merge reshapes: [w, x, y, z] -> [wx, yz]. Sharding
  // is on w and y. In the consumer, it is applied to wx and yz.
  // An improvement is to support multi-levels of sharding (not a real case yet)
  // if they are all outer splits. For example: For the reshape [h] -> [a, h/a]
  // where the h is sharded twice: [h] -> [cp, h/cp] -> [cp, tp, h/(cp*tp)]
  // This is currently blocked by `getShardedLogicalAxis` which does not allow
  // device dimensions to be on the inner of a split.

  // A more general approach maybe to "undo" the reshape (reverse transforms
  // from root to logical domain), followed by simplification of the consumer
  // loop domain to move DID upwards.

  TensorView* producer = view_op->in();
  TensorView* consumer = view_op->out();

  const auto pairwise_map = PairwiseLogicalDomainMap(producer, consumer);
  const std::unordered_map<IterDomain*, IterDomain*>& c2p =
      pairwise_map.mapConsumerToProducer();
  const std::unordered_map<IterDomain*, IterDomain*>& p2c =
      pairwise_map.mapProducerToConsumer();
  auto p_logical_reshaped_ids = getReshapedIds(view_op, c2p);

  auto p_loop_domain = producer->getLoopDomain();

  // Track number of DID axis on reshaped ids that were propagated to the
  // consumer. These will not be included later in TransformReplay.
  int64_t num_reshape_shardings = 0;

  for (IterDomain* p_logical_reshaped_id : p_logical_reshaped_ids) {
    // Find all transforms between the producer reshaped logical id and the
    // device ids.
    auto p_transforms = DependencyCheck::getAllExprsBetween(
        {p_logical_reshaped_id},
        {p_loop_domain.begin(), p_loop_domain.begin() + did_pos});

    if (p_transforms.empty()) {
      continue;
    }

    NVF_ERROR(
        p_transforms.size() == 1 && p_transforms.front()->isA<Split>(),
        "Expected a split transform producing the device id. Got:\n",
        p_transforms);

    // Get the reshape transforms corresponding to this root id.
    IterDomain* c_root_reshaped_id = p2c.at(p_logical_reshaped_id);
    auto reshape_transforms = DependencyCheck::getAllExprsBetween(
        {c_root_reshaped_id},
        {consumer->getLoopDomain().begin(), consumer->getLoopDomain().end()});

    // Obtain the loop axis to be sharded in the consumer following the
    // outermost path.
    IterDomain* c_sharded_id = c_root_reshaped_id;
    for (auto transform : reshape_transforms) {
      if (transform->isA<Split>()) {
        c_sharded_id = transform->as<Split>()->outer();
      }
      if (transform->isA<Merge>()) {
        NVF_ERROR(
            c_sharded_id == transform->as<Merge>()->outer(),
            "Expected the sharding to be on the outer reshaped id.");
        c_sharded_id = transform->as<Merge>()->out();
      }
    }

    auto p_did_split = p_transforms.front()->as<Split>();
    IterDomain* p_did = p_did_split->outer();
    int64_t p_did_pos = getLoopAxis(producer, p_did);

    // Replaying the DID split on the outermost producer reshaped id is
    // equivalent to the DID split on the outermost consumer reshaped id if:
    // 1. The extent of sharded id is divisible by split_factor (num_devices)
    // 2. The producer reshaped id is outer split by num_devices to produce the
    // device dimension.
    NVF_ERROR(
        isSplitDivisible(p_did_split->in(), p_did_split),
        "Expected the split to be divisble. Got: ",
        p_did_split->in(),
        " and split factor: ",
        p_did_split->factor());
    NVF_ERROR(
        !p_did_split->innerSplit() && p_did_split->outer()->isDeviceDim(),
        "Expected an outer-split producing a device dimension.");

    int64_t c_did_pos = getLoopAxis(consumer, c_sharded_id);
    splitLike(consumer, c_did_pos, p_did_split);
    consumer->axis(c_did_pos)->parallelize(p_did->getParallelType());

    // Move this did_pos back to avoid using TransformReplay on it.
    producer->reorder({{p_did_pos, -1}});
    num_reshape_shardings++;
  }

  return did_pos - num_reshape_shardings;
}

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

// Sort the given tvs by the number of device dimensions in descending order.
// Break ties by the total number of dimensions.
// Only includes TensorViews that have a device mesh.
template <typename Range>
std::vector<TensorView*> sortTvsByDeviceDims(const Range& tvs) {
  // Filter out TVs without a device mesh
  std::vector<TensorView*> tvs_with_mesh = filterTvsWithMesh(tvs);

  // Then sort the filtered TVs
  std::stable_sort(
      tvs_with_mesh.begin(), tvs_with_mesh.end(), [](auto a, auto b) {
        int64_t a_device_dims = numDeviceDims(a);
        int64_t b_device_dims = numDeviceDims(b);
        if (a_device_dims != b_device_dims) {
          return a_device_dims > b_device_dims;
        }
        // Break ties by the total number of dimensions
        return a->nDims() > b->nDims();
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

// Reorder the DID axis with the given parallel types to the front.
// Returns the number of device dimensions that were reordered to the front.
// This allows us to limit propagation to only the relevant DID axis.
int64_t selectiveReorderDIDToFront(
    TensorView* tv,
    const std::unordered_set<ParallelType>& selected_parallel_types) {
  std::unordered_map<int64_t, int64_t> old2new;
  int64_t current_pos = 0;

  for (auto&& [pos, id] : enumerate(tv->getLoopDomain())) {
    if (id->isDeviceDim() &&
        selected_parallel_types.count(id->getParallelType())) {
      old2new[pos] = current_pos;
      current_pos++;
    }
  }

  tv->reorder(old2new);
  return current_pos;
}

// Returns the set of parallel types seen on the loop domain of the given tvs.
std::unordered_set<ParallelType> getParallelTypesToPropagate(
    std::vector<TensorView*> tvs) {
  // Get the set of parallel types seen on the loop domain of the given tvs.
  std::unordered_set<ParallelType> existing_parallel_types;
  for (auto tv : tvs) {
    for (auto id : tv->getLoopDomain()) {
      if (!id->isReduction() && id->isDeviceDim()) {
        existing_parallel_types.insert(id->getParallelType());
      }
    }
  }
  std::unordered_set<ParallelType> selected_parallel_types;
  for (ParallelType pt : kParallelTypeDIDs) {
    if (!existing_parallel_types.count(pt)) {
      selected_parallel_types.insert(pt);
    }
  }
  return selected_parallel_types;
}

} // namespace

// This presegmentation pass propagates shardings from fusion inputs to
// downstream tensorviews.
// 1. Forward propagating DID loop splits and parallelization from inputs to
// outputs that don't have a mesh using TransformReplay
// 2. Reshape is handled manually since the DID loop split transforms conflict
// with the reshape root-to-logical transforms if using TransformReplay
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
    // Propagate shardings from reference inputs in order.
    for (auto* ref_input : reference_inputs) {
      // Skip if the input has no device mesh or is nullptr.
      NVF_ERROR(
          ref_input != nullptr && ref_input->hasDeviceMesh(),
          "Reference input ",
          ref_input,
          " has no device mesh.");

      // Reorder the DID axis to the front only if it does not have a parallel
      // type already seen on the outputs. This avoids propagating the same
      // parallel type on multiple axis of the output when using multiple
      // reference inputs. Consider out [M, N] = linear (inp [M, K], weight (N,
      // K)) with inp sharded on M ([DIDx(d), M/d, K]) and weight sharded on N
      // ([DIDy(d), N/d, K]). We propagate from weights first, so the output
      // will be [M, DIDx(d), N/d]. When we propagate from inp next, we should
      // not propagate DIDx parallel type to the output. Otherwise, the output
      // will have multiple DIDx shardings which is invalid.
      std::unordered_set<ParallelType> selected_parallel_types =
          getParallelTypesToPropagate(outputs_without_mesh);

      // This restricts the transform propagation to only the relevant DID axis.
      int64_t did_pos =
          selectiveReorderDIDToFront(ref_input, selected_parallel_types);

      if (auto* view_op = dynamic_cast<ViewOp*>(expr)) {
        // Propagation of reshape will return how many DID axis were propagated.
        // They are reordered behind non-propagated DID axis
        did_pos = shardViewOp(view_op, did_pos);
      }

      // Propagate the DID loop split to the outputs without mesh.
      propagateDIDTransform(
          /*ref=*/ref_input,
          /*tvs=*/outputs_without_mesh,
          /*did_pos=*/did_pos,
          PropagateDirection::kForward);

      // Apply parallelization on the outputs without mesh.
      shardAllLike(ref_input, outputs_without_mesh, selected_parallel_types);
    }
  }

  // Back-propagate device meshes. This makes sure all TensorViews have a mesh
  // if any of them has one. This is needed in addition to the forward
  // propagation for ops that don't take any TensorView operands, e.g.,
  // `uniform` used in dropout. See MultiDeviceTest.BackpropMeshes for an
  // example. For non-fusion inputs, we also propagate shardings from outputs to
  // inputs. See MultiDevicePresegPassesTest.ResidualAdd for an example.
  for (Expr* expr : exprs | std::views::reverse) {
    const auto& outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    // All outputs of an expression (Welford, SDPA) should be uniformly sharded.
    // We pick the most parallel output as the reference.
    // This is to avoid picking seed/offset tvs in SDPA.
    std::vector<TensorView*> sorted_outputs = sortTvsByDeviceDims(outputs);

    if (sorted_outputs.empty()) {
      // No output with a device mesh.
      continue;
    }

    TensorView* ref_output = sorted_outputs.front();
    NVF_ERROR(
        ref_output != nullptr && ref_output->hasDeviceMesh(),
        "Reference output ",
        ref_output,
        " has no device mesh.");

    // For fusion inputs, only check if they have a device mesh. We do not
    // modify their sharding. For non-fusion inputs, we try to propagate
    // shardings from the reference output for parallel types that are not
    // already present.
    const auto& inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    std::vector<TensorView*> sharding_candidates;
    for (auto* tv : inputs) {
      if (tv->isFusionInput()) {
        if (!tv->hasDeviceMesh()) {
          tv->setDeviceMesh(ref_output->getDeviceMesh());
        }
        continue;
      }
      if (!tv->hasDeviceMesh() || numDeviceDims(tv) == 0) {
        sharding_candidates.push_back(tv);
      }
    }

    if (sharding_candidates.empty()) {
      continue;
    }

    std::unordered_set<ParallelType> selected_parallel_types =
        getParallelTypesToPropagate(sharding_candidates);
    int64_t did_pos =
        selectiveReorderDIDToFront(ref_output, selected_parallel_types);
    // Note: We do not have to manually shard for reshape here.
    // TransformReplay can handle reshapes when going from consumer to
    // producer.
    propagateDIDTransform(
        /*ref=*/ref_output,
        /*tvs=*/sharding_candidates,
        /*did_pos=*/did_pos,
        PropagateDirection::kBackward);
    shardAllLike(ref_output, sharding_candidates, selected_parallel_types);
  }
}

} // namespace nvfuser::preseg_passes

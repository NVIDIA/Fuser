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
#include <linked_hash_map.h>
#include <multidevice/utils.h>
#include <preseg_passes/propagate_shardings.h>
#include <scheduler/utils.h>

namespace nvfuser::preseg_passes {

namespace {
// Returns true if the reference split is divisible and the given id is
// divisible by the split factor.
bool isSplitDivisible(IterDomain* id, Split* ref_split) {
  if (!ref_split->isDivisible()) {
    return false;
  }

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

// Returns the set of parallel types not seen on the loop domain of the given
// tvs and hence, can be propagated.
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

// Returns true if the given id has a root-to-logical transform
// if traversed from loop domain to root domain.
bool hasRootToLogicalTransform(IterDomain* id, const TensorView* tv) {
  auto logical_ids = IterVisitor::getInputsTo(
      {id}, {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()});
  return std::any_of(
      logical_ids.begin(), logical_ids.end(), [&](Val* logical_val) {
        return logical_val->definition() != nullptr;
      });
}

bool isInDomain(IterDomain* id, const std::vector<IterDomain*>& domain) {
  return std::find(domain.begin(), domain.end(), id) != domain.end();
}

// Traverses root-to-logical transforms to find the outermost logical ID.
IterDomain* getOutermostLogicalId(IterDomain* id, const TensorView* tv) {
  auto transforms = DependencyCheck::getAllExprsBetween(
      {id}, {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()});

  for (auto transform : transforms) {
    if (transform->isA<Split>()) {
      id = transform->as<Split>()->outer();
    } else if (transform->isA<Merge>()) {
      NVF_ERROR(
          id == transform->as<Merge>()->outer(),
          "Expected the sharding to be on the outer reshaped id.");
      id = transform->as<Merge>()->out();
    }
  }
  return id;
}

std::unordered_map<IterDomain*, IterDomain*> getRef2TargetMap(
    const TensorView* ref,
    const TensorView* target,
    PropagateDirection direction) {
  if (direction == PropagateDirection::kForward) {
    return PairwiseLogicalDomainMap(ref, target).mapProducerToConsumer();
  }
  return PairwiseLogicalDomainMap(target, ref).mapConsumerToProducer();
}

// Propagates the given device ids from ref to target.
// This includes loop domain transformations and parallelization.
void transformLoopDomain(
    TensorView* target,
    const TensorView* ref,
    const std::unordered_set<IterDomain*>& device_ids,
    PropagateDirection direction) {
  std::unordered_map<IterDomain*, IterDomain*> ref2target =
      getRef2TargetMap(ref, target, direction);

  auto get_target_id = [&](IterDomain* ref_id) -> IterDomain* {
    // Finds the target id to be sharded corresponding to the ref_id.
    // This handles any root-to-logical transforms present on ref_id
    // or the mapped target_id.
    // For e.g., if ref [h] -> target [a, h/a], returns `a` for ref_id `h`.
    // Similarly, if target [h] -> ref [a, h/a], returns `h` for ref_id `a`.
    if (!ref2target.contains(ref_id)) {
      // Find the root domain id.
      std::unordered_set<IterDomain*> inputs =
          getInputsInTargetDomain({ref_id}, ref->getMaybeRootDomain());
      NVF_ERROR_EQ(
          inputs.size(),
          1,
          "Expected one input for ",
          ref_id,
          " in the root domain.");
      ref_id = *inputs.begin();
    }

    NVF_ERROR(
        ref2target.contains(ref_id),
        "Expected the ref to be in the ref2target map.");

    IterDomain* target_id = ref2target.at(ref_id);
    if (isInDomain(target_id, target->getLoopDomain())) {
      return target_id;
    }
    return getOutermostLogicalId(target_id, target);
  };

  auto validate_split = [](Split* split, IterDomain* id) -> void {
    NVF_ERROR(
        !split->innerSplit() && split->outer()->isDeviceDim(),
        "Expected the outer id of the split to be a device dimension. Got: ",
        split->outer());
    NVF_ERROR(
        split->isDivisible(),
        "Expected the split to be divisible. Got: ",
        split);
    NVF_ERROR(
        isSplitDivisible(id, split),
        "Require the sharded ID to be divisible by the split factor. Got: ",
        id,
        " and split factor: ",
        split->factor());
  };

  // Start with the original loop domain.
  LinkedHashMap<IterDomain*, std::monostate> transformed_loop;
  for (IterDomain* id : target->getLoopDomain()) {
    transformed_loop.pushBack(id, std::monostate());
  }

  std::vector<Expr*> transforms = DependencyCheck::getAllExprsBetween(
      {ref->getLogicalDomain().begin(), ref->getLogicalDomain().end()},
      {device_ids.begin(), device_ids.end()});

  for (Expr* transform : transforms) {
    auto* split = dynamic_cast<Split*>(transform);
    NVF_ERROR(
        split != nullptr,
        "Expected a split transform producing the device id. Got: ",
        transform);

    IterDomain* ref_id = split->in();
    IterDomain* target_id = get_target_id(ref_id);
    NVF_ERROR(
        transformed_loop.contains(target_id),
        "Expected the target ID, ",
        target_id,
        ", to be in the loop domain.");

    // Sharding on producer logical id is equivalent to sharding on the
    // outermost consumer reshaped id iff:
    // 1. The reference is outer split by num_devices.
    // 2. The extent of sharded id in producer / consumer is divisible by
    // split_factor. NOTE: We can only check if DID(d) is on the outer of the
    // split regardless of the split_factor. However, when applying the split to
    // the target, the split_factor will need to be num_devices. For e.g.: A[h]
    // -> reshape -> B[a, h/a] If A is inner split `h/d`, then directly
    // replaying the split on `a` will produce `a/(h/d), h/d` instead of `d,
    // a/d`. So we should instead outer split by num_devices.

    // Find the consumer between the reference and target.
    auto [consumer_id, consumer_tv] = direction == PropagateDirection::kForward
        ? std::make_pair(target_id, target)
        : std::make_pair(ref_id, ref);

    if (hasRootToLogicalTransform(consumer_id, consumer_tv)) {
      validate_split(split, target_id);
    }

    auto it = transformed_loop.erase(target_id).second;
    auto [outer, inner] =
        IterDomain::split(target_id, split->factor(), split->innerSplit());

    transformed_loop.insert(it, outer, std::monostate());
    transformed_loop.insert(it, inner, std::monostate());

    // Add mapping between ref and target for the propagated DID split.
    // This is used to propagate 2D sharding and parallelization.
    ref2target[split->outer()] = outer;
    ref2target[split->inner()] = inner;
  }

  // Parallelize based on the ref2target map.
  for (IterDomain* device_id : device_ids) {
    NVF_ERROR(
        ref2target.contains(device_id),
        "Failed to propagate ",
        device_id,
        " to ",
        target);
    IterDomain* target_id = ref2target.at(device_id);
    target_id->parallelize(device_id->getParallelType());
  }

  auto new_loop = std::views::keys(transformed_loop);
  target->setLoopDomain({new_loop.begin(), new_loop.end()});
}

void propagateDIDTransform(
    const TensorView* ref,
    TensorView* tv,
    const std::unordered_set<ParallelType>& selected_parallel_types,
    PropagateDirection direction) {
  tv->setDeviceMesh(ref->getDeviceMesh());

  std::unordered_set<IterDomain*> device_ids;
  const std::unordered_map<IterDomain*, IterDomain*> ref2target =
      getRef2TargetMap(ref, tv, direction);

  for (IterDomain* device_id : ref->getLoopDomain()) {
    if (selected_parallel_types.count(device_id->getParallelType()) == 0) {
      continue;
    }
    // Get input of device_id in the root / logical domain
    // that will be present in ref2target mapping.
    std::unordered_set<IterDomain*> inputs =
        direction == PropagateDirection::kForward
        ? getInputsInTargetDomain({device_id}, ref->getLogicalDomain())
        : getInputsInTargetDomain({device_id}, ref->getMaybeRootDomain());
    NVF_ERROR_EQ(
        inputs.size(),
        1,
        "Expected one input for ",
        device_id,
        " in the root / logical domain.");
    IterDomain* ref_id = *inputs.begin();
    IterDomain* target_id = getOrDefault(ref2target, ref_id);
    if (target_id == nullptr) {
      continue;
    }

    // Get the outermost logical ID corresponding to the target ID.
    // This is the ID to which the sharding will be propagated.
    // For e.g., if ref [h] -> target [a, h/a], then the outermost logical ID
    // is `a`.
    IterDomain* outermost_target_id = getOutermostLogicalId(target_id, tv);

    if (outermost_target_id->isParallelized()) {
      continue;
    }
    // Skip if the target is not in the loop domain (i.e. further transformed).
    if (!isInDomain(outermost_target_id, tv->getLoopDomain())) {
      continue;
    }
    device_ids.insert(device_id);
  }

  transformLoopDomain(tv, ref, device_ids, direction);
}

void propagateDIDTransform(
    const TensorView* ref,
    const std::vector<TensorView*>& tvs,
    const std::unordered_set<ParallelType>& selected_parallel_types,
    PropagateDirection direction) {
  for (auto tv : tvs) {
    propagateDIDTransform(ref, tv, selected_parallel_types, direction);
  }
}

} // namespace

// This presegmentation pass propagates shardings from fusion inputs to
// downstream tensorviews.
// 1. Forward propagating DID loop splits and parallelization from inputs to
// outputs that don't have a mesh using TransformReplay
// 2. Back-propagating device meshes to ensure all TensorViews have consistent
// meshes. This also splits and parallelizes unsharded inputs based on outputs.
// See `MultiDevicePresegPassesTest.ResidualAdd` for an example.
void PropagateShardingsPass::runPass(Fusion* fusion) {
  // Any tensorview with a device mesh is considered scheduled by user and not
  // modified in this pass.
  auto user_sharded_tvs = [&]() {
    const auto all_tvs = fusion->allTvs();
    auto filtered = fusion->allTvs() |
        std::views::filter(std::mem_fn(&TensorView::hasDeviceMesh));
    return std::unordered_set<TensorView*>(filtered.begin(), filtered.end());
  }();

  const std::vector<Expr*>& exprs = fusion->exprs();

  for (Expr* expr : exprs) {
    // TensorViews without a mesh are assumed to have no user-specified sharding
    // and are sharded like the producers.
    const auto& outputs_without_mesh = getOutputsWithoutMesh(expr);
    if (outputs_without_mesh.empty()) {
      continue;
    }

    const auto& reference_inputs = getOrderedReferenceInputs(expr);
    // Propagate shardings from reference inputs in order.
    for (auto* ref_input : reference_inputs) {
      NVF_ERROR(ref_input != nullptr);
      NVF_ERROR(
          ref_input->hasDeviceMesh(),
          "Reference input ",
          ref_input,
          " has no device mesh.");

      // Consider out [M, N] = linear (inp [M, K], weight (N,
      // K)) with inp sharded on M ([DIDx(d), M/d, K]) and weight sharded on N
      // ([DIDy(d), N/d, K]). We propagate from weights first, so the output
      // will be [M, DIDx(d), N/d]. When we propagate from inp next, we should
      // not propagate DIDx parallel type to the output. Otherwise, the output
      // will have multiple DIDx shardings which is invalid.
      std::unordered_set<ParallelType> selected_parallel_types =
          getParallelTypesToPropagate(outputs_without_mesh);

      propagateDIDTransform(
          /*ref=*/ref_input,
          /*tvs=*/outputs_without_mesh,
          selected_parallel_types,
          PropagateDirection::kForward);
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
      if (user_sharded_tvs.count(tv) != 0) {
        continue;
      }
      if (tv->isFusionInput()) {
        tv->setDeviceMesh(ref_output->getDeviceMesh());
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

    propagateDIDTransform(
        /*ref=*/ref_output,
        /*tvs=*/sharding_candidates,
        selected_parallel_types,
        PropagateDirection::kBackward);
  }
}

} // namespace nvfuser::preseg_passes

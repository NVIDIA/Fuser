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
// Validates meshes and returns true if any TensorView has a device mesh.
bool validateMeshes(Fusion* fusion) {
  // Validate that meshes are assigned to all TensorViews or none.
  bool tv_with_mesh_found = false;
  bool tv_without_mesh_found = false;
  
  for (auto tv : fusion->allTvs()) {
    if (tv->isCpuScalar()) {
      continue;
    }
    tv->hasDeviceMesh() ? tv_with_mesh_found = true : tv_without_mesh_found = true;
  }
  NVF_CHECK(
      !(tv_with_mesh_found && tv_without_mesh_found),
      "Cannot have some TensorViews with device mesh and some without.");
  return tv_with_mesh_found;
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

// Custom selector to specify direction of transform propagation.
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

void reorderLoopAsAllocation(std::vector<TensorView*> tvs) {
  // Transform the maybe allocation domain to the loop domain.
  // using exprs between logical and loop and get the permutation required to
  // reorder the loop domain in the same relative order as the allocation domain.
  for (auto tv : tvs) {
    auto alloc_dom = tv->getMaybeAllocationDomain();
    // Allocation domain should be a permutation of logical domain at this point.
    std::vector<Expr*> transform_exprs = DependencyCheck::getAllExprsBetween(
        {alloc_dom.begin(), alloc_dom.end()},
        {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
    NVF_ERROR(
        std::all_of(
            transform_exprs.begin(),
            transform_exprs.end(),
            [](Expr* expr) { return expr->isA<Split>(); }),
        "Expected all transform exprs to be a split between logical and loop domain during sharding propagation.");
    scheduler_utils::applyTransforms(alloc_dom, transform_exprs);
    std::optional<std::vector<int64_t>> permutation = ir_utils::computePermutation(alloc_dom, tv->getLoopDomain());
    NVF_ERROR(
      permutation.has_value(),
      "Failed to find a valid permutation for reordering",
      tv->getLoopDomain(),
      " as ",
      alloc_dom);
    tv->reorder(permutation.value());
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

// Returns the set of parallel types seen on the loop domain of the given tvs.
std::unordered_set<ParallelType> getTvParallelTypes(std::vector<TensorView*> tvs) {
  std::unordered_set<ParallelType> parallel_types;
  for (auto tv : tvs) {
    for (auto id : tv->getLoopDomain()) {
      if (id->isDeviceDim()) {
        parallel_types.insert(id->getParallelType());
      }
    }
  }
  return parallel_types;
}

void propagateDIDTransform(TensorView* ref, std::vector<TensorView*> tvs, int64_t did_pos, bool allow_c2p, bool allow_p2c) {
  TransformPropagator propagator(ref, did_pos);
  PropagateShardingsSelector selector(
      {tvs.begin(), tvs.end()},
      allow_c2p,
      allow_p2c);
  MaxLogicalDomainInfoSpanningTree(ref, &selector)
      .traverse(&propagator);
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
      std::unordered_set<ParallelType> existing_parallel_types = getTvParallelTypes(outputs_without_mesh);
      std::unordered_map<int64_t, int64_t> new2old =
          selectiveReorderDIDToFront(ref_input, existing_parallel_types);

      // This restricts the transform propagation to the DID axis.
      int64_t num_device_dims = new2old.size();

      // Propagate the DID loop split to the outputs without mesh.
      propagateDIDTransform(/*ref=*/ref_input, /*tvs=*/outputs_without_mesh, /*did_pos=*/num_device_dims, /*allow_c2p=*/false, /*allow_p2c=*/true);

      // Apply parallelization on the outputs without mesh.
      shardAllLike(ref_input, outputs_without_mesh, existing_parallel_types);
    }
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
    // modify their sharding. For non-fusion inputs, we try to propagate shardings
    // from the reference output for parallel types that are not already present.
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

    for (auto tv : sharding_candidates) {
      std::unordered_set<ParallelType> existing_parallel_types = getTvParallelTypes({tv});
      std::unordered_map<int64_t, int64_t> new2old = selectiveReorderDIDToFront(ref_output, existing_parallel_types);
      int64_t did_pos = new2old.size();
      // Note: We do not have to manually shard for reshape here.
      // TransformPropagator can handle reshapes when going from consumer to
      // producer.
      propagateDIDTransform(
        /*ref=*/ref_output, 
        /*tvs=*/{tv}, 
        /*did_pos=*/did_pos, 
        /*allow_c2p=*/true, 
        /*allow_p2c=*/false);
      shardAllLike(ref_output, {tv}, existing_parallel_types);
    }
  }

  bool has_mesh = validateMeshes(fusion);
  if (has_mesh) {
    // Reorder the loop domain since the transform propagator may
    // have reordered the iterdomains in loop domain. For example: Consider
    // linear op: in = [b, m, k] weight = [DIDx(d), n/d, k] After
    // transformation, the loop domain of linear output is [DIDx(d), n/d, b,
    // m, r{k}]. Since, we set allocation to be the same as loop, we reorder it
    // as allocation domain in the interim. Ideally, this should follow logical
    // domain and DIDx axis at the front. The allocation domain should follow
    // any stride order specified/inferred.
    reorderLoopAsAllocation(fusion->allTvs());
    for (auto tv : fusion->allTvs()) {
      tv->setAllocationDomain(tv->getLoopDomain(), true);
    }
  }
}

} // namespace nvfuser::preseg_passes

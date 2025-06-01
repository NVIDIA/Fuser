// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/reorder_sharded_axis.h>

#include <fusion.h>
#include <host_ir/lower_to_communication.h>
#include <ir/allocation_utils.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {
bool isLocalSizeOne(IterDomain* id) {
  return id->isParallelized() || id->isBroadcast() || id->isReduction();
}

// Always returns canonicalized.
Layout getRequiredLayout(
    TensorView* tv,
    const CommunicationType type,
    IterDomain* sharded_id) {
  NVF_ERROR(
      std::find(
          tv->getLogicalDomain().begin(),
          tv->getLogicalDomain().end(),
          sharded_id) != tv->getLogicalDomain().end());

  if (type == CommunicationType::Reduce ||
      type == CommunicationType::Allreduce) {
    Layout layout = *canonicalizeLayout(tv);
    layout.makeContiguous();
    return layout;
  }

  // FIXME: helper: is sharded_id in front?
  Layout layout = *canonicalizeLayout(tv);
  for (IterDomain* id : layout.allocation_domain) {
    if (id == sharded_id) {
      layout.makeContiguous();
      return layout;
    }
    if (!isLocalSizeOne(id)) {
      // FIXME: helper
      Layout sharded_in_front;
      sharded_in_front.allocation_domain.reserve(
          layout.allocation_domain.size());
      sharded_in_front.allocation_domain.push_back(sharded_id);
      for (IterDomain* alloc_id : layout.allocation_domain) {
        if (alloc_id != sharded_id) {
          sharded_in_front.allocation_domain.push_back(alloc_id);
        }
      }
      sharded_in_front.contiguity = TensorDomain::getContiguityFilledWith(
          sharded_in_front.allocation_domain, true);
      return sharded_in_front;
    }
  }
  NVF_THROW(
      "Should never reach here - sharded_id must be found in allocation domain");
}

// FIXME: reuese
bool contiguityIsCompliant(
    const std::optional<bool>& actual,
    const std::optional<bool>& required) {
  if (actual == true && required == false) {
    return true;
  }
  return actual == required;
}

// FIXME: reuse
// Returns whether `layout` is compliant with `required`. This is
// uni-directional. For example, `contiguity=[t,t]` is compliant with
// `contiguity=[f,f]` but not vice versa.
bool isCompliantWith(const Layout& layout, const Layout& required) {
  if (layout.allocation_domain != required.allocation_domain) {
    // This can be relaxed by allowing broadcast dimensions to be ordered
    // differently.
    return false;
  }

  for (const auto i : arange(layout.size())) {
    if (!contiguityIsCompliant(layout.contiguity[i], required.contiguity[i])) {
      return false;
    }
  }
  return true;
}

// FIXME: reuse
std::optional<Layout> mapInLayoutToOutRoot(
    const std::optional<Layout>& preferred_in_layout,
    TensorView* in,
    TensorView* out) {
  if (!preferred_in_layout.has_value()) {
    return std::nullopt;
  }

  if (!ir_utils::computePermutation(
           in->getLogicalDomain(), preferred_in_layout->allocation_domain)
           .has_value()) {
    // Give up when `in`'s allocation domain is not an logical permutation. As
    // an extension, we could map in_alloc to in_logical and apply the inverse
    // mapping to out_root.
    return std::nullopt;
  }

  std::unordered_map<IterDomain*, IterDomain*> in_logical_to_out_root =
      PairwiseLogicalDomainMap(in, out).mapProducerToConsumer();

  Layout preferred_out_layout;
  for (auto&& [in_alloc_id, contiguity] :
       zip(preferred_in_layout->allocation_domain,
           preferred_in_layout->contiguity)) {
    IterDomain* out_root_id = getOrDefault(in_logical_to_out_root, in_alloc_id);
    if (out_root_id == nullptr) {
      // This can happen when in_alloc_id is of type reduction or squeezed out.
      continue;
    }
    preferred_out_layout.allocation_domain.push_back(out_root_id);
    preferred_out_layout.contiguity.push_back(contiguity);
  }
  return preferred_out_layout;
}

void makeCommunicationLayoutCompliant(
    Expr* expr,
    CommunicationInfo communication_info) {
  auto* input = expr->inputs().at(0)->as<TensorView>();
  auto* output = expr->outputs().at(0)->as<TensorView>();

  IterDomain* p_sharded_id = communication_info.p_sharded_id;
  IterDomain* c_sharded_id = communication_info.c_sharded_id;

  Layout p_layout =
      getRequiredLayout(input, communication_info.type, p_sharded_id);
  // FIXME: isCompliantWith from alias_analysis.cc
  if (!isCompliantWith(*canonicalizeLayout(input), p_layout)) {
    TensorView* input_copy = set(input);
    TransformReplay::selfReplay(
        input->domain(), input_copy->domain(), /*ignore_reductions=*/true);
    ir_utils::replaceValInExprInputs(expr, input, input_copy);
    p_layout = *mapInLayoutToOutRoot(p_layout, input, input_copy);
    input = input_copy;
  }
  // FIXME: helper?
  input->setAllocationDomain(p_layout.allocation_domain, p_layout.contiguity);

  // FIXME: dedup
  Layout c_layout =
      getRequiredLayout(output, communication_info.type, c_sharded_id);
  if (output->hasAllocation()) {
    if (!isCompliantWith(*canonicalizeLayout(output), c_layout)) {
      TensorView* output_copy = set(output);
      TransformReplay::selfReplay(
          output->domain(), output_copy->domain(), /*ignore_reductions=*/true);
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, output_copy);
    }
  }
  output->setAllocationDomain(c_layout.allocation_domain, c_layout.contiguity);
}

} // namespace

void ReorderShardedAxisPass::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);

  const std::vector<Expr*>& exprs = fusion->exprs();

  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;

    if (!(expr->isA<LoadStoreOp>() &&
          (expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set)) &&
        !expr->isA<ReductionOp>()) {
      continue;
    }

    if (!isResharding(expr)) {
      continue;
    }

    // FIXME: p_sharded == c_sharded == 0. Some Allreduces with mesh{0} fail
    // getCommunicationInfo.
    auto communication_info = getCommunicationInfo(expr);
    if (!communication_info.has_value()) {
      continue;
    }

    makeCommunicationLayoutCompliant(expr, *communication_info);
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << std::endl
            << "Fusion Transforms after " << name() << ":" << std::endl;
    fusion->printTransforms();
  }
}

} // namespace nvfuser::preseg_passes

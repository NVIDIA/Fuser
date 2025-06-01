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

namespace nvfuser::preseg_passes {

namespace {

#if 0
// Returns the position of an IterDomain in given domain.
int64_t posInDomain(
    const std::vector<IterDomain*>& domain,
    const IterDomain* id) {
  auto pos = std::find(domain.begin(), domain.end(), id);
  NVF_ERROR(
      pos != domain.end(),
      "Expected id ",
      id->toString(),
      " in domain ",
      domain);
  return std::distance(domain.begin(), pos);
}
#endif

// Propagates the DID transformations of ref to tv and parallelizes tv like ref.
void transformAndParallelizeLike(
    TensorView* ref,
    TensorView* tv,
    const PropagateDirection direction) {
  auto old2new = reorderDIDToFront(ref);
  propagateDIDTransform(ref, {tv}, /*did_pos=*/old2new.size(), direction);
  shardAllLike(ref, {tv}, {kParallelTypeDIDs.begin(), kParallelTypeDIDs.end()});
}

#if 0
bool isReduceOrAllreduce(CommunicationInfo communication_info) {
  return communication_info.type == CommunicationType::Reduce ||
      communication_info.type == CommunicationType::Allreduce;
}
#endif

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

bool contiguityIsCompliant(
    const std::optional<bool>& actual,
    const std::optional<bool>& required) {
  if (actual == true && required == false) {
    return true;
  }
  return actual == required;
}

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

void makeCommunicationLayoutCompliant(
    Expr* expr,
    CommunicationInfo communication_info) {
  auto* input = expr->inputs().at(0)->as<TensorView>();
  auto* output = expr->outputs().at(0)->as<TensorView>();

  IterDomain* p_sharded_id = communication_info.p_sharded_id;
  IterDomain* c_sharded_id = communication_info.c_sharded_id;

  // FIXME: distinguish p and c?
  Layout p_layout =
      getRequiredLayout(input, communication_info.type, p_sharded_id);
  // FIXME: isCompliantWith from alias_analysis.cc
  if (!isCompliantWith(*canonicalizeLayout(input), p_layout)) {
    TensorView* input_copy = input->cacheAfter();
    transformAndParallelizeLike(
        input, input_copy, PropagateDirection::kForward);
    input = input_copy;
  }
  // FIXME: helper?
  input->setAllocationDomain(p_layout.allocation_domain, p_layout.contiguity);

  // FIXME: dedup
  Layout c_layout =
      getRequiredLayout(output, communication_info.type, c_sharded_id);
  if (output->hasAllocation() || output->isFusionOutput()) {
    if (!isCompliantWith(*canonicalizeLayout(output), c_layout)) {
      TensorView* output_copy = output->cacheBefore();
      transformAndParallelizeLike(
          output, output_copy, PropagateDirection::kBackward);
      output = output_copy;
    }
  }
  output->setAllocationDomain(c_layout.allocation_domain, c_layout.contiguity);

#if 0
  // Copy input if:
  // 1. Input is not contiguous.
  // 2. Input is not allocation compliant and is not a reduce/allreduce.
  if (isTvContiguous(input) &&
      (isReduceOrAllreduce(communication_info) ||
       isAllocationOrderCompliant(input, p_sharded_id))) {
    // If the input is already in the required memory layout,
    // set its allocation explicitly to avoid changes by other passes.
    input->setAllocationDomain(input->getMaybeAllocationDomain(), true);
  } else {
    // Copy input into required memory layout.
    // Note: If input is not a fusion input, we should ideally be able to
    // specify allocation domain of input instead. However, some schedulers
    // (e.g. reduction) may not support this, so creating a copy here.
    // This cannot always be done, for example, when this input is
    // noncontiguous.

    TensorView* input_copy = set(input);

    // Find index of p_sharded_id in input_copy's logical domain.
    // Reduction axis in input can change the position of p_sharded_id in
    // input_copy's logical domain.
    // This reordering is not needed for reduce/allreduce.
    std::unordered_map<int64_t, int64_t> reorder_map;
    if (!isReduceOrAllreduce(communication_info)) {
      auto p2c =
          PairwiseLogicalDomainMap(input, input_copy).mapProducerToConsumer();

      IterDomain* input_copy_sharded_id = p2c.at(p_sharded_id);
      int64_t input_copy_sharded_idx = posInDomain(
          input_copy->getMaybeAllocationDomain(), input_copy_sharded_id);
      reorder_map[input_copy_sharded_idx] = 0;
    }

    input_copy->setAllocationDomain(
        TensorDomain::orderedAs(
            input_copy->getMaybeAllocationDomain(), reorder_map),
        true);

    // New communication expression.
    // We do not create a new output tensorview for the communication.
    // Instead, we replace the input of the communication with the input_copy
    // in the communication expression. This preserves the sharding annotations
    // of the original communication output.
    if (expr->isA<LoadStoreOp>()) {
      IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, output, input_copy);
    } else {
      IrBuilder::create<ReductionOp>(
          expr->as<ReductionOp>()->getReductionOpType(),
          expr->as<ReductionOp>()->init(),
          output,
          input_copy);
    }
    transformAndParallelizeLike(input, input_copy);
  }

  // For reduce/allreduce, c_sharded_id is reduction axis and considered
  // compliant.
  if (isAllocationOrderCompliant(output, c_sharded_id)) {
    // If the output is already in the required memory layout,
    // set its allocation explicitly to avoid changes by other passes.
    output->setAllocationDomain(output->getMaybeAllocationDomain(), true);
  } else {
    std::optional<Layout> output_layout = canonicalizeLayout(output);

    NVF_ERROR(
        output_layout.has_value(),
        "Cannot canonicalize layout for ",
        output->domain()->toString(0, false));
    int64_t output_sharded_idx =
        posInDomain((*output_layout).allocation_domain, c_sharded_id);

    // If the output has allocation domain,
    // create a copy of the output to revert the allocation domain.
    if (output->hasAllocation()) {
      TensorView* output_copy = set(output);

      auto p2c_map =
          PairwiseLogicalDomainMap(output, output_copy).mapProducerToConsumer();
      std::vector<IterDomain*> output_copy_allocation_domain;

      for (IterDomain* output_id :
           TensorDomain::noReductions((*output_layout).allocation_domain)) {
        IterDomain* output_copy_id = p2c_map.at(output_id);
        output_copy_allocation_domain.push_back(output_copy_id);
      }

      output_copy->setAllocationDomain(output_copy_allocation_domain, true);
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, output_copy);
      transformAndParallelizeLike(output, output_copy);
    }

    output->setAllocationDomain(
        TensorDomain::orderedAs(
            (*output_layout).allocation_domain, {{output_sharded_idx, 0}}),
        true);
  }
#endif
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

#if 0
    if (isCommunicationLayoutCompliant(expr)) {
      // Set the allocation domain explicitly to avoid changes by other passes.
      // No reordering / copying is needed.
      auto* input = expr->inputs().at(0)->as<TensorView>();
      auto* output = expr->outputs().at(0)->as<TensorView>();
      input->setAllocationDomain(input->getMaybeAllocationDomain(), true);
      output->setAllocationDomain(output->getMaybeAllocationDomain(), true);
      continue;
    }
#endif

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

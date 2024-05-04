// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <c10/util/irange.h>
#include <compute_at.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/double_buffer.h>
#include <exceptions.h>
#include <fusion.h>
#include <inlining.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <scheduler/mma_utils.h>

// Cleanup
#include <transform_iter.h>
#include <transform_replay.h>

namespace nvfuser {

namespace {
DataType aten_opt_type_map(const c10::optional<at::ScalarType>& scalar_type) {
  return scalar_type.has_value() ? aten_to_data_type(scalar_type.value())
                                 : DataType::Null;
}
} // namespace

TensorView::TensorView(
    IrBuilderPasskey passkey,
    TensorDomain* domain,
    DataType dtype,
    MemoryType mtype)
    : Val(passkey, ValType::TensorView, dtype),
      domain_(domain),
      memory_type_(mtype) {}

NVFUSER_DEFINE_CLONE(TensorView)

std::string TensorView::toString(int indent_size) const {
  std::stringstream ss;
  ss << ir_utils::varName(this);
  switch (getMemoryType()) {
    case MemoryType::Global:
      ss << "_g";
      break;
    case MemoryType::Shared:
      ss << "_s";
      break;
    case MemoryType::Local:
      ss << "_l";
      break;
    default:
      NVF_ERROR(false, "Unknown tensor memory type.");
  }
  ss << domain()->toString(indent_size);

  if (getComputeAtPosition() > 0) {
    ss << " ca_pos( ";
    ss << getComputeAtPosition();
    ss << " )";
  }
  if (hasComputeWith()) {
    ss << " compute_with( ";
    bool first = true;
    if (hasResolvedComputeWith()) {
      for (auto consumer : getComputeWithConsumers()) {
        if (!first) {
          ss << ", ";
        }
        ss << ir_utils::varName(consumer);
        first = false;
      }
      ss << ", ";
    }
    ss << getComputeWithPosition();
    ss << " )";
  }
  if (getMaxProducerPosition() > 0) {
    ss << " produce_pos( ";
    ss << getMaxProducerPosition();
    ss << " )";
  }
  if (getMaybeMaxProducerPosition() > getMaxProducerPosition()) {
    ss << " maybe_produce_pos( ";
    ss << getMaybeMaxProducerPosition();
    ss << " )";
  }
  if (hasDeviceMesh()) {
    ss << " (" << getDeviceMesh() << ")";
  }
  return ss.str();
}

std::string TensorView::toInlineString(int indent_size) const {
  return toString(indent_size);
}

TensorView::TensorView(const TensorView* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      domain_(ir_cloner->clone(src->domain_)),
      compute_at_pos_(src->compute_at_pos_),
      max_producer_pos_(src->max_producer_pos_),
      memory_type_(src->memory_type_),
      is_double_buffered_(src->is_double_buffered_),
      is_circular_buffered_(src->is_circular_buffered_),
      circular_buffer_stage_(src->circular_buffer_stage_),
      cpu_scalar_(src->cpu_scalar_),
      has_swizzle_op_(src->has_swizzle_op_),
      compute_with_consumers_(ir_cloner->clone(src->compute_with_consumers_)),
      compute_with_pos_(src->compute_with_pos_),
      promote_reuse_(src->promote_reuse_),
      mesh_(src->mesh_) {}

void TensorView::printTransforms() const {
  IrTransformPrinter(std::cout).printTransforms(this);
}

// sets cpu_scalar_ value, which is special handling for CPU based zero-dim
// tensors (i.e. CPU Tensors that only have one value). This is only used if
// on an input value, otherwise ignored. This is important as special handling
// because these "scalars" should be type promoted as a tensor, but we want to
// avoid explicit copying of the data, so we want to pass the data value as a
// standard kernel argument value.
void TensorView::setCpuScalar(bool is_cpu_scalar) {
  NVF_ERROR(nDims() == 0, "Only 0-dim tensors can be marked as a cpu scalar.");
  cpu_scalar_ = is_cpu_scalar;
}

IterDomain* TensorView::axis(int64_t pos) const {
  NVF_ERROR(nDims() > 0, "Tried to access an axis in a 0-dim TensorView");
  return domain()->axis(wrapDim(pos));
}

void TensorView::inlineAt(
    int64_t pos,
    bool best_effort,
    MaxPosCalculator* calc) {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");

  std::unique_ptr<MaxPosCalculator> calc_owner;
  if (calc == nullptr) {
    calc_owner = std::make_unique<MaxPosCalculator>();
    calc = calc_owner.get();
  }

  pos = nvfuser::wrapDim(pos, nDims() + 1);

  int64_t max_inline_pos = (int64_t)calc->getMaxPosAll(this, best_effort);

  if (best_effort) {
    pos = std::min<int64_t>(max_inline_pos, pos);
  }

  // hoist inner most broadcast
  while (pos > 0 && axis(pos - 1)->isBroadcast()) {
    pos--;
  }

  NVF_ERROR(
      pos <= max_inline_pos,
      "Invalid inline position for T",
      name(),
      ": ",
      pos,
      ". Maximum allowed value:",
      max_inline_pos);

  if (isFusionInput()) {
    return;
  }

  if (pos <= compute_at_pos_) {
    return;
  }

  compute_at_pos_ = pos;

  // If the new computeAt position is further inlined than the
  // computeWith position, reset the computeWith setting
  if (compute_at_pos_ >= compute_with_pos_) {
    clearComputeWith();
  }

  for (auto consumer : ir_utils::consumerTvsOf(this)) {
    consumer->updateMaxProducerPosition();
  }
}

namespace {

// Try to find the aligned position on consumer's domain corresponding to a
//  position of producer domain. No checking on actual
//  producer-consumer relationship.
int64_t getConsumerPosAlignedToProducerCA(
    TensorView* consumer,
    TensorView* producer,
    int64_t producer_pos) {
  // Locate consumer's position that aligns with
  //  the producer's position. We need broadcast axes forwarded so we
  //  need to replay PasC as CasP will not forward braodcast dims. For example
  //  if we have:
  // T2[ iS22{( 3 * 1 )} ] ca_pos( 1 ) = broadcast( T1[ iS1{3} ] ca_pos( 1 )
  // produce_pos( 1) ) CasP will have the mapping iS1{3} -> iS2{3} and PasC will
  // have the mapping iS22{( 3 * 1 )} <- iS1{3} We need the latter. Refer to
  // NVFuserTest.FusionComplexBCast1_CUDA

  auto disjoint_sets =
      BestEffortReplay::replayPasC(
          producer, consumer, -1, PairwiseRootDomainMap(producer, consumer))
          .getIterDomainEquivalence();

  // Find the innermost position of consumer that has
  //  been mapped within the producer ca axis.
  int64_t consumer_pos = consumer->nDims();
  while (consumer_pos > 0) {
    auto consumer_id = consumer->axis(consumer_pos - 1);
    auto p_dom = producer->getLeafDomain();
    if (std::any_of(
            p_dom.begin(),
            p_dom.begin() + producer_pos,
            [&consumer_id, &disjoint_sets](IterDomain* p_id) {
              return disjoint_sets.permissiveAreMapped(consumer_id, p_id);
            })) {
      break;
    }
    consumer_pos--;
  }

  return consumer_pos;
}

} // namespace

void TensorView::updateMaxProducerPosition() {
  for (auto producer : ir_utils::producerTvsOf(this)) {
    max_producer_pos_ = std::max(
        max_producer_pos_,
        getConsumerPosAlignedToProducerCA(
            this, producer, producer->getComputePosition(this)));
  }

  maybe_max_producer_pos_ = max_producer_pos_;

  // When a producer may be computed with this tensor, i.e., it isn't
  // yet resolved, reflect that in maybe_max_producer_pos_. If all
  // producers are already resolved, i.e., after the initial
  // resolveComputeWith in lowering, this should be just equal to
  // max_producer_pos_.
  for (auto producer : ir_utils::producerTvsOf(this)) {
    if (producer->hasComputeWith() && !producer->hasResolvedComputeWith()) {
      maybe_max_producer_pos_ = std::max(
          maybe_max_producer_pos_,
          getConsumerPosAlignedToProducerCA(
              this, producer, producer->getComputeWithPosition()));
    }
  }
}

TensorView* TensorView::computeAt(
    TensorView* consumer,
    int64_t position,
    ComputeAtMode mode) {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  // Make sure this and consumer are not the same tensor, that's illegal
  NVF_CHECK(!sameAs(consumer), "Cannot call this->computeAt(this, ...)");

  // We support negative axes, so increment it by consumer->nDims() + 1 and make
  // sure the result is within consumer->nDims() + 1. being at consumer->nDims()
  // means producer will be computed inline with consumer, hence the +1.
  if (position < 0) {
    position += int64_t(consumer->nDims()) + 1;
  }

  NVF_CHECK(
      (position >= 0 && position < consumer->nDims() + 1) ||
          mode == ComputeAtMode::BestEffort,
      "Compute at called on an position outside valid range.");

  if (mode == ComputeAtMode::BestEffort) {
    position = std::max(-1L, position);
    position = std::min(consumer->nDims(), position);
  }

  ComputeAt::runAt(this, consumer, (unsigned int)position, mode);

  return this;
}

void TensorView::computeWith(int64_t pos, bool best_effort) {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");

  if (isFusionInput()) {
    return;
  }

  NVF_CHECK(
      !ir_utils::consumerTvsOf(this).empty(),
      "There must be at least one consumer of this tensor to use computeWith: ",
      toString());

  pos = nvfuser::wrapDim(pos, nDims() + 1);

  const int64_t max_inline_pos =
      (int64_t)MaxPosCalculator({}, true).getMaxPosAll(this, best_effort);

  if (best_effort) {
    pos = std::min(max_inline_pos, pos);
  }

  // hoist inner most broadcast
  while (pos > 0 && axis(pos - 1)->isBroadcast()) {
    pos--;
  }

  NVF_CHECK(
      pos <= max_inline_pos,
      "Invalid computeWith position for T",
      name(),
      ": ",
      pos,
      ". Maximum allowed value:",
      max_inline_pos);

  // The position must be right of the computeAt position
  NVF_CHECK(
      pos >= getComputeAtPosition(),
      "Position must be right of the computeAt position. Position: ",
      pos,
      ", computeAt position: ",
      getComputeAtPosition());

  // If it's already set to be computed with the consumer and the
  // position is higher, nothing to change
  if (getComputeWithPosition() >= pos) {
    return;
  }

  // Update the siblings together
  auto siblings = ir_utils::filterByType<TensorView>(definition()->outputs());

  for (auto sibling : siblings) {
    sibling->clearComputeWith();
  }

  // If the given position is the same as the computeAt position, this
  // is a no-op
  if (pos == getComputeAtPosition()) {
    return;
  }

  for (auto sibling : siblings) {
    sibling->compute_with_pos_ = (unsigned int)pos;
  }

  for (auto consumer : ir_utils::consumerTvsOf(this)) {
    consumer->updateMaxProducerPosition();
  }
}

bool TensorView::isComputedWith(const TensorView* consumer) const {
  if (!hasComputeWith()) {
    return false;
  }

  // Quering is an error if the compute-with consumer is still unresolved
  NVF_ERROR(hasResolvedComputeWith(), "Not resolved yet: ", toString());

  return std::find(
             getComputeWithConsumers().begin(),
             getComputeWithConsumers().end(),
             consumer) != getComputeWithConsumers().end();
}

const std::vector<TensorView*>& TensorView::getComputeWithConsumers() const {
  NVF_ERROR(
      !hasComputeWith() || hasResolvedComputeWith(),
      "computeWith not yet resolved: ",
      toString());
  return compute_with_consumers_;
}

int64_t TensorView::getComputePosition(const TensorView* consumer) const {
  if (hasResolvedComputeWith() && isComputedWith(consumer)) {
    return getComputeWithPosition();
  } else {
    return getComputeAtPosition();
  }
}

bool TensorView::resolveComputeWith(const std::vector<Expr*>& sorted_exprs) {
  NVF_ERROR(container()->isA<kir::Kernel>(), "Function invalid for fusion.");

  auto siblings = ir_utils::filterByType<TensorView>(definition()->outputs());

  for (auto sibling : siblings) {
    NVF_ERROR(
        sibling->hasComputeWith(),
        "Invlaid attempt to resolve computeWith: ",
        sibling->toString());
  }

  // It may have been already resolved through its siblings
  if (hasResolvedComputeWith()) {
    return false;
  }

  std::unordered_set<Expr*> use_set;
  for (auto sibling : siblings) {
    use_set.insert(sibling->uses().begin(), sibling->uses().end());
  }

  for (auto expr : sorted_exprs) {
    if (!use_set.count(expr)) {
      continue;
    }

    // First use found. Set it as the computeWith target tensor
    std::vector<TensorView*> use_out_tvs{
        ir_utils::filterByType<TensorView>(expr->outputs()).begin(),
        ir_utils::filterByType<TensorView>(expr->outputs()).end()};

    for (auto sibling : siblings) {
      sibling->compute_with_consumers_ = use_out_tvs;
    }

    for (auto consumer_tv : compute_with_consumers_) {
      consumer_tv->updateMaxProducerPosition();
    }

    return true;
  }

  // No expr found
  NVF_ERROR(false, "No use expr found in the sorted expr list: ", toString());
}

void TensorView::clearComputeWith() {
  //! This should be only used while in a Fusion container.
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");

  compute_with_pos_ = getComputeAtPosition();

  // compute_with_consumers_ should still be empty
  NVF_ERROR(compute_with_consumers_.empty());
}

TensorView* TensorView::split(
    int64_t axis,
    Val* factor,
    bool inner_split,
    bool trim_out_of_bounds) {
  // Only check things associated with axis, factor will be validated in
  // IterDomain
  NVF_ERROR(
      nDims() > 0,
      "Tried to do split on a 0-dim TensorView. ",
      "Tensor: ",
      toString());

  axis = wrapDim(axis);

  NVF_CHECK(
      axis >= getMaxComputePosition(),
      "Cannot split axis within compute at position. Axis = ",
      axis,
      " computePosition = ",
      getMaxComputePosition(),
      ". Tensor: ",
      toString());

  NVF_CHECK(
      axis >= getMaybeMaxProducerPosition(),
      "Cannot split axis within max producer position. Axis = ",
      axis,
      " maxProducerPosition = ",
      getMaybeMaxProducerPosition(),
      ". Tensor: ",
      toString());

  NVF_CHECK(
      this->axis(axis)->getParallelType() == ParallelType::Serial,
      "Splitting an axis of non-Serial parallel type is not supported at this time."
      " Parallelization strategy must be set after calling split.",
      ". Tensor: ",
      toString());

  if (factor->dtype() != DataType::Index) {
    factor = castOp(DataType::Index, factor);
  }

  domain()->split(axis, factor, inner_split, trim_out_of_bounds);
  return this;
}

TensorView* TensorView::split(
    int64_t axis,
    int64_t factor,
    bool inner_split,
    bool trim_out_of_bounds) {
  // NOTE: safe cast to int64_t, factor (unsigned int) is within int64_t range
  NVF_CHECK(
      factor > 0,
      "Invalid factor for split. Factor must be greater than 0. Factor = ",
      factor);
  split(
      axis,
      IrBuilder::create<Val>(factor, DataType::Index),
      inner_split,
      trim_out_of_bounds);
  return this;
}

// Merge "axis_o" and "axis_i" into 1 dimension
TensorView* TensorView::merge(int64_t axis_o, int64_t axis_i) {
  NVF_ERROR(nDims() > 0, "Tried to do merge on a 0-dim TensorView");
  axis_o = wrapDim(axis_o);
  axis_i = wrapDim(axis_i);

  NVF_CHECK(
      axis_o >= getMaxComputePosition() && axis_i >= getMaxComputePosition(),
      false,
      "Cannot merge axes within compute at position. Either axis ",
      axis_o,
      " or ",
      axis_i,
      " are within computePosition = ",
      getMaxComputePosition());

  NVF_CHECK(
      axis_o >= getMaybeMaxProducerPosition() &&
          axis_i >= getMaybeMaxProducerPosition(),
      "Cannot merge axes within max producer position. Either axis ",
      axis_o,
      " or ",
      axis_i,
      " are within maxProducerPosition = ",
      getMaybeMaxProducerPosition());

  NVF_CHECK(
      axis(axis_o)->getParallelType() == ParallelType::Serial ||
          axis(axis_i)->getParallelType() == ParallelType::Serial,
      "Merging axes of non-Serial parallel type is not supported at this time."
      " Parallelization strategy must be set after calling split.");

  domain()->merge(axis_o, axis_i);
  return this;
}

TensorView* TensorView::reorder(
    const std::unordered_map<int64_t, int64_t>& old2new_) {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  NVF_ERROR(
      !(nDims() == 0 && !old2new_.empty()),
      "Tried to reorder a 0-dim TensorView");

  for (auto entry : old2new_) {
    auto old_pos = entry.first < 0 ? entry.first + nDims() : entry.first;
    auto new_pos = entry.second < 0 ? entry.second + nDims() : entry.second;
    if (old_pos == new_pos) {
      continue;
    }
    NVF_ERROR(
        old_pos >= 0,
        "Found \"old\" position that's less than 0 even though already adjusted by nDims: ",
        old_pos);
    NVF_ERROR(
        new_pos >= 0,
        "Found \"new\" position that's less than 0 even though already adjusted by nDims: ",
        new_pos);
    NVF_CHECK(
        old_pos >= getMaxComputePosition() &&
            new_pos >= getMaxComputePosition(),
        "Cannot reorder axes within compute at position. Either axis ",
        old_pos,
        " or ",
        new_pos,
        " are within computePosition = ",
        getMaxComputePosition());

    NVF_CHECK(
        old_pos >= getMaybeMaxProducerPosition() &&
            new_pos >= getMaybeMaxProducerPosition(),
        "Cannot reorder axes within max producer position. Either axis ",
        old_pos,
        " or ",
        new_pos,
        " are within maxProducerPosition = ",
        getMaybeMaxProducerPosition());
  }

  domain()->reorder(old2new_);
  return this;
}

TensorView* TensorView::swizzle(
    SwizzleType swizzle_type,
    int64_t x,
    int64_t y) {
  x = wrapDim(x);
  y = wrapDim(y);

  // Check swizzle specific constraints on the input axes:
  auto x_id = axis(x);
  auto y_id = axis(y);

  NVF_ERROR(
      x_id->extent()->isConstInt() && y_id->extent()->isConstInt(),
      "Only constant iterdomains supported on given swizzle type");

  int64_t in_x_size = x_id->extent()->evaluate().as<int64_t>();
  int64_t in_y_size = y_id->extent()->evaluate().as<int64_t>();

  // Check size constraints based on swizzle type
  if (swizzle_type == SwizzleType::XOR) {
    NVF_ERROR(in_x_size == in_y_size, "Swizzle: equal dim iterdomains only");
  }

  if (swizzle_type == SwizzleType::XOR) {
    // XOR swizzle only support power of 2 swizzle unit sizes:
    bool is_pow_of_2 = in_x_size > 1 && ((in_x_size & (in_x_size - 1)) == 0);
    NVF_ERROR(is_pow_of_2, "XOR swizzle only support power of 2 domain sizes.");
  }

  domain()->swizzle(swizzle_type, x, y);

  return this;
}

TensorView* TensorView::swizzle(
    Swizzle2DType swizzle_type,
    int64_t x,
    int64_t y,
    SwizzleMode swizzle_mode) {
  has_swizzle_op_ = true;
  x = wrapDim(x);
  y = wrapDim(y);

  NVF_CHECK(
      !(getMemoryType() == MemoryType::Global &&
        swizzle_mode == SwizzleMode::Data),
      "Data swizzle on global memory is not supported.");

  NVF_CHECK(
      x >= getMaxComputePosition(),
      "Cannot swizzle axes within compute at position. Axis ",
      x,
      " is within computePosition = ",
      getMaxComputePosition());

  NVF_CHECK(
      y >= getMaybeMaxProducerPosition(),
      "Cannot swizzle axes within max producer position. Axis ",
      y,
      " is within maxProducerPosition = ",
      getMaybeMaxProducerPosition());

  // Disable unsupported use cases at the current step.
  //  Currently do not support reducing or broadcasting
  //   swizzled dimensions.
  auto all_inputs = InputsOf::outputs({axis(x), axis(y)});
  for (auto id : ir_utils::filterByType<IterDomain>(all_inputs)) {
    NVF_ERROR(
        !id->isBroadcast() && !id->isReduction(),
        "Unsupported use case for swizzle.");
  }

  // Also checking that the scheduler is not trying to
  //  compose swizzles, which is not yet supported either.
  auto all_exprs = DependencyCheck::getAllValsBetween(
      {all_inputs.begin(), all_inputs.end()}, {axis(x), axis(y)});
  for (auto expr : all_exprs) {
    NVF_ERROR(
        !expr->isA<Swizzle2D>(), "Composing swizzles is not yet supported");
  }

  // Check swizzle specific constraints on the input axes:
  if (swizzle_type != Swizzle2DType::ZShape) {
    auto x_id = axis(x);
    auto y_id = axis(y);

    NVF_ERROR(
        x_id->extent()->isConstInt() && y_id->extent()->isConstInt(),
        "Only constant iterdomains supported on given swizzle type");

    int64_t in_x_size = x_id->extent()->evaluate().as<int64_t>();
    int64_t in_y_size = y_id->extent()->evaluate().as<int64_t>();

    // Check size constraints based on swizzle type
    if (swizzle_type == Swizzle2DType::XOR ||
        swizzle_type == Swizzle2DType::CyclicShift) {
      NVF_ERROR(in_x_size == in_y_size, "Swizzle: equal dim iterdomains only");
    }

    if (swizzle_type == Swizzle2DType::XOR) {
      // XOR swizzle only support power of 2 swizzle unit sizes:
      bool is_pow_of_2 = in_x_size > 1 && ((in_x_size & (in_x_size - 1)) == 0);
      NVF_ERROR(
          is_pow_of_2, "XOR swizzle only support power of 2 domain sizes.");
    }
  }

  domain()->swizzle(swizzle_type, x, y, swizzle_mode);

  return this;
}

TensorView* TensorView::rFactor(const std::vector<int64_t>& axes) {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  // TODO: I think we should do this but
  // NVFuserTest.FusionSmemBlockGemmCache_CUDA prevents it from going in at the
  // moment.

  // NVF_ERROR(
  //     !hasComputeAt(), "Cannot rfactor tensors after compute at has been
  //     set.");
  NVF_ERROR(nDims() > 0, "Tried to rFactor a 0-dim TensorView");
  FusionGuard fg(fusion());
  NVF_CHECK(
      definition() != nullptr &&
          (definition()->isStrictlyOneOf<ReductionOp, MmaOp>()),
      "Error rfactoring ",
      this,
      " its definition is either a nullptr or not a reduction.");
  NVF_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  NVF_CHECK(
      !definition()->isA<GroupedReductionOp>(),
      "For GroupedReductionOp, use TensorView::rFactor(const std::vector<int64_t>& axes, const std::vector<TensorView*>& tvs)");

  // Split tensor view into 2 parts
  auto domain_pair = domain()->rFactor(axes);

  // Producer in the pair
  auto producer_domain = domain_pair.first;
  // Consumer in the pair
  auto consumer_domain = domain_pair.second;

  // This domain will be the consumer, so create the producer
  TensorView* producer =
      IrBuilder::create<TensorView>(producer_domain, getDataType().value());

  producer->setDeviceMesh(mesh_);

  // Set domain of consumer
  setDomain(consumer_domain);
  TensorView* consumer = this;

  if (auto this_reduction = dynamic_cast<ReductionOp*>(definition())) {
    // Setup dependency chain, inserting producer before this op.
    // Expr* producer_definition =
    IrBuilder::create<ReductionOp>(
        this_reduction->getReductionOpType(),
        this_reduction->init(),
        producer,
        this_reduction->in());

    // Expr* consumer_definition =
    IrBuilder::create<ReductionOp>(
        this_reduction->getReductionOpType(),
        this_reduction->init(),
        consumer,
        producer);
  } else if (auto this_mma = dynamic_cast<MmaOp*>(definition())) {
    // Initial reduction that still uses mma to combine
    //  the input.
    IrBuilder::create<MmaOp>(
        producer,
        this_mma->inA(),
        this_mma->inB(),
        this_mma->init(),
        this_mma->macro(),
        this_mma->layout());

    // Remaining reduction that can be scheduled cross
    //  warp or cta.
    IrBuilder::create<ReductionOp>(
        BinaryOpType::Add, this_mma->init(), consumer, producer);
  } else {
    NVF_ERROR(false, "RFactor: unsupported tensor definition");
  }
  return producer;
}

TensorView* TensorView::multiOutputRFactorHelper(
    TensorView* tv,
    const std::vector<int64_t>& axes) {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  // Hack:
  // Semantically we should always keep the outputs of multi reduction ops
  // scheduled the same but the user end cannot guarantee that. In order to
  // guarantee that the rFactor is defined meaningfully the scheduling of the
  // output TV that got the rfactor call is force replayed towards the other two

  if (this != tv) {
    auto root = tv->getRootDomain();
    auto this_root = getRootDomain();

    // construct a trivial root domain map
    std::unordered_map<IterDomain*, IterDomain*> id_map;
    for (const auto i : c10::irange(root.size())) {
      id_map[this_root[i]] = root[i];
    }

    // replay on the target tv
    ReplayTransformations replay(getLeafDomain(), id_map);

    // construct the new tensor domain
    std::vector<IterDomain*> new_id;
    for (auto id : getLeafDomain()) {
      NVF_ERROR(
          replay.getReplay().count(id), "Multi-output reduction replay failed");
      new_id.push_back(replay.getReplay().at(id));
    }

    std::vector<std::optional<bool>> new_contig(tv->domain()->contiguity());
    // replace tensor domain of target tv
    tv->setDomain(IrBuilder::create<TensorDomain>(
        tv->getRootDomain(), new_id, new_contig));
  }

  // Split tensor view into 2 parts
  auto domain_pair = tv->domain()->rFactor(axes);
  // Producer in the pair
  auto producer_domain = domain_pair.first;
  // Consumer in the pair
  auto consumer_domain = domain_pair.second;

  // This domain will be the consumer, so create the producer
  TensorView* producer =
      IrBuilder::create<TensorView>(producer_domain, tv->getDataType().value());

  // Set domain of consumer
  tv->setDomain(consumer_domain);

  return producer;
}

std::vector<TensorView*> TensorView::rFactor(
    const std::vector<int64_t>& axes,
    const std::vector<TensorView*>& tvs) {
  NVF_CHECK(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  NVF_CHECK(nDims() > 0, "Tried to rFactor a 0-dim TensorView");
  FusionGuard fg(fusion());
  NVF_CHECK(
      definition() != nullptr && ir_utils::isReductionOp(definition()),
      "Error rfactoring multi-output reduction op ",
      this,
      " its definition is either a nullptr or not a GroupedReductionOp or a multi-output reduction op.");

  NVF_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  NVF_CHECK(
      definition()->outputs().size() == tvs.size(),
      "Rfactor of a multi-output reduction not used correctly");

  for (const auto i : c10::irange(tvs.size())) {
    NVF_CHECK(
        definition()->output(i) == tvs.at(i),
        "Rfactor of a multi-output reduction not used correctly");
  }

  // Currently grouping of welford is only supported through
  // ParallelType::Group, so GroupedWelfordOp is only created during
  // the lowering time. As rFactor is done before lowering, there
  // should be no GroupedWelfordOp at this point.
  NVF_ERROR(
      !definition()->isA<GroupedWelfordOp>(),
      "GroupedWelfordOp found: ",
      definition()->toString());

  std::vector<TensorView*> rf_tvs(tvs.size());

  // Make sure this gets rfactored last so everybody gets
  //  replayed correctly
  for (const auto i : c10::irange(tvs.size())) {
    if (this != tvs.at(i)) {
      rf_tvs.at(i) = multiOutputRFactorHelper(tvs.at(i), axes);
    }
  }

  for (const auto i : c10::irange(tvs.size())) {
    if (this == tvs.at(i)) {
      rf_tvs.at(i) = multiOutputRFactorHelper(tvs.at(i), axes);
    }
  }

  if (auto wop = dynamic_cast<WelfordOp*>(definition())) {
    TensorView* producer_avg = rf_tvs.at(0);
    TensorView* producer_var = rf_tvs.at(1);
    TensorView* producer_n = rf_tvs.at(2);

    // Setup dependency chain, inserting producer before this op.
    // Expr* producer_definition =
    IrBuilder::create<WelfordOp>(
        producer_avg,
        producer_var,
        producer_n,
        wop->inAvg(),
        wop->inVar(),
        wop->inN(),
        wop->initAvg(),
        wop->initVar(),
        wop->initN());

    // Expr* consumer_definition =
    IrBuilder::create<WelfordOp>(
        wop->outAvg(),
        wop->outVar(),
        wop->outN(),
        producer_avg,
        producer_var,
        producer_n,
        wop->initAvg(),
        wop->initVar(),
        wop->initN());
  } else if (
      auto grouped_rop = dynamic_cast<GroupedReductionOp*>(definition())) {
    IrBuilder::create<GroupedReductionOp>(
        grouped_rop->getReductionOpTypes(),
        grouped_rop->initVals(),
        std::vector<Val*>{rf_tvs.begin(), rf_tvs.end()},
        grouped_rop->inputs());

    IrBuilder::create<GroupedReductionOp>(
        grouped_rop->getReductionOpTypes(),
        grouped_rop->initVals(),
        grouped_rop->outputs(),
        std::vector<Val*>{rf_tvs.begin(), rf_tvs.end()});
  } else {
    NVF_ERROR(false, "Invalid definition: ", definition()->toString());
  }

  return rf_tvs;
}

TensorView* TensorView::cacheBefore(LoadStoreOpType op_type) {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  NVF_CHECK(
      definition() != nullptr && !isFusionInput(),
      "Error adding cacheBefore ",
      this,
      " its definition is a nullptr and we restrict using cacheBefore on an input.");

  // Previously, caching computed-at tensors was allowed but was never
  // really robust. Make it an error unless it is really needed.
  NVF_CHECK(
      !hasComputeAt(),
      "Caching computed-at tensors is not allowed. Apply caching before computeAt");

  // It also did additional transformation when a producer tensor has computeAt.
  // Make sure we no longer rely on that behavior.
  for (TensorView* producer_of_producer :
       ir_utils::filterByType<TensorView>(definition()->inputs())) {
    NVF_CHECK(
        !producer_of_producer->hasComputeAt(),
        "Potentially invalid computeAt and caching detected. Apply caching before computeAt.");
  }

  // Create Producer Domain
  // This domain will be the consumer which needs a new domain, so replace the
  // producers domain with this domain.

  TensorView* producer = IrBuilder::create<TensorView>(
      container(),
      IrBuilder::create<TensorDomain>(
          container(),
          getRootDomain(),
          getRFactorDomain(),
          getAllocationDomain(),
          getLeafDomain(),
          getContiguity()),
      getDataType().value());

  // Set domain of consumer
  TensorView* consumer = this;

  size_t i = 0;
  auto no_reduction_root_domain =
      TensorDomain::noReductions(getMaybeRFactorDomain());
  std::vector<IterDomain*> new_root_domain(no_reduction_root_domain.size());
  for (const auto& dom : no_reduction_root_domain) {
    new_root_domain[i++] = dom->cloneWithoutRFactor();
  }

  // Warning: allocation domain is temporarily discarded. It will be recovered
  // later.
  consumer->setDomain(IrBuilder::create<TensorDomain>(
      container(),
      new_root_domain,
      TensorDomain::getContiguityFilledWith(new_root_domain, true)));

  // Insert producer - Cache_Before (CB) - before this TV.
  // Before: Prev TV -> [Definition Op] -> This TV
  // After:  Prev TV -> [Definition Op] -> New CB TV -> [Set Op] -> This TV

  std::vector<Val*> replaced_siblings;
  replaced_siblings.reserve(definition()->outputs().size());
  for (auto outp : definition()->outputs()) {
    replaced_siblings.push_back(outp == this ? producer : outp);
  }
  ir_utils::transferDefinitionToNewOutputs(definition(), replaced_siblings);

  IrBuilder::create<LoadStoreOp>(container(), op_type, consumer, producer);

  // definition_ is no longer valid
  // setDefinition(nullptr);

  auto replayed_consumer_pair = TransformReplay::replayCasP(
      consumer, producer, -1, TransformReplayOptions().replayAllocation());

  consumer->setDomain(replayed_consumer_pair.first);

  return producer;
}

TensorView* TensorView::cacheFork() {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  // Before: [Expr] -> This TV (Global Output) -> [Usage Expr]
  // After:  [Expr] -> This TV (Local) -> [Usage Expr] > Next TV
  //                            (Fork) -> [Set Expr]   -> New TV (Global Output)

  NVF_CHECK(
      this->isFusionOutput() && !this->uses().empty(),
      "Error adding cacheFork ",
      this,
      " this TensorView must be an output with subsequent uses");

  // Previously, caching computed-at tensors was allowed but was never
  // really robust. Make it an error unless it is really needed.
  NVF_CHECK(
      !hasComputeAt(),
      "Caching computed-at tensors is not allowed. Apply caching before computeAt");

  // This domain will be the producer, so create the consumer
  auto root_domain = TensorDomain::noReductions(getMaybeRFactorDomain());

  TensorView* new_output = IrBuilder::create<TensorView>(
      container(),
      IrBuilder::create<TensorDomain>(
          container(),
          IterDomain::clone(root_domain),
          TensorDomain::getContiguityFilledWith(root_domain, true)),
      getDataType().value());

  // Create write operation from this TV to new output
  IrBuilder::create<LoadStoreOp>(
      container(), LoadStoreOpType::Set, new_output, this);

  // The new TV becomes an output.
  // New TV has global memory type.
  // This TV has local memory type.
  fusion()->replaceOutput(this, new_output);

  // Transform new output according to this TV
  auto replayed_output_pair = TransformReplay::replayCasP(
      new_output, this, -1, TransformReplayOptions().replayAllocation());
  new_output->setDomain(replayed_output_pair.first);

  return new_output;
}

TensorView* TensorView::cacheAfter(LoadStoreOpType op_type, CacheOp cache_op) {
  NVF_ERROR(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  // Get all the uses for this Tensorview
  NVF_CHECK(
      !uses().empty(),
      "Error adding cacheAfter ",
      this,
      " we restrict using cacheAfter on tensors that have no further uses.");

  // Previously, caching computed-at tensors was allowed but was never
  // really robust. Make it an error unless it is really needed.
  NVF_CHECK(
      !hasComputeAt(),
      "Caching computed-at tensors is not allowed. Apply caching before computeAt.");

  bool is_allowed_op =
      !ir_utils::isTvUsedByOpsOfType<SliceOp, SelectOp, PadOp>(this) &&
      !ir_utils::isIndexSelectLookupTv(this);
  NVF_CHECK(
      is_allowed_op,
      "Right now, caching tensors that are input to the select/slice/pad ops are not allowed as they must be in global memory.")

  // It also did additional transformation when this tensor is an
  // input and the outputs of its consumers have computeAt. Make sure
  // we no longer rely on that behavior.
  if (isFusionInput()) {
    for (const auto& expr : uses()) {
      for (TensorView* output :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        NVF_CHECK(
            !output->hasComputeAt(),
            "Potentially invalid computeAt and caching detected. Apply caching before computeAt.");
      }
    }
  }

  // Create Consumer Domain
  // Keep Broadcast Axis (Permanent)
  // Remove Reduction Axis
  size_t i = 0;
  auto no_reduction_root_domain =
      TensorDomain::noReductions(getMaybeRFactorDomain());
  std::vector<IterDomain*> new_root_domain(no_reduction_root_domain.size());
  for (const auto& dom : no_reduction_root_domain) {
    new_root_domain[i++] = dom->cloneWithoutRFactor();
  }

  // This domain will be the producer, so create the consumer
  TensorView* consumer = IrBuilder::create<TensorView>(
      container(),
      IrBuilder::create<TensorDomain>(
          container(),
          new_root_domain,
          TensorDomain::getContiguityFilledWith(new_root_domain, true)),
      getDataType().value());

  // Set domain of producer - No Change
  TensorView* producer = this;

  // Insert consumer - Cache_After (CA) - after this TV.
  // Before: This TV -> [Use Op] -> Next TV
  // After:  This TV -> [Set Op] -> New CA TV -> [Use Op] -> Next TV

  // Expr* consumer_uses =
  for (auto expr : fusion()->unordered_uses(this)) {
    ir_utils::replaceValInExprInputs(expr, this, consumer);
  }

  // Expr* consumer_definition =
  IrBuilder::create<LoadStoreOp>(
      container(), op_type, consumer, producer, cache_op);

  auto replayed_consumer_pair = TransformReplay::replayCasP(
      consumer, producer, -1, TransformReplayOptions().replayAllocation());

  consumer->setDomain(replayed_consumer_pair.first);

  return consumer;
}

void TensorView::setMemoryType(MemoryType mt) {
  memory_type_ = mt;
  if (isFusionInput() || isFusionOutput()) {
    NVF_ERROR(
        mt == MemoryType::Global,
        "Tried to set an input or output to the fusion to a non-global memory type.");
  }
}

void TensorView::clearReductionIterDomains() {
  NVF_ERROR(
      !domain()->hasRFactor(),
      "should not call clearReductionIterDomains on rfactor tv");

  NVF_ERROR(
      getLeafDomain() == getRootDomain(),
      "should not call clearReductionIterDomains on already transformed TensorDomains");

  if (domain()->hasAllocation()) {
    NVF_ERROR(
        std::is_permutation(getRootDomain().begin(), getRootDomain().end(),
                            getAllocationDomain().begin(), getAllocationDomain().end()),
        "should not call clearReductionIterDomains on transformed allocation domain");
    std::vector<IterDomain*> new_root;
    std::vector<IterDomain*> new_alloc;
    std::vector<std::optional<bool>> new_contig;
    for (const auto i : c10::irange(getRootDomain().size())) {
      auto root_i = getRootDomain().at(i);
      if (!root_i->isReduction()) {
        new_root.push_back(root_i);
      }
      auto alloc_i = getAllocatoinDomain().at(i);
      if (!alloc_i->isReduction()) {
        new_alloc.push_back(alloc_i);
        new_contig.push_back(domain()->contiguity().at(i));
      }
    }

    setDomain(IrBuilder::create<TensorDomain>(container(), new_root, std::vector<IterDomain*>(), new_alloc, new_root, new_contig));
  } else {
    std::vector<IterDomain*> new_root;
    std::vector<std::optional<bool>> new_contig;
    for (const auto i : c10::irange(getRootDomain().size())) {
      auto root_i = getRootDomain().at(i);
      if (!root_i->isReduction()) {
        new_root.push_back(root_i);
        new_contig.push_back(domain()->contiguity().at(i));
      }
    }

    setDomain(IrBuilder::create<TensorDomain>(container(), new_root, new_contig));
  }
}

void TensorView::doubleBuffer() {
  // Early correctness checking. May miss eventual errors as the
  // checks depend on memory types and parallelization, which may not
  // be finalized until lowering.
  validateDoubleBufferedTensor(this);
  is_double_buffered_ = true;
}

void TensorView::circularBuffer(int64_t stage) {
  // Early correctness checking. May miss eventual errors as the
  // checks depend on memory types and parallelization, which may not
  // be finalized until lowering.
  NVF_ERROR(stage > 1, "Unsupported stage number");
  if (stage == 2) {
    // Re-direct to double buffer interface if stage is 2;
    doubleBuffer();
    return;
  }
  validateDoubleBufferedTensor(this);
  is_circular_buffered_ = true;
  circular_buffer_stage_ = stage;
}

bool TensorView::isEmptyTensor() const {
  auto& root_domain = getMaybeRFactorDomain();
  return std::all_of(
      root_domain.begin(), root_domain.end(), [](IterDomain* id) {
        return id->extent()->isZeroInt();
      });
}

void TensorView::applyMmaSwizzle(MmaOperand operand) {
  switch (operand) {
    case MmaOperand::Accumulator:
      mma_utils::WarpMmaSwizzler::scheduleMmaWarpOutput(this);
      if (definition()->isA<MmaOp>()) {
        setAllocationDomain(getLeafDomain(), true);
      }
      break;
    case MmaOperand::A:
    case MmaOperand::B:
      mma_utils::WarpMmaSwizzler::scheduleOperandRead(this, operand);
      if (ir_utils::isLdMatrixOp(definition())) {
        setAllocationDomain(getLeafDomain(), true);
        mma_utils::WarpMmaSwizzler::scheduleLdMatrix(this, operand);
      }
      break;
    default:
      NVF_ERROR(false, "unknown operand flag");
      break;
  }
}

void TensorView::applyMmaSwizzle(
    MmaInputSmemSwizzle swizzle,
    bool transpose,
    bool transpose2) {
  NVF_ERROR(
      getMemoryType() == MemoryType::Shared,
      "Shared memory swizzle is only supported for shared memory");
  mma_utils::WarpMmaSwizzler::scheduleOperandRead(
      this, swizzle, transpose, transpose2);
}

void TensorView::commitLeafToRFactor() {
  NVF_CHECK(
      ir_utils::consumerTvsOf(this).empty(),
      "Changing the rFactor domain of an intermediate tensor is not supported yet");
  setDomain(IrBuilder::create<TensorDomain>(
      container(),
      domain_->root(),
      domain_->leaf(),
      domain_->allocation(),
      domain_->leaf(),
      // TODO: If needed, we can let commitLeafToRFactor to take a parameter to
      // allow customizing contiguity. But there is no such need now, so I will
      // just fill the contiguity with true.
      TensorDomain::getContiguityFilledWith(
          (domain_->hasAllocation() ? domain_->allocation() : domain_->leaf()),
          true)));
}

TensorViewBuilder& TensorViewBuilder::ndims(int64_t ndims) {
  NVF_CHECK(ndims >= 0);
  NVF_CHECK(shape_.empty() || (int64_t)shape_.size() == ndims);
  NVF_CHECK(contiguity_.empty() || (int64_t)contiguity_.size() == ndims);
  ndims_ = ndims;
  return *this;
}

TensorViewBuilder& TensorViewBuilder::dtype(DataType dtype) {
  dtype_ = dtype;
  return *this;
}

TensorViewBuilder& TensorViewBuilder::contiguity(
    std::vector<std::optional<bool>> contiguity) {
  NVF_CHECK(
      contiguity_.empty() && !uniform_contiguity_.has_value(),
      "Attempting to reset contiguity");
  contiguity_ = std::move(contiguity);
  return *this;
}

TensorViewBuilder& TensorViewBuilder::contiguity(bool contiguity) {
  NVF_CHECK(
      contiguity_.empty() && !uniform_contiguity_.has_value(),
      "Attempting to reset contiguity");
  uniform_contiguity_ = contiguity;
  return *this;
}

TensorViewBuilder& TensorViewBuilder::shape(const std::vector<int64_t>& shape) {
  NVF_CHECK(shape_.empty(), "Attempting to reset shape");
  if (!shape.empty()) {
    NVF_CHECK(ndims_ == 0 || ndims_ == (int64_t)shape.size());
    ndims_ = (int64_t)shape.size();
  }
  shape_.clear();
  shape_.reserve(shape.size());
  for (int64_t i : shape) {
    if (i == -1) {
      shape_.emplace_back(IrBuilder::create<Val>(DataType::Index));
    } else if (i == 1) {
      shape_.emplace_back(FusionGuard::getCurFusion()->oneVal());
    } else if (i == 0) {
      shape_.emplace_back(FusionGuard::getCurFusion()->zeroVal());
    } else {
      NVF_CHECK(
          i >= 0,
          "Invalid extent value. ",
          "For a tensor representing a single scalar use ndims = 0 with no sizes set.");
      shape_.emplace_back(IrBuilder::create<Val>(i, DataType::Index));
    }
  }
  return *this;
}

TensorViewBuilder& TensorViewBuilder::shape(std::vector<Val*> shape) {
  NVF_CHECK(shape_.empty(), "Attempting to reset shape");
  if (!shape.empty()) {
    NVF_CHECK(ndims_ == 0 || ndims_ == (int64_t)shape.size());
    ndims_ = (int64_t)shape.size();
  }
  shape_ = std::move(shape);
  return *this;
}

TensorViewBuilder& TensorViewBuilder::strideOrder(
    std::vector<int64_t> stride_order) {
  NVF_CHECK(stride_order_.empty(), "Attempting to reset stride_order");
  if (!stride_order.empty()) {
    NVF_CHECK(ndims_ == 0 || ndims_ == (int64_t)stride_order.size());
    ndims_ = (int64_t)stride_order.size();
  }

  // TODO: this shouldn't be necessary. For details see issue
  // https://github.com/NVIDIA/Fuser/issues/1399
  //
  // skip stride_order if its alloc_domain is in the same order as with rfactor
  // domain. We don't need this and we should be able to just use stride_order_,
  // but currently alloc_domain support isn't ideal and could prevent
  // vectorization. Adding this workaround to restore performance.
  if (std::adjacent_find(
          stride_order.begin(), stride_order.end(), [](int64_t l, int64_t r) {
            return l <= r;
          }) != stride_order.end()) {
    // stride_order is not in descending order, we cannot skip it.
    stride_order_ = std::move(stride_order);
  }
  return *this;
}

TensorViewBuilder& TensorViewBuilder::expanded(std::vector<bool> expanded) {
  NVF_CHECK(expanded_.empty(), "Attempting to reset expanded shape");
  if (!expanded.empty()) {
    NVF_CHECK(ndims_ == 0 || ndims_ == (int64_t)expanded.size());
    ndims_ = (int64_t)expanded.size();
  }
  expanded_ = std::move(expanded);
  return *this;
}

TensorView* TensorViewBuilder::build() const {
  // Build the domain
  std::vector<IterDomain*> domain(ndims_, nullptr);
  for (const auto i : c10::irange(ndims_)) {
    bool is_expanded = false;
    Val* extent = nullptr;
    Val* expanded_extent = nullptr;

    // shape_extent means "which extent, `extent` or `expanded_extent`, is
    // shape_[i] describing?" If expanded_[i] is false, then we should create a
    // regular ID with extent shape_[i], that is, shape_[i] is describing
    // `extent`. If expanded_[i] is true, then we need to create a broadcasting
    // ID with extent 1 and expanded extent shape_[i], that is, shape_[i] is
    // describing `expanded_extent`.
    Val** shape_extent = &extent;

    if (!expanded_.empty()) {
      is_expanded = expanded_.at(i);
    }
    if (is_expanded) {
      extent = FusionGuard::getCurFusion()->oneVal();
      shape_extent = &expanded_extent;
    }
    if (shape_.empty()) {
      *shape_extent = IrBuilder::create<Val>(DataType::Index);
    } else {
      *shape_extent =
          SimplifyingIrBuilder::maybeCastExpr(DataType::Index, shape_.at(i));
    }
    IterDomainBuilder builder(FusionGuard::getCurFusion()->zeroVal(), extent);
    if (extent->isConstScalar() && extent->evaluate() == 1) {
      builder.iter_type(IterType::Broadcast);
    }
    if (expanded_extent != nullptr) {
      builder.expanded_extent(expanded_extent);
    }
    domain[i] = builder.build();
  }

  NVF_CHECK(
      contiguity_.empty() || contiguity_.size() == domain.size(),
      "The size of contiguity must equal to the number of non-broadcasting IterDomains");

  if (uniform_contiguity_.has_value()) {
    NVF_ERROR(
        contiguity_.empty(),
        "contiguity_ and uniform_contiguity_ can not be set at the same time");
    // Create the final TensorView
    return IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            domain,
            stride_order_,
            TensorDomain::getContiguityFilledWith(
                domain, *uniform_contiguity_)),
        dtype_);
  } else {
    // Create the final TensorView
    return IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(domain, stride_order_, contiguity_),
        dtype_);
  }
}

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <algorithm>
#include <iterator>
#include <list>
#include <numeric>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ir/cloner.h>
#include <ir/internal_base_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <transform_rfactor.h>
#include <transform_view.h>
#include <type.h>

namespace nvfuser {

IterDomainBuilder::IterDomainBuilder(Val* _start, Val* _extent)
    : start_(_start), extent_(_extent) {
  NVF_ERROR(
      start_ != nullptr && extent_ != nullptr,
      "Start and extent are required to build an iter domain.");
}

IterDomainBuilder::IterDomainBuilder(const IterDomain* id)
    : start_(id->start()),
      extent_(id->extent()),
      expanded_extent_(
          id->hasExpandedExtent() ? id->expandedExtent() : nullptr),
      stop_offset_(id->stopOffset()),
      parallel_type_(id->getParallelType()),
      iter_type_(id->getIterType()),
      is_rfactor_domain_(id->isRFactorProduct()),
      is_padded_dimension_(id->hasPaddingToMultipleOfWarp()),
      is_clustered_dimension_(id->isClusteredBlockDim()),
      padded_to_size_(id->getMaybeSizeAfterPadding()) {}

IterDomainBuilder& IterDomainBuilder::resetSchedulingParams() {
  parallel_type_ = ParallelType::Serial;
  is_rfactor_domain_ = false;
  is_padded_dimension_ = false;
  is_clustered_dimension_ = false;
  padded_to_size_ = std::nullopt;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::resetRfactor() {
  return is_rfactor_domain(false);
}

IterDomainBuilder& IterDomainBuilder::start(Val* _start) {
  start_ = _start;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::extent(Val* _extent) {
  extent_ = _extent;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::expanded_extent(Val* _expanded_extent) {
  expanded_extent_ = _expanded_extent;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::stop_offset(Val* _stop_offset) {
  stop_offset_ = _stop_offset;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::parallel_type(
    ParallelType _parallel_type) {
  parallel_type_ = _parallel_type;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::iter_type(IterType _iter_type) {
  iter_type_ = _iter_type;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::is_rfactor_domain(
    bool _is_rfactor_domain) {
  is_rfactor_domain_ = _is_rfactor_domain;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::is_padded_dimension(
    bool _is_padded_dimension) {
  is_padded_dimension_ = _is_padded_dimension;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::padded_to_size(
    std::optional<int64_t> _padded_to_size) {
  padded_to_size_ = _padded_to_size;
  return *this;
}

IterDomain* IterDomainBuilder::build() const {
  NVF_ERROR(
      start_ != nullptr && extent_ != nullptr,
      "Start and extent are required to build an iter domain.");
  return IrBuilder::createInContainer<IterDomain>(start_->container(), *this);
}

IterDomain::IterDomain(
    IrBuilderPasskey passkey,
    Val* start,
    Val* extent,
    Val* expanded_extent,
    Val* stop_offset,
    ParallelType parallel_type,
    IterType iter_type,
    bool is_rfactor_domain,
    bool is_padded_dimension,
    bool is_clustered_blocks,
    std::optional<int64_t> padded_to_size)
    : IterDomain(
          passkey,
          ValType::IterDomain,
          start,
          extent,
          expanded_extent,
          stop_offset,
          parallel_type,
          iter_type,
          is_rfactor_domain,
          is_padded_dimension,
          is_clustered_blocks,
          padded_to_size) {}

IterDomain::IterDomain(
    IrBuilderPasskey passkey,
    ValType vtype,
    Val* start,
    Val* extent,
    Val* expanded_extent,
    Val* stop_offset,
    ParallelType parallel_type,
    IterType iter_type,
    bool is_rfactor_domain,
    bool is_padded_dimension,
    bool is_clustered_blocks,
    std::optional<int64_t> padded_to_size)
    : Val(passkey, vtype),
      start_(start),
      extent_(extent),
      expanded_extent_(expanded_extent),
      stop_offset_(
          stop_offset == nullptr ? passkey.ir_container_->zeroVal()
                                 : stop_offset),
      parallel_type_(parallel_type),
      iter_type_(iter_type),
      is_rfactor_domain_(is_rfactor_domain),
      is_padded_dimension_(is_padded_dimension),
      is_clustered_dimension_(is_clustered_blocks),
      padded_to_size_(padded_to_size) {
  // NOTE: We previously asserted !(isRFactorProduct() && isBroadcast()), i.e.
  // that an IterDomain could not be both a broadcast and an logical domain.
  // However, since the introduction of the resize op, we now have a legitimate
  // case where this may be true; namely, whenever we resize an IterDomain to
  // size 1, we will mark it as Broadcast, but the resize must lie between root
  // and rfactor.

  NVF_ERROR(
      extent->dtype() == DataType::Index,
      "Cannot create an iter domain over an extent that is not an "
      "nvfuser_index_t but received ",
      extent->dtype(),
      " .");

  NVF_ERROR(
      expanded_extent == nullptr || expanded_extent->dtype() == DataType::Index,
      "Cannot create an iter domain over an expanded_extent that is not an "
      "nvfuser_index_t but received ",
      expanded_extent->dtype(),
      " .");

  NVF_ERROR(
      start->dtype() == DataType::Index,
      "Cannot create an iter domain with a start that is not an "
      "nvfuser_index_t but received ",
      start->dtype(),
      " .");

  NVF_ERROR(
      stop_offset_->dtype() == DataType::Index,
      "Cannot create an iter domain with a stop_offset_ that is not an "
      "nvfuser_index_t but received ",
      stop_offset_->dtype(),
      " .");
}

IterDomain::IterDomain(IrBuilderPasskey passkey, const IterDomainBuilder& args)

    : IterDomain(
          passkey,
          args.start_,
          args.extent_,
          args.expanded_extent_,
          args.stop_offset_,
          args.parallel_type_,
          args.iter_type_,
          args.is_rfactor_domain_,
          args.is_padded_dimension_,
          args.is_clustered_dimension_,
          args.padded_to_size_) {}

IterDomain::IterDomain(const IterDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      start_(ir_cloner->clone(src->start_)),
      extent_(ir_cloner->clone(src->extent_)),
      expanded_extent_(
          src->hasExpandedExtent() ? ir_cloner->clone(src->expandedExtent())
                                   : nullptr),
      stop_offset_(ir_cloner->clone(src->stop_offset_)),
      parallel_type_(src->parallel_type_),
      iter_type_(src->iter_type_),
      is_rfactor_domain_(src->is_rfactor_domain_),
      is_padded_dimension_(src->is_padded_dimension_),
      is_clustered_dimension_(src->is_clustered_dimension_),
      padded_to_size_(src->padded_to_size_) {}

NVFUSER_DEFINE_CLONE(IterDomain)

// The ITERDOMAIN_SAME_FN macro is used to define the sameAs and sameDefinition
// functions. Here are the data fields of checked in the macro:
//   * start_
//   * extent_
//   * expanded_extent_
//   * stop_offset_
//   * parallel_type_
//   * iter_type_
//   * is_rfactor_domain_
//   * is_padded_dimension_
//   * padded_to_size_
//
// Do not take is_rfactor_domain_ into account. IterDomains are considered the
// same if they are rfactor or not.
//
// TODO: Consider managing them as attributes

#define ITERDOMAIN_SAME_FN(sameFunctionName, OtherType)                     \
  bool IterDomain::sameFunctionName(const OtherType* other) const {         \
    if (other == this) {                                                    \
      return true;                                                          \
    }                                                                       \
    if (!other->isA<IterDomain>()) {                                        \
      return false;                                                         \
    }                                                                       \
    const auto* other_id = other->as<IterDomain>();                         \
    return start()->sameFunctionName(other_id->start()) &&                  \
        extent()->sameFunctionName(other_id->extent()) &&                   \
        hasExpandedExtent() == other_id->hasExpandedExtent() &&             \
        (!hasExpandedExtent() ||                                            \
         expandedExtent()->sameFunctionName(other_id->expandedExtent())) && \
        stopOffset()->sameFunctionName(other_id->stopOffset()) &&           \
        getParallelType() == other_id->getParallelType() &&                 \
        getIterType() == other_id->getIterType() &&                         \
        hasPaddingToMultipleOfWarp() ==                                     \
        other_id->hasPaddingToMultipleOfWarp() &&                           \
        isClusteredBlockDim() == other_id->isClusteredBlockDim() &&         \
        getMaybeSizeAfterPadding() == other_id->getMaybeSizeAfterPadding(); \
  }

ITERDOMAIN_SAME_FN(sameAs, Statement)
ITERDOMAIN_SAME_FN(sameDefinition, Val)

std::string IterDomain::toString(int indent_size) const {
  std::stringstream ss;
  ss << getIterType();
  ss << getParallelType();
  ss << name();
  ss << "{";
  if (!start()->isZeroInt()) {
    ss << start()->toInlineString() << " : ";
  }
  if (stop() != extent()) {
    ss << stop()->toInlineString() << " : ";
  }
  ss << extent()->toInlineString();
  if (hasExpandedExtent()) {
    ss << " ex " << expandedExtent()->toInlineString();
  }
  ss << "}";
  if (isRFactorProduct()) {
    ss << "rf";
  }
  if (hasPaddingToMultipleOfWarp()) {
    ss << "_p";
  }
  if (isClusteredBlockDim()) {
    ss << "_c";
  }
  return ss.str();
}

std::string IterDomain::toInlineString(int indent_size) const {
  return toString(indent_size);
}

// Returns a new IterDomain matching properties of this except for
// is_rfactor_domain_
IterDomain* IterDomain::cloneWithoutRFactor(bool map_with_original) {
  auto cloned = IterDomainBuilder(this).resetRfactor().build();

  if (map_with_original) {
    fusion()->registerExactMapping(this, cloned);
  }

  return cloned;
}

/*static*/ std::vector<IterDomain*> IterDomain::clone(
    const std::vector<IterDomain*>& domains) {
  std::vector<IterDomain*> cloned_domains;
  std::transform(
      domains.begin(),
      domains.end(),
      std::back_inserter(cloned_domains),
      [](auto id) { return id->cloneWithoutRFactor(); });
  return cloned_domains;
}

// Merging does not propagate the start and stop values of the input
// domains to the merged output domain. The actual range of the
// domains is enforced by predicates. Note that since only root
// domains have valid start and stop, it's not possible to contiguous
// predication.
IterDomain* IterDomain::merge(
    IterDomain* outer,
    IterDomain* inner,
    std::optional<bool> rfactor_domain,
    std::optional<IterType> iter_type) {
  NVF_CHECK(
      outer->isReduction() == inner->isReduction(),
      "Merging IterDomains requires that their iteration types match. ",
      "Outer: ",
      outer->toString(),
      ", Inner: ",
      inner->toString());

  NVF_CHECK(
      !outer->isStride() && !inner->isStride(),
      "No support for merging stride domains");

  // By default, if not specified, don't create rfactor
  // outputs. Reshape transformations should propagate the flag, which
  // should explicitly specify the flag
  if (!rfactor_domain.has_value()) {
    rfactor_domain = false;
  }

  Val* merged_id_size =
      SimplifyingIrBuilder::mulExpr(outer->extent(), inner->extent());

  if (!iter_type.has_value()) {
    iter_type = outer->getIterType();

    if (outer->isBroadcast() && inner->isBroadcast()) {
      iter_type = IterType::Broadcast;
    }

    if ((outer->isBroadcast() || inner->isBroadcast()) &&
        (outer->getIterType() == IterType::Iteration ||
         inner->getIterType() == IterType::Iteration)) {
      iter_type = IterType::Iteration;
    }

    if ((outer->isBroadcast() || inner->isBroadcast()) &&
        (outer->getIterType() == IterType::GatherScatter ||
         inner->getIterType() == IterType::GatherScatter)) {
      iter_type = IterType::GatherScatter;
    }
  }

  Val* expanded_extent = nullptr;
  if (outer->hasExpandedExtent() || inner->hasExpandedExtent()) {
    if (outer->hasExpandedExtent() && inner->hasExpandedExtent()) {
      expanded_extent = mul(outer->expandedExtent(), inner->expandedExtent());
    } else if (outer->hasExpandedExtent() && !inner->hasExpandedExtent()) {
      if (inner->isBroadcast()) {
        expanded_extent = outer->expandedExtent();
      } else {
        expanded_extent = mul(outer->expandedExtent(), inner->extent());
      }
    } else if (!outer->hasExpandedExtent() && inner->hasExpandedExtent()) {
      if (outer->isBroadcast()) {
        expanded_extent = inner->expandedExtent();
      } else {
        expanded_extent = mul(outer->extent(), inner->expandedExtent());
      }
    }
  }

  IterDomain* merged_id =
      IterDomainBuilder(outer->container()->zeroVal(), merged_id_size)
          .parallel_type(outer->getParallelType())
          .expanded_extent(expanded_extent)
          .iter_type(*iter_type)
          .is_rfactor_domain(*rfactor_domain)
          .build();

  IrBuilder::createInContainer<Merge>(
      outer->container(), merged_id, outer, inner);

  return merged_id;
}

std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    Val* factor,
    bool inner_split,
    std::optional<bool> rfactor_domain,
    std::optional<IterType> outer_iter_type,
    std::optional<IterType> inner_iter_type) {
  NVF_CHECK(
      factor->isIntegralScalar(), "Cannot split by non-integer value ", factor);

  // outer loop size
  Val* remainder = SimplifyingIrBuilder::ceilDivExpr(in->extent(), factor);
  Val* expanded_remainder = nullptr;
  if (in->hasExpandedExtent()) {
    expanded_remainder =
        SimplifyingIrBuilder::ceilDivExpr(in->expandedExtent(), factor);
  }

  // By default, if not specified, don't create rfactor
  // outputs. Reshape transformations should propagate the flag, which
  // should explicitly specify the flag
  if (!rfactor_domain.has_value()) {
    rfactor_domain = false;
  }

  // If not specified, inherit these properties from the input iter domain
  if (!outer_iter_type.has_value()) {
    outer_iter_type = in->getIterType();
  }

  if (!inner_iter_type.has_value()) {
    inner_iter_type = in->getIterType();
  }

  // outer loop IterDomain
  IterDomain* ido =
      IterDomainBuilder(
          in->container()->zeroVal(), inner_split ? remainder : factor)
          .expanded_extent(
              in->hasExpandedExtent() && inner_split ? expanded_remainder
                                                     : nullptr)
          .parallel_type(in->getParallelType())
          .iter_type(*outer_iter_type)
          .is_rfactor_domain(*rfactor_domain)
          .build();

  // inner loop IterDomain
  IterDomain* idi =
      IterDomainBuilder(
          in->container()->zeroVal(), inner_split ? factor : remainder)
          .expanded_extent(
              in->hasExpandedExtent() && !inner_split ? expanded_remainder
                                                      : nullptr)
          .parallel_type(in->getParallelType())
          .iter_type(*inner_iter_type)
          .is_rfactor_domain(*rfactor_domain)
          .build();

  IrBuilder::createInContainer<Split>(
      in->container(), ido, idi, in, factor, inner_split);
  return {ido, idi};
}

std::pair<IterDomain*, IterDomain*> IterDomain::stridedSplit(int64_t factor) {
  // Use partial split so that only valid values are retained
  auto split_out = IterDomain::split(
      this,
      IrBuilder::createInContainer<Val>(container(), factor, DataType::Index),
      true,
      true);

  split_out.second->iter_type_ = IterType::Stride;
  split_out.first->is_rfactor_domain_ = true;
  split_out.second->is_rfactor_domain_ = true;
  return split_out;
}

std::pair<IterDomain*, IterDomain*> IterDomain::swizzle(
    SwizzleType swizzle_type,
    IterDomain* in_x,
    IterDomain* in_y) {
  NVF_CHECK(
      !in_x->extent()->isZeroInt() && !in_y->extent()->isZeroInt(),
      "Invalid swizzling of a empty dimension.");

  // TODO: reduction check on swizzle:
  NVF_CHECK(
      !in_x->isReduction() && !in_y->isReduction(),
      "swizzled reduction not yet supported");

  for (auto input : InputsOf::outputs({in_x, in_y})) {
    NVF_CHECK(
        !input->as<IterDomain>()->isBroadcast(),
        "swizzling broadcast axes not yet supported");
  }

  IterDomain* out_x = IterDomainBuilder(in_x).build();

  IterDomain* out_y = IterDomainBuilder(in_y).build();

  IrBuilder::createInContainer<Swizzle>(
      in_x->container(), out_x, out_y, in_x, in_y, swizzle_type);

  return std::make_pair(out_x, out_y);
}

std::pair<IterDomain*, IterDomain*> IterDomain::swizzle(
    Swizzle2DType swizzle_type,
    IterDomain* in_x,
    IterDomain* in_y,
    SwizzleMode swizzle_mode) {
  NVF_CHECK(
      !in_x->extent()->isZeroInt() && !in_y->extent()->isZeroInt(),
      "Invalid swizzling of a empty dimension.");

  // TODO: reduction check on swizzle:
  NVF_CHECK(
      !in_x->isReduction() && !in_y->isReduction(),
      "swizzled reduction not yet supported");

  for (auto input : InputsOf::outputs({in_x, in_y})) {
    NVF_CHECK(
        !input->as<IterDomain>()->isBroadcast(),
        "swizzling broadcast axes not yet supported");
  }

  IterDomain* out_x = IterDomainBuilder(in_x).build();

  IterDomain* out_y = IterDomainBuilder(in_y).build();

  IrBuilder::createInContainer<Swizzle2D>(
      in_x->container(), out_x, out_y, in_x, in_y, swizzle_type, swizzle_mode);

  return std::make_pair(out_x, out_y);
}

IterDomain* IterDomain::resize(
    IterDomain* in,
    Val* left_expansion,
    Val* right_expansion,
    bool mark_as_rfactor,
    std::optional<IterType> iter_type_opt) {
  NVF_CHECK(
      left_expansion->isIntegralScalar(),
      "Expansion factor must be an integer scalar: ",
      left_expansion->toString());
  NVF_CHECK(
      right_expansion->isIntegralScalar(),
      "Expansion factor must be an integer scalar: ",
      right_expansion->toString());

  if (left_expansion->isConstInt() && right_expansion->isConstInt()) {
    auto left = left_expansion->evaluate();
    auto right = right_expansion->evaluate();
    if (left == 0 && right == 0) {
      // This is a trivial resize. Check that we are not changing the IterType,
      // then return the input.
      NVF_CHECK(
          !iter_type_opt.has_value() ||
              iter_type_opt.value() == in->getIterType(),
          "If IterType is specified in pad with zero expansion then it must "
          "match input");
      return in;
    }
  }
  NVF_CHECK(
      in->getIterType() == IterType::Iteration ||
          in->getIterType() == IterType::Broadcast ||
          in->getIterType() == IterType::Symbolic,
      "Not a valid IterType: ",
      in->getIterType());

  NVF_CHECK(
      in->start()->isZeroInt(),
      "Non-zero start not supported: ",
      in->toString());
  NVF_CHECK(
      in->stopOffset()->isZeroInt(),
      "Non-zero stop offset not considered: ",
      in->toString());

  // The overall extent is (in_extent + left_expansion +
  // right_expansion). This can be simplified for a slice op as
  // the right expansion should look like (slice_end_offset -
  // in_extent), or (slice_end_offset + (- in_extent)), so the
  // overall extent is left_expansion + slice_end_offset.

  // Detect common slice patterns and return a simplified Val
  // representing (in_extent + right_expansion) if possible
  auto simplify_input_extent_plus_right_expansion = [](Val* right_expansion,
                                                       Val* in_extent) -> Val* {
    auto bop = dynamic_cast<BinaryOp*>(right_expansion->definition());
    if (bop == nullptr) {
      return nullptr;
    }
    Val* sub_rhs = nullptr;
    if (bop->getBinaryOpType() == BinaryOpType::Sub) {
      sub_rhs = bop->rhs();
    } else if (bop->getBinaryOpType() == BinaryOpType::Add) {
      // Note that SimplifyingIrBuilder may turn (a - b) to (a + (- b))
      if (auto uop = dynamic_cast<UnaryOp*>(bop->rhs()->definition());
          uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Neg) {
        sub_rhs = uop->in();
      }
    }
    if (sub_rhs == in_extent) {
      return bop->lhs();
    } else {
      return nullptr;
    }
  };

  Val* resized_id_size = nullptr;
  if (auto simplified_val = simplify_input_extent_plus_right_expansion(
          right_expansion, in->getMaybeExpandedExtent())) {
    resized_id_size =
        SimplifyingIrBuilder::addExpr(left_expansion, simplified_val);
  } else {
    resized_id_size = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::addExpr(
            in->getMaybeExpandedExtent(), left_expansion),
        right_expansion);
  }

  // If output IterType is provided, use it. Otherwise, if we can prove the
  // resized extent is 1, set to Broadcast, if we can prove it is >1 set to
  // Iteration, and otherwise fall back to Symbolic.
  auto iter_type = IterType::Symbolic;
  if (iter_type_opt.has_value()) {
    iter_type = iter_type_opt.value();
  } else if (left_expansion->isConstInt() && right_expansion->isConstInt()) {
    auto left = left_expansion->evaluate();
    auto right = right_expansion->evaluate();
    if (resized_id_size->isConstInt()) {
      // Means input extent is also known
      auto out_extent = resized_id_size->evaluate();
      iter_type = out_extent == 1 ? IterType::Broadcast : IterType::Iteration;
    } else if (left + right > 1) {
      // Input extent is non-negative, so we know out_extent > 1
      iter_type = IterType::Iteration;
    }
  }

  auto resized_id =
      IterDomainBuilder(
          in->container()->zeroVal(),
          // Set immediate constant size of 1 if resize produces broadcast
          iter_type == IterType::Broadcast ? in->fusion()->oneVal()
                                           : resized_id_size)
          .is_rfactor_domain(mark_as_rfactor)
          .iter_type(iter_type)
          .build();

  IrBuilder::createInContainer<Resize>(
      in->container(), resized_id, in, left_expansion, right_expansion);

  return resized_id;
}

// TODO: We should change parallelize interface to be on tensorview or at least
// vectorize should be done on tensorview. This would let us check that we don't
// vectorize to the left of the computeAt domain, and could allow us to do some
// simple validation of vectorize as it's inputs are right most and contiguous.
void IterDomain::parallelize(ParallelType t) {
  if (parallel_type_ == t) {
    // No op, don't do any more checks, it was already set to this value.
    return;
  }

  if (t == ParallelType::Unroll || isParallelTypeVectorize(t) ||
      t == ParallelType::Group) {
    NVF_CHECK(
        start()->isZeroInt() && extent()->isConstScalar(),
        "Vectorization, unrolling, unswitching and grouping are only supported "
        "with start = 0 and extent as a const int, but got ",
        "a start of ",
        start(),
        " and extent ",
        extent()->toInlineString(),
        " .");
  }

  if (t == ParallelType::Group) {
    NVF_CHECK(
        getIterType() == IterType::Iteration ||
            getIterType() == IterType::GatherScatter,
        "Grouping IterDomain of non Iteration / GatherScatter type is not "
        "allowed. ",
        getIterType());
  }

  parallel_type_ = t;
}

bool IterDomain::maybePartial() const {
  return !start()->isZeroInt() || !stopOffset()->isZeroInt();
}

Val* IterDomain::stopOffset() const {
  return stop_offset_;
}

Val* IterDomain::stop() const {
  if (stopOffset()->isZeroInt()) {
    return extent();
  }

  return sub(extent(), stopOffset());
}

namespace {
void validateContiguity(
    const std::vector<IterDomain*>& allocation_domain,
    const std::vector<std::optional<bool>>& contiguity) {
  NVF_CHECK(
      contiguity.size() == allocation_domain.size(),
      "Invalid contiguity information provided, incorrect size. Received "
      "vector of size ",
      contiguity.size(),
      " but needed one of size ",
      allocation_domain.size());
  for (auto i : arange(contiguity.size())) {
    bool expect_null =
        (allocation_domain.at(i)->isBroadcast() ||
         allocation_domain.at(i)->isReduction());
    NVF_CHECK(
        expect_null != contiguity.at(i).has_value(),
        "The contiguity of a broadcast/reduction dimension must be None. "
        "The contiguity of a non-broadcast/reduction dimension must be "
        "true/false. alloation_domain=[",
        toDelimitedString(allocation_domain),
        "], contiguity=[",
        toDelimitedString(contiguity),
        "]");
  }
}

// Check if loop_domain is a valid domain with no
// redundancy. The logical domain is used as a reference to find if
// there's any ID that's not covered by the new loop domain.
void validateLoopDomain(
    const std::vector<IterDomain*>& logical_domain,
    const std::vector<IterDomain*>& loop_domain,
    const std::vector<IterDomain*>& additional_ids) {
  // Skip if there's any symbolic ID
  if (std::any_of(
          logical_domain.begin(),
          logical_domain.end(),
          [](IterDomain* id) { return id->isSymbolic(); }) ||
      std::any_of(
          loop_domain.begin(),
          loop_domain.end(),
          [](IterDomain* id) { return id->isSymbolic(); }) ||
      std::any_of(
          additional_ids.begin(), additional_ids.end(), [](IterDomain* id) {
            return id->isSymbolic();
          })) {
    return;
  }

  std::vector<IterDomain*> reference;
  reference.reserve(logical_domain.size() + additional_ids.size());
  reference.insert(
      reference.end(), logical_domain.begin(), logical_domain.end());
  // additional_ids are also considered part of the reference domain
  reference.insert(
      reference.end(), additional_ids.begin(), additional_ids.end());

  auto [redundant_ids, _, unreachable_reference_ids] =
      ir_utils::compareDomainWithReference(loop_domain, reference);

  auto empty_or_broadcast = [](const auto& ids) {
    return std::all_of(ids.begin(), ids.end(), [](IterDomain* id) {
      return id->isBroadcast();
    });
  };

  NVF_ERROR(
      empty_or_broadcast(redundant_ids),
      "Trying to set a loop domain with non-broadcast redundant IDs: ",
      toDelimitedString(redundant_ids));

  NVF_ERROR(
      empty_or_broadcast(unreachable_reference_ids),
      "Not all logical IDs are covered by loop domain. Loop: ",
      toDelimitedString(loop_domain),
      ". Unreachable logical IDs: ",
      toDelimitedString(unreachable_reference_ids));
}

} // namespace

RaggedIterDomain::RaggedIterDomain(
    IrBuilderPasskey passkey,
    TensorView* extents,
    IterType iter_type,
    ParallelType parallel_type)
    : IterDomain(
          passkey,
          ValType::RaggedIterDomain,
          /*start=*/passkey.ir_container_->zeroVal(),
          /*extent=*/passkey.ir_container_->oneVal(), // Placeholder
          /*expanded_extent=*/nullptr,
          /*stop_offset=*/nullptr,
          parallel_type,
          iter_type,
          /*is_rfactor_domain=*/false,
          /*is_padded_dimension=*/false,
          /*is_clustered_blocks=*/false,
          /*padded_to_size=*/std::nullopt),
      extents_(extents) {
  // Extents must be non-null
  NVF_ERROR(
      extents_ != nullptr, "RaggedIterDomain requires non-null extents tensor");

  // Extents must have integer dtype
  NVF_ERROR_EQ(
      extents_->dtype(),
      DataType::Index,
      "RaggedIterDomain extents must have index type, got ",
      extents_->dtype());

  // Only IterType::Iteration is supported at this moment
  NVF_ERROR_EQ(
      iter_type,
      IterType::Iteration,
      "Only IterType::Iteration is supported: ",
      iter_type);
}

RaggedIterDomain::RaggedIterDomain(
    const RaggedIterDomain* src,
    IrCloner* ir_cloner)
    : IterDomain(src, ir_cloner), extents_(ir_cloner->clone(src->extents_)) {}

NVFUSER_DEFINE_CLONE(RaggedIterDomain)

bool RaggedIterDomain::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }

  if (!other->isA<RaggedIterDomain>()) {
    return false;
  }

  auto other_ragged = other->as<RaggedIterDomain>();

  // Compare parent IterDomain properties
  if (!IterDomain::sameAs(other)) {
    return false;
  }

  // Compare extents tensor
  return extents_->sameAs(other_ragged->extents_);
}

std::string RaggedIterDomain::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << getIterType();
  ss << getParallelType();
  ss << name();
  ss << "Ragged{";
  ss << "extents=" << extents_->toInlineString();
  ss << "}";
  return ss.str();
}

std::string RaggedIterDomain::toString(int indent_size) const {
  return toInlineString(indent_size);
}

std::pair<IterDomain*, RaggedIterDomain*> RaggedIterDomain::partition(
    IterDomain* in,
    TensorView* extents) {
  NVF_ERROR(in != nullptr, "partition: input IterDomain is null");

  NVF_ERROR(
      !in->isA<RaggedIterDomain>(),
      "partition: input is already RaggedIterDomain, cannot partition again");

  NVF_ERROR_EQ(
      in->getParallelType(),
      ParallelType::Serial,
      "Partitioning of parallelized IterDomain not supported: ",
      in->toString());

  NVF_ERROR_EQ(
      in->getIterType(),
      IterType::Iteration,
      "partition: only IterType::Iteration is supported, got ",
      in->getIterType(),
      " for IterDomain: ",
      in->toString());

  NVF_ERROR(extents != nullptr, "partition: extents tensor is null");

  NVF_ERROR_EQ(
      extents->dtype(),
      DataType::Index,
      "partition: extents must have Index type, got ",
      extents->dtype());

  const auto& extents_domain = extents->getLogicalDomain();
  NVF_ERROR_EQ(
      extents_domain.size(),
      1,
      "partition: extents tensor must be 1D, got ",
      extents_domain.size(),
      "D tensor. Multi-dimensional extents not yet supported.");

  auto container = in->container();

  // Create component IterDomain
  // Component extent = number of components = length of extents tensor
  auto zero = container->zeroVal(DataType::Index);
  auto component_extent = extents_domain.at(0)->extent();
  auto component_id = IterDomainBuilder(zero, component_extent)
                          .parallel_type(ParallelType::Serial)
                          .iter_type(IterType::Iteration)
                          .build();

  auto ragged_id =
      IrBuilder::create<RaggedIterDomain>(extents, in->getIterType());

  IrBuilder::create<Partition>(component_id, ragged_id, in, extents);

  return {component_id, ragged_id};
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> logical_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      logical_domain_(std::move(logical_domain)),
      loop_domain_(logical_domain_),
      initial_loop_domain_(loop_domain_),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> logical_domain,
    std::vector<int64_t> stride_order,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      logical_domain_(std::move(logical_domain)),
      loop_domain_(logical_domain_),
      initial_loop_domain_(loop_domain_),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  // setting the proper allocation domain
  if (!stride_order.empty()) {
    auto rank = logical_domain_.size();
    NVF_ERROR(
        rank == stride_order.size(), "Invalid size of stride_order vector");

    // checking stride_order is indeed a permutation
    std::vector<int64_t> inc_vec(rank);
    std::iota(inc_vec.begin(), inc_vec.end(), 0);
    NVF_ERROR(
        std::is_permutation(
            stride_order.begin(), stride_order.end(), inc_vec.begin()),
        "stride_order is not a valid: " + toDelimitedString(stride_order));

    allocation_domain_.resize(rank, nullptr);
    for (auto i : arange(rank)) {
      allocation_domain_[rank - 1 - stride_order[i]] = logical_domain_[i];
    }
  }
  validateContiguity(maybeAllocation(), contiguity_);
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> logical_domain,
    std::vector<IterDomain*> loop_domain,
    std::vector<std::optional<bool>> contiguity,
    bool skip_loop_validation)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      logical_domain_(std::move(logical_domain)),
      loop_domain_(std::move(loop_domain)),
      initial_loop_domain_(loop_domain_),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  if (!skip_loop_validation) {
    NVF_CHECK(
        loop_domain_.empty() == logical_domain_.empty(),
        "logical domain and loop domain can only be both empty or neither "
        "empty");
    validateLoopDomain(logical_domain_, loop_domain_, additional_ids_);
  }
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> logical_domain,
    std::vector<IterDomain*> loop_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      logical_domain_(std::move(logical_domain)),
      loop_domain_(std::move(loop_domain)),
      initial_loop_domain_(loop_domain_),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  NVF_CHECK(
      loop_domain_.empty() == logical_domain_.empty(),
      "logical domain and loop domain can only be both empty or neither empty");
  validateLoopDomain(logical_domain_, loop_domain_, additional_ids_);
  if (!root_domain_.empty()) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, root_domain_, additional_ids_);
  }
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> logical_domain,
    std::vector<IterDomain*> allocation_domain,
    std::vector<IterDomain*> loop_domain,
    std::vector<std::optional<bool>> contiguity,
    std::vector<IterDomain*> additional_ids)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      logical_domain_(std::move(logical_domain)),
      allocation_domain_(std::move(allocation_domain)),
      loop_domain_(std::move(loop_domain)),
      initial_loop_domain_(loop_domain_),
      additional_ids_(std::move(additional_ids)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  NVF_CHECK(
      loop_domain_.empty() == logical_domain_.empty(),
      "logical domain and loop domain can only be both empty or neither empty");
  validateLoopDomain(logical_domain_, loop_domain_, additional_ids_);
  if (!root_domain_.empty()) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, root_domain_, additional_ids_);
  }
  if (!allocation_domain_.empty()) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, allocation_domain_, additional_ids_);
  }
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> logical_domain,
    std::vector<IterDomain*> allocation_domain,
    std::vector<IterDomain*> loop_domain,
    std::optional<std::vector<IterDomain*>> alternate_loop_domain,
    std::vector<std::optional<bool>> contiguity,
    std::vector<IterDomain*> additional_ids,
    bool skip_validation)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      logical_domain_(std::move(logical_domain)),
      allocation_domain_(std::move(allocation_domain)),
      loop_domain_(std::move(loop_domain)),
      alternate_loop_domain_(alternate_loop_domain),
      initial_loop_domain_(loop_domain_),
      additional_ids_(std::move(additional_ids)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  NVF_CHECK(
      loop_domain_.empty() == logical_domain_.empty(),
      "logical domain and loop domain can only be both empty or neither empty");

  if (!skip_validation) {
    validateLoopDomain(logical_domain_, loop_domain_, additional_ids_);
    if (!root_domain_.empty()) {
      ir_utils::validateDomainEquivalence(
          logical_domain_, root_domain_, additional_ids_);
    }
    if (!allocation_domain_.empty()) {
      ir_utils::validateDomainEquivalence(
          logical_domain_, allocation_domain_, additional_ids_);
    }
    if (alternate_loop_domain_.has_value()) {
      validateLoopDomain(
          logical_domain_, alternate_loop_domain_.value(), additional_ids_);
    }
  }
}

TensorDomain::TensorDomain(IrBuilderPasskey passkey, const TensorDomain* src)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(src->root_domain_),
      logical_domain_(src->logical_domain_),
      allocation_domain_(src->allocation_domain_),
      loop_domain_(src->loop_domain_),
      alternate_loop_domain_(src->alternate_loop_domain_),
      initial_loop_domain_(src->initial_loop_domain_),
      additional_ids_(src->additional_ids_),
      contiguity_(src->contiguity_) {}

TensorDomain::TensorDomain(const TensorDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      root_domain_(ir_cloner->clone(src->root_domain_)),
      logical_domain_(ir_cloner->clone(src->logical_domain_)),
      allocation_domain_(ir_cloner->clone(src->allocation_domain_)),
      loop_domain_(ir_cloner->clone(src->loop_domain_)),
      alternate_loop_domain_(ir_cloner->clone(src->alternate_loop_domain_)),
      initial_loop_domain_(ir_cloner->clone(src->initial_loop_domain_)),
      additional_ids_(ir_cloner->clone(src->additional_ids_)),
      contiguity_(src->contiguity()) {}

NVFUSER_DEFINE_CLONE(TensorDomain)

bool TensorDomain::hasBlockBroadcast() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isBroadcast() && id->isThreadDim();
      });
}

bool TensorDomain::hasReduction() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isReduction();
      });
}

bool TensorDomain::hasBroadcast() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isBroadcast();
      });
}

bool TensorDomain::hasGridBroadcast() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isBroadcast() && id->isBlockDim();
      });
}

bool TensorDomain::sameDefinition(const Val* other) const {
  // Val::sameDefinition checks nullptr, dtype, vtype, and definition.
  if (!Val::sameDefinition(other)) {
    return false;
  }
  const TensorDomain* other_td = other->as<TensorDomain>();

  // Check root domains. They are created by ReshapeOp and rFactor to track
  // transformations to original domain.
  if (root_domain_.size() != other_td->root_domain_.size()) {
    return false;
  }
  for (auto&& [id, other_id] : zip(root_domain_, other_td->root_domain_)) {
    if (!id->sameDefinition(other_id)) {
      return false;
    }
  }

  // This check is based on the legacy TensorRecord operator== check.
  // Check number of dimensions
  if (logical_domain_.size() != other_td->logical_domain_.size()) {
    return false;
  }
  for (auto&& [id, other_id] :
       zip(logical_domain_, other_td->logical_domain_)) {
    if (!id->sameDefinition(other_id)) {
      return false;
    }
  }

  // Check stride order
  if (allocation_domain_.size() != other_td->allocation_domain_.size()) {
    return false;
  }
  for (auto&& [id, other_id] :
       zip(allocation_domain_, other_td->allocation_domain_)) {
    if (!id->sameDefinition(other_id)) {
      return false;
    }
  }

  // Check contiguity
  if (contiguity_.size() != other_td->contiguity_.size()) {
    return false;
  }
  return std::ranges::equal(contiguity_, other_td->contiguity_);
}

bool TensorDomain::operator==(const TensorDomain& other) const {
  // Checks equality of each class field. Derived domains such as reduction or
  // broadcast views are computed on demand from these fields.
  return root_domain_ == other.root_domain_ &&
      loop_domain_ == other.loop_domain_ &&
      alternate_loop_domain_ == other.alternate_loop_domain_ &&
      logical_domain_ == other.logical_domain_ &&
      allocation_domain_ == other.allocation_domain_ &&
      contiguity_ == other.contiguity_;
}

bool TensorDomain::sameAs(const Statement* const other) const {
  if (this == other) {
    return true;
  }

  if (!other->isA<TensorDomain>()) {
    return false;
  }

  const auto* other_td = other->as<TensorDomain>();

  if (nDims() != other_td->nDims()) {
    return false;
  }
  if (root().size() != other_td->root().size()) {
    return false;
  }
  if (logical().size() != other_td->logical().size()) {
    return false;
  }
  if (allocation().size() != other_td->allocation().size()) {
    return false;
  }

  for (const auto i : arange(nDims())) {
    if (!(axis(i)->sameAs(other_td->axis(i)))) {
      return false;
    }
  }

  for (const auto i : arange(root().size())) {
    if (!(root()[i]->sameAs(other_td->root()[i]))) {
      return false;
    }
  }

  for (const auto i : arange(logical().size())) {
    if (!(logical()[i]->sameAs(other_td->logical()[i]))) {
      return false;
    }
  }

  for (const auto i : arange(allocation().size())) {
    if (!(allocation()[i]->sameAs(other_td->allocation()[i]))) {
      return false;
    }
  }

  for (const auto i : arange(loop().size())) {
    if (!(loop()[i]->sameAs(other_td->loop()[i]))) {
      return false;
    }
  }

  // this_td has_value is not the same as other_td
  if (alternateLoop().has_value() != other_td->alternateLoop().has_value()) {
    return false;
  }

  // has_value is false for both this_td and other_td
  if (!alternateLoop().has_value() && !other_td->alternateLoop().has_value()) {
    return true;
  }

  // has_value is true for both this_td and other_td, so verify that all
  // iterDomains are the same.
  return std::ranges::all_of(
      std::ranges::iota_view{0LL, (int64_t)alternateLoop().value().size()},
      [&](int64_t i) {
        return alternateLoop().value()[i]->sameAs(
            other_td->alternateLoop().value()[i]);
      });
}

bool TensorDomain::sameAs(
    const std::vector<IterDomain*>& lhs,
    const std::vector<IterDomain*>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  size_t i = 0;
  for (auto td_lhs : lhs) {
    if (!td_lhs->sameAs(rhs[i++])) {
      return false;
    }
  }
  return true;
}

std::string TensorDomain::toString(const int indent_size, const bool loop_only)
    const {
  std::stringstream ss;
  if (loop_only) {
    indent(ss, indent_size) << "[" << toDelimitedString(loop()) << "]";
  } else {
    indent(ss, indent_size)
        << "logical=[" << toDelimitedString(logical()) << "]" << std::endl;
    if (hasRoot()) {
      indent(ss, indent_size + 1)
          << "root=[" << toDelimitedString(root()) << "]" << std::endl;
    }
    indent(ss, indent_size + 1)
        << "loop=[" << toDelimitedString(loop()) << "]" << std::endl;
    if (hasAllocation()) {
      indent(ss, indent_size + 1)
          << "allocation=[" << toDelimitedString(allocation()) << "]"
          << std::endl;
    }
    if (alternateLoop().has_value()) {
      indent(ss, indent_size + 1)
          << "alternate_loop=[" << toDelimitedString(alternateLoop().value())
          << "]" << std::endl;
    }
  }
  return ss.str();
}

std::string TensorDomain::toString(const int indent_size) const {
  return toString(indent_size, /*loop_only=*/true);
}

std::string TensorDomain::toInlineString(int indent_size) const {
  return toString(indent_size);
}

void TensorDomain::setContiguity(
    const std::vector<std::optional<bool>>& contig) {
  validateContiguity(maybeAllocation(), contig);
  contiguity_ = contig;
}

std::vector<int64_t> TensorDomain::strideOrder() const {
  // short-circuit: no allocation domain; default stride-order
  if (allocation_domain_.empty()) {
    return {};
  }

  // The allocation domain is set by the loop domain not by permuting the
  // logical domain.
  if (loop_domain_ == allocation_domain_) {
    return {};
  }

  NVF_ERROR(logical_domain_.size() == allocation_domain_.size());

  // Operations like preprocessGroupedMatmulInputSf pad the logical domain to
  // create the allocation domain. strideOrder only checks for permutations
  // between logical and allocation domains.
  bool is_complex_allocation = std::all_of(
      allocation_domain_.begin(), allocation_domain_.end(), [](IterDomain* id) {
        return id->extent()->definition() != nullptr;
      });
  NVF_CHECK(
      !is_complex_allocation,
      "Encountered non-trivial allocation domain not expressible with stride "
      "order.");

  std::vector<int64_t> stride_order;
  stride_order.reserve(logical_domain_.size());

  for (size_t logical_idx : arange(logical_domain_.size())) {
    IterDomain* logical_id = logical_domain_.at(logical_idx);
    auto alloc_iter = std::find(
        allocation_domain_.begin(), allocation_domain_.end(), logical_id);
    NVF_ERROR(
        alloc_iter != allocation_domain_.end(),
        "Unable to find logical IterDomain in allocation domain.");
    int64_t alloc_idx = std::distance(allocation_domain_.begin(), alloc_iter);
    stride_order.push_back((int64_t)logical_domain_.size() - 1 - alloc_idx);
  }

  return stride_order;
}

bool TensorDomain::hasBlockReduction() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isReduction() && id->isThreadDim();
      });
}

bool TensorDomain::hasGridReduction() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isReduction() && id->isBlockDim() &&
            !id->isClusteredBlockDim();
      });
}

bool TensorDomain::hasClusterReduction() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isReduction() && id->isBlockDim() &&
            id->isClusteredBlockDim();
      });
}

bool TensorDomain::hasSymbolicAxis() const {
  // If there's any Symbolic axis, there must be one at the root or
  // logical domain.
  return (hasRoot() &&
          std::any_of(
              root().begin(),
              root().end(),
              [](auto id) {
                return id->getIterType() == IterType::Symbolic;
              })) ||
      std::any_of(logical().begin(), logical().end(), [](auto id) {
           return id->getIterType() == IterType::Symbolic;
         });
}

bool TensorDomain::hasViewLikeRFactor() const {
  if (!hasRoot()) {
    // Can't have view like rfactor if there is no logical domain
    return false;
  }

  // If there's an logical domain and no rfactor product is a reduction, this is
  // a view like rfactor
  return std::none_of(logical().begin(), logical().end(), [](IterDomain* id) {
    return (id->isReduction() || id->isStride()) && id->isRFactorProduct();
  });
}

bool TensorDomain::hasVectorize() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return isParallelTypeVectorize(id->getParallelType());
      });
}

std::optional<int64_t> TensorDomain::getReductionAxis() const {
  auto it = std::find_if(
      loop_domain_.begin(), loop_domain_.end(), [](const auto& id) {
        return id->isReduction();
      });
  if (it == loop_domain_.end()) {
    return std::optional<int64_t>();
  } else {
    return std::optional<int64_t>(std::distance(loop_domain_.begin(), it));
  }
}

// i here is int, as we want to accept negative value and ::size_type can be a
// uint.
IterDomain* TensorDomain::axis(int64_t i) const {
  NVF_ERROR(nDims() > 0, "Tried to access an axis in a 0-dim domain");
  return loop_domain_[wrapDim(i)];
}

int64_t TensorDomain::posOf(IterDomain* id) const {
  NVF_ERROR(nDims() > 0, "Tried to find an axis in a 0-dim domain");
  int64_t i = 0;
  while (i < (int64_t)loop_domain_.size()) {
    if (loop_domain_[i] == id) {
      return i;
    }
    i++;
  }
  NVF_CHECK(false, "Provided id is not part of this domain.");
}

int64_t TensorDomain::rootPosOf(IterDomain* id) const {
  NVF_ERROR(
      !maybeRoot().empty(), "Tried to find an axis in a 0-dim root domain");
  auto it = std::find(maybeRoot().begin(), maybeRoot().end(), id);
  NVF_ERROR(it != maybeRoot().end(), "Provided id is not part of root domain.");
  return std::distance(maybeRoot().begin(), it);
}

void TensorDomain::broadcast(int64_t axis, Val* extent) {
  axis = nvfuser::wrapDim(axis, nDims() + 1);
  IterDomain* id = IterDomainBuilder(fusion()->zeroVal(), extent)
                       .iter_type(IterType::Broadcast)
                       .build();
  loop_domain_.insert(loop_domain_.begin() + axis, id);
  additional_ids_.push_back(id);
}

void TensorDomain::split(int64_t axis, Val* factor, bool inner_split) {
  NVF_ERROR(nDims() > 0, "Tried to do split on a 0-dim domain");
  axis = wrapDim(axis);

  IterDomain* id = this->axis(axis);

  auto split_ids = IterDomain::split(id, factor, inner_split);
  loop_domain_.erase(loop_domain_.begin() + axis);
  loop_domain_.insert(loop_domain_.begin() + axis, split_ids.second);
  loop_domain_.insert(loop_domain_.begin() + axis, split_ids.first);
}

// Merge "axis_o" and "axis_i" into 1 dimension
void TensorDomain::merge(int64_t axis_o, int64_t axis_i) {
  NVF_ERROR(nDims() > 0, "Tried to do merge on a 0-dim domain");
  axis_o = wrapDim(axis_o);
  axis_i = wrapDim(axis_i);

  NVF_CHECK(
      axis_o != axis_i,
      "Invalid merge detected, axes provided are the same axis.");

  IterDomain* first = axis(axis_o);
  IterDomain* second = axis(axis_i);

  IterDomain* merged_id = IterDomain::merge(first, second);

  // axis_o is the outer input of this merge but does not
  // automatically mean it's an outer domain in TensorDomain.
  auto td_outer_pos = axis_o < axis_i ? axis_o : axis_i;
  auto td_inner_pos = axis_o < axis_i ? axis_i : axis_o;

  loop_domain_.erase(loop_domain_.begin() + td_inner_pos);
  loop_domain_.erase(loop_domain_.begin() + td_outer_pos);
  loop_domain_.insert(loop_domain_.begin() + td_outer_pos, merged_id);
}

// Partition "axis" into component and ragged dimensions. Follow the
// pattern of TensorDomain::split.
void TensorDomain::partition(int64_t axis, TensorView* extents) {
  NVF_ERROR(nDims() > 0, "Tried to do partition on a 0-dim domain");
  axis = wrapDim(axis);

  IterDomain* id = this->axis(axis);

  auto [component_id, ragged_id] = RaggedIterDomain::partition(id, extents);

  // Remove the original axis and insert component and ragged dimensions
  loop_domain_.erase(loop_domain_.begin() + axis);
  loop_domain_.insert(loop_domain_.begin() + axis, ragged_id);
  loop_domain_.insert(loop_domain_.begin() + axis, component_id);
}

// Reorder axes according to map[old_pos] = new_pos
void TensorDomain::reorder(
    const std::unordered_map<int64_t, int64_t>& old2new_) {
  NVF_ERROR(
      nDims() != 0 || old2new_.empty(), "Tried to reorder a 0-dim domain");
  loop_domain_ = orderedAs(loop_domain_, old2new_);
}

std::vector<IterDomain*> TensorDomain::orderedAs(
    const std::vector<IterDomain*>& dom,
    const std::unordered_map<int64_t, int64_t>& old2new_) {
  NVF_ERROR(
      !dom.empty() || old2new_.empty(), "Tried to reorder a 0-dim domain");

  // Eventhough these checks are already in TensorView, we want to redo them as
  // we can enter this function from other places, not through TensorView

  auto new2old = ir_utils::normalizeOld2New(old2new_, (int64_t)dom.size());

  std::vector<IterDomain*> reordered_domain;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::back_inserter(reordered_domain),
      [dom](int64_t i) -> IterDomain* { return dom[i]; });

  return reordered_domain;
}

void TensorDomain::swizzle(SwizzleType swizzle_type, int64_t x, int64_t y) {
  NVF_ERROR(nDims() > 0, "Tried to do merge on a 0-dim domain");
  x = wrapDim(x);
  y = wrapDim(y);

  IterDomain* axis_x = axis(x);
  IterDomain* axis_y = axis(y);

  IterDomain* axis_out_x = nullptr;
  IterDomain* axis_out_y = nullptr;

  std::tie(axis_out_x, axis_out_y) =
      IterDomain::swizzle(swizzle_type, axis_x, axis_y);

  loop_domain_.erase(loop_domain_.begin() + x);
  loop_domain_.insert(loop_domain_.begin() + x, axis_out_x);

  loop_domain_.erase(loop_domain_.begin() + y);
  loop_domain_.insert(loop_domain_.begin() + y, axis_out_y);
}

void TensorDomain::swizzle(
    Swizzle2DType swizzle_type,
    int64_t x,
    int64_t y,
    SwizzleMode swizzle_mode) {
  NVF_ERROR(nDims() > 0, "Tried to do merge on a 0-dim domain");
  x = wrapDim(x);
  y = wrapDim(y);

  IterDomain* axis_x = axis(x);
  IterDomain* axis_y = axis(y);

  IterDomain* axis_out_x = nullptr;
  IterDomain* axis_out_y = nullptr;

  std::tie(axis_out_x, axis_out_y) =
      IterDomain::swizzle(swizzle_type, axis_x, axis_y, swizzle_mode);

  loop_domain_.erase(loop_domain_.begin() + x);
  loop_domain_.insert(loop_domain_.begin() + x, axis_out_x);

  loop_domain_.erase(loop_domain_.begin() + y);
  loop_domain_.insert(loop_domain_.begin() + y, axis_out_y);
}

void TensorDomain::resize(
    int64_t axis,
    Val* left_expansion,
    Val* right_expansion,
    std::optional<IterType> iter_type) {
  NVF_ERROR(nDims() > 0, "Tried to do resize on a 0-dim domain");
  axis = wrapDim(axis);

  IterDomain* id = this->axis(axis);

  auto resized_id = IterDomain::resize(
      id,
      left_expansion,
      right_expansion,
      /*mark_as_rfactor=*/false,
      iter_type);
  loop_domain_.at(axis) = resized_id;
}

std::vector<IterDomain*> TensorDomain::noReductions(
    const std::vector<IterDomain*>& td) {
  std::vector<IterDomain*> noReductionDomain;
  std::copy_if(
      td.begin(),
      td.end(),
      std::back_inserter(noReductionDomain),
      [](IterDomain* id) { return !id->isReduction() && !id->isStride(); });
  return noReductionDomain;
}

std::vector<IterDomain*> TensorDomain::noBroadcasts(
    const std::vector<IterDomain*>& td) {
  std::vector<IterDomain*> noBroadcastDomain;
  std::copy_if(
      td.begin(),
      td.end(),
      std::back_inserter(noBroadcastDomain),
      [](IterDomain* id) { return !id->isBroadcast(); });
  return noBroadcastDomain;
}

std::vector<IterDomain*> TensorDomain::noDevices(
    const std::vector<IterDomain*>& td) {
  std::vector<IterDomain*> noDeviceDomain;
  std::copy_if(
      td.begin(),
      td.end(),
      std::back_inserter(noDeviceDomain),
      [](IterDomain* id) { return !id->isDeviceDim(); });
  return noDeviceDomain;
}

/*static*/ std::vector<std::optional<bool>> TensorDomain::
    getContiguityFilledWith(
        const std::vector<IterDomain*>& allocation_domain,
        bool fill_value) {
  std::vector<std::optional<bool>> contiguity;
  contiguity.reserve(allocation_domain.size());
  for (auto id : allocation_domain) {
    if (id->isBroadcast() || id->isReduction()) {
      contiguity.push_back(std::nullopt);
    } else {
      contiguity.emplace_back(fill_value);
    }
  }
  return contiguity;
}

bool TensorDomain::hasBroadcast(const std::vector<IterDomain*>& td) {
  for (auto id : td) {
    if (id->isBroadcast()) {
      return true;
    }
  }
  return false;
}

bool TensorDomain::hasReduction(const std::vector<IterDomain*>& td) {
  for (auto id : td) {
    if (id->isReduction()) {
      return true;
    }
  }
  return false;
}

TensorDomain* TensorDomain::view(const AnalyzeViewResult& view_analysis) {
  NVF_ERROR(nDims() > 0, "Tried to view transform a 0-dim domain");
  return transformView(this, view_analysis);
}

TensorDomain* TensorDomain::flatten(int64_t start_dim, int64_t end_dim) {
  auto inp_domain = noReductions(logical());

  if (start_dim < 0) {
    start_dim += (int64_t)inp_domain.size();
  }
  if (end_dim < 0) {
    end_dim += (int64_t)inp_domain.size();
  }
  NVF_CHECK(
      start_dim >= 0 && start_dim < int64_t(inp_domain.size()),
      "Invalid start_dim ",
      start_dim);
  NVF_CHECK(
      end_dim >= 0 && end_dim < int64_t(inp_domain.size()),
      "Invalid end_dim ",
      end_dim);
  NVF_CHECK(start_dim <= end_dim, "start_dim must be <= end_dim");

  std::vector<IterDomain*> new_root_domain;
  new_root_domain.reserve(inp_domain.size());
  for (auto i : arange((int64_t)inp_domain.size())) {
    bool is_rfactor_dim = i >= start_dim && i <= end_dim;
    auto inp_id = inp_domain[i];
    auto out_id = IterDomainBuilder(inp_id)
                      .is_rfactor_domain(is_rfactor_dim)
                      .extent(
                          (is_rfactor_dim && inp_id->hasExpandedExtent())
                              ? inp_id->expandedExtent()
                              : inp_id->extent())
                      .iter_type(
                          (is_rfactor_dim && inp_id->isBroadcast())
                              ? IterType::Iteration
                              : inp_id->getIterType())
                      .expanded_extent(nullptr)
                      .build();
    new_root_domain.push_back(out_id);
  }

  std::vector<IterDomain*> logical_domain;
  logical_domain.reserve(new_root_domain.size() - (end_dim - start_dim));
  for (auto i : arange(start_dim)) {
    logical_domain.push_back(new_root_domain[i]);
  }

  IterDomain* merged_id = new_root_domain[start_dim];
  for (auto i : arange(start_dim + 1, end_dim + 1)) {
    merged_id = IterDomain::merge(
        merged_id, new_root_domain.at(i), /*rfactor_domain=*/true);
  }
  logical_domain.push_back(merged_id);

  for (auto i : arange(end_dim + 1, inp_domain.size())) {
    logical_domain.push_back(new_root_domain[i]);
  }

  return IrBuilder::create<TensorDomain>(
      new_root_domain,
      logical_domain,
      logical_domain,
      TensorDomain::getContiguityFilledWith(logical_domain, true));
}

// TODO: Rfactor a Welford

// pair is in order where second is the consumer of first
std::pair<TensorDomain*, TensorDomain*> TensorDomain::rFactor(
    const std::vector<int64_t>& axes_) {
  return TransformRFactor::runReplay(this, axes_);
}

void TensorDomain::setLoopDomain(std::vector<IterDomain*> new_loop_domain) {
  validateLoopDomain(logical(), new_loop_domain, additionalIDs());
  loop_domain_ = std::move(new_loop_domain);
  initial_loop_domain_ = loop_domain_;
}

void TensorDomain::setAlternateLoopDomain(
    std::vector<IterDomain*> new_loop_domain) {
  validateLoopDomain(logical(), new_loop_domain, additionalIDs());
  alternate_loop_domain_ = std::move(new_loop_domain);
}

void TensorDomain::setAllocationDomain(
    std::vector<IterDomain*> new_allocation_domain,
    std::vector<std::optional<bool>> new_contiguity,
    bool skip_validation) {
  validateContiguity(new_allocation_domain, new_contiguity);

  if (!skip_validation) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, new_allocation_domain, additional_ids_);
  }

  allocation_domain_ = std::move(new_allocation_domain);
  contiguity_ = std::move(new_contiguity);
}

std::vector<const std::vector<IterDomain*>*> TensorDomain::allDomains() const {
  std::vector<const std::vector<IterDomain*>*> all_domains = {
      &loop_domain_,
      &logical_domain_,
      &root_domain_,
      &initial_loop_domain_,
      &allocation_domain_,
      &additional_ids_};
  if (alternate_loop_domain_.has_value()) {
    all_domains.push_back(&alternate_loop_domain_.value());
  }
  return all_domains;
}

std::vector<IterDomain*> TensorDomain::allIDs() const {
  const std::vector<const std::vector<IterDomain*>*> all_domains = allDomains();
  VectorOfUniqueEntries<IterDomain*> discovered_ids;
  for (auto domain : all_domains) {
    discovered_ids.pushBack(*domain);
  }

  // We only care about IDs on the shortest path between domains
  std::unordered_multimap<IterDomain*, IterDomain*> out2in;
  for (auto i : arange(all_domains.size() - 1)) {
    if (all_domains[i]->empty()) {
      continue;
    }
    for (auto j : arange(i + 1, all_domains.size())) {
      if (all_domains[j]->empty()) {
        continue;
      }
      auto path = getExprsBetween<IRBFS>(
                      {all_domains[i]->begin(), all_domains[i]->end()},
                      {all_domains[j]->begin(), all_domains[j]->end()},
                      false)
                      .first;
      for (auto [expr, _] : path) {
        discovered_ids.pushBack(
            ir_utils::filterByType<IterDomain>(expr->outputs()));
        discovered_ids.pushBack(
            ir_utils::filterByType<IterDomain>(expr->inputs()));
        for (auto in : expr->inputs()) {
          for (auto out : expr->outputs()) {
            out2in.emplace(out->as<IterDomain>(), in->as<IterDomain>());
          }
        }
      }
    }
  }

  // Topological sort all IDs
  std::list<IterDomain*> ids_to_be_sorted(
      discovered_ids.begin(), discovered_ids.end());
  VectorOfUniqueEntries<IterDomain*> sorted_ids;
  while (!ids_to_be_sorted.empty()) {
    auto it = ids_to_be_sorted.begin();
    while (it != ids_to_be_sorted.end()) {
      auto range = out2in.equal_range(*it);
      if (std::all_of(range.first, range.second, [&](const auto& kv) {
            return sorted_ids.has(kv.second);
          })) {
        sorted_ids.pushBack(*it);
        it = ids_to_be_sorted.erase(it);
      } else {
        it++;
      }
    }
  }
  return sorted_ids.vector();
}

std::vector<Expr*> TensorDomain::allExprs() const {
  auto all_ids = allIDs();
  std::unordered_set<Val*> all_id_set{all_ids.begin(), all_ids.end()};

  VectorOfUniqueEntries<Expr*> exprs;
  for (auto id : all_ids) {
    auto def = id->definition();
    if (def == nullptr) {
      continue;
    }

    if (std::all_of(def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
          return all_id_set.find(inp) != all_id_set.end();
        })) {
      exprs.pushBack(def);
    } else {
      NVF_ERROR(std::none_of(
          def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
            return all_id_set.find(inp) != all_id_set.end();
          }));
    }
  }

  return exprs.vector();
}

std::vector<Statement*> TensorDomain::allStatements() const {
  auto all_ids = allIDs();
  std::unordered_set<Val*> all_id_set{all_ids.begin(), all_ids.end()};

  VectorOfUniqueEntries<Statement*> stmts;
  for (auto id : all_ids) {
    // Visit definition if available and all inputs are already visited
    auto def = id->definition();
    if (def != nullptr) {
      if (std::all_of(
              def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
                return all_id_set.find(inp) != all_id_set.end();
              })) {
        stmts.pushBack(def);
      } else {
        NVF_ERROR(std::none_of(
            def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
              return all_id_set.find(inp) != all_id_set.end();
            }));
      }
    }

    stmts.pushBack(id);
  }

  return stmts.vector();
}

} // namespace nvfuser

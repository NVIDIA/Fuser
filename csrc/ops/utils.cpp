// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <algorithm>
#include <limits>

#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/utils.h>

namespace nvfuser {
namespace ops {

TensorView* maybe_broadcast_inner_to_rank(TensorView* t, size_t rank) {
  size_t t_rank = TensorDomain::noReductions(t->getLogicalDomain()).size();

  // broadcast inner on inp to match rank with other.
  if (t_rank < rank) {
    const int num_bcast = static_cast<int>(rank - t_rank);
    std::vector<bool> inner_bcast_dims(rank, false);
    std::fill(
        inner_bcast_dims.begin(), inner_bcast_dims.begin() + num_bcast, true);
    t = broadcast(t, inner_bcast_dims);
  }
  return t;
}

TensorView* maybeBroadcastIndexTv(TensorView* t, size_t dim, size_t rank) {
  size_t oli_rank = TensorDomain::noReductions(t->getLogicalDomain()).size();
  NVF_ERROR(
      oli_rank == 1,
      "The rank of index tensorview in index_select must be 1, but got ",
      oli_rank);
  NVF_ERROR(
      dim < rank,
      "The dim of index_select must be < rank, but got ",
      dim,
      " >= ",
      rank);
  std::vector<bool> bcast_dims(rank, false);
  // outer broadcast on input to match rank with other.
  if (dim + 1 < rank) {
    std::fill(bcast_dims.begin() + (int64_t)dim + 1, bcast_dims.end(), true);
  }
  // inner broadcast on input to match rank with other.
  if (dim > 0) {
    std::fill(bcast_dims.begin(), bcast_dims.begin() + (int64_t)dim, true);
  }
  if (dim + 1 < rank || dim > 0) {
    t = broadcast(t, bcast_dims);
  }
  return t;
}

bool isIndexAlreadyBroadcast(
    const std::vector<IterDomain*>& index_domain,
    size_t dim,
    size_t rank) {
  // short-circuit: Expected index domain to be same size as lookup domain
  if (index_domain.size() != rank) {
    return false;
  }
  // All dimensions except for selected dimension must be a broadcast in index
  // TensorView.
  IterDomain* selected_dim = index_domain.at(dim);
  return std::all_of(
      index_domain.begin(), index_domain.end(), [&](IterDomain* id) {
        return (id == selected_dim || id->isBroadcast());
      });
}

Val* simplifiedInt(Val* val) {
  NVF_ERROR(val->isConstInt(), "Expecting Const Int's only in this routine.");
  if (val->value().hasValue()) {
    return val;
  }
  return IrBuilder::create<Val>(val->evaluate(), val->dtype());
}

// If one size is nullptr, return the other. If both symbolic just return v1. If
// one's concrete, prefer that one (simplified). If both concrete make sure
// they're the same size.
Val* promoteSize(Val* v1, Val* v2) {
  if (v1 == nullptr) {
    NVF_ERROR(
        v2 == nullptr || v2->isIntegralScalar(),
        "Expecting Int's only in this routine.");
    return v2;
  }
  if (v2 == nullptr) {
    return v1;
  }
  NVF_ERROR(
      v1->isIntegralScalar() && v2->isIntegralScalar(),
      "Expecting Int's only in this routine.");

  if (!v1->isConstInt() && !v2->isConstInt()) {
    return v1;
  } else if (v1->isConstInt() && v2->isConstInt()) {
    NVF_ERROR(
        v1->evaluate() == v2->evaluate(),
        "Expected sizes of, ",
        v1->toString(),
        " and ",
        v2->toString(),
        " to match but found ",
        v1->evaluate(),
        " and ",
        v2->evaluate(),
        ".");
    return simplifiedInt(v1);
  } else if (v1->isConstInt()) {
    return simplifiedInt(v1);
  }
  return simplifiedInt(v2);
}

// Will return a new value of type val with the DataType dtype.
Val* newScalar(ValType vtype, DataType dtype) {
  switch (vtype) {
    case (ValType::NamedScalar):
    case (ValType::Others):
      return IrBuilder::create<Val>(dtype);
    default:
      break;
  }

  NVF_CHECK(
      false,
      "Cannot handle ValType: ",
      vtype,
      " with DataType:",
      dtype,
      " in newScalar.");
}

IterType promoteIterType(IterType type1, IterType type2) {
  // Iteration: Default
  // Reduction: Should not appear here
  // Broadcast: Propagated only if type1 and type2 are Broadcast
  // GatherScatter: Converted to Iteration
  // Stride: Shold not appear here
  // VectorComponent: Converted to Iteration

  NVF_ERROR(
      type1 != IterType::Reduction && type1 != IterType::Stride,
      "Invalid IterType: ",
      type1)
  NVF_ERROR(
      type2 != IterType::Reduction && type2 != IterType::Stride,
      "Invalid IterType: ",
      type2);

  // Do not propagate GatherScatter and VectorComponent
  if (type1 == IterType::VectorComponent || type1 == IterType::GatherScatter) {
    type1 = IterType::Iteration;
  }
  if (type2 == IterType::VectorComponent || type2 == IterType::GatherScatter) {
    type2 = IterType::Iteration;
  }

  // At this point, type1 and type2 must be either Iteration or
  // Broadcast. Note Symbolic is either Iteration or Broadcast
  NVF_ERROR(
      type1 == IterType::Iteration || type1 == IterType::Broadcast ||
          type1 == IterType::Symbolic,
      "Unexpected IterType: ",
      type1);
  NVF_ERROR(
      type2 == IterType::Iteration || type2 == IterType::Broadcast ||
          type2 == IterType::Symbolic,
      "Unexpected IterType: ",
      type2);

  // If either is Iteration, the output type is also Iteration. If
  // none of them is Iteration and either of them is Symbolic, the
  // output is also Symbolic.
  if (type1 == IterType::Iteration || type2 == IterType::Iteration) {
    return IterType::Iteration;
  } else if (type1 == IterType::Symbolic || type2 == IterType::Symbolic) {
    return IterType::Symbolic;
  } else {
    return IterType::Broadcast;
  }
}

//! For MatmulOp, the input iterdomains at a given index do not necessarily map
//! to the output iterdomain at that index This function aligns the input
//! iterdomain to the output and returns a vector where each element is the
//! input iterdomain corresponding to the output iterdomain at that index.
//! If the element is nullptr, there is no mapping between input-output at that
//! index.
std::vector<IterDomain*> mapMatmulOpIterDomains(
    const std::vector<IterDomain*>& input_domain,
    int64_t input_position,
    size_t out_size) {
  NVF_ERROR(
      input_position == 0 || input_position == 1,
      "Input position must be 0 or 1. Found ",
      input_position);
  std::vector<IterDomain*> mapping(out_size, nullptr);
  auto inp_size = (int64_t)input_domain.size();

  // Input A to matmul: {*, M, K}
  // Input B to matmul: {*, K, N}
  auto kpos = inp_size - 1;
  if (input_position == 1 && inp_size > 1) {
    kpos = inp_size - 2;
  }
  bool k_bcast = input_domain.at(kpos)->isBroadcast();
  int64_t red_dims = k_bcast ? 0 : 1;

  // Last position is a reduction dimension mapping to K if K is not broadcast.
  if (!k_bcast) {
    mapping[out_size - 1] = input_domain.at(kpos);
    ;
  }

  if (inp_size == 1) {
    return mapping;
  }

  for (auto out_idx = (int64_t)out_size - 1 - red_dims, inp_idx = inp_size - 1;
       inp_idx >= 0;
       inp_idx--) {
    if (inp_idx != kpos) {
      mapping[out_idx] = input_domain[inp_idx];
      out_idx--;
    }
    // Consider [iM, iK] x [iK]: [iM, rK]. Since out_size < inp_size,
    // input A and output are not right-aligned. In this case, the output index
    // pointer should not be moved when the reduction axis is encountered.
    else if (inp_size <= (int64_t)out_size - red_dims) {
      out_idx--;
    }
  }

  return mapping;
}

std::vector<IterDomain*> mapLinearOpIterDomains(
    const std::vector<IterDomain*>& input_domain,
    int64_t input_position,
    size_t out_size,
    bool k_bcast) {
  std::vector<IterDomain*> mapping(out_size, nullptr);

  // Input: {*_i, K}
  // Weight: {*_wb, N, K}
  // Bias: {*_wb, N}
  // Output: {*_wb, *_i, N, (rK)}. rK exists iff K is not a broadcast.
  if (input_position == 0) {
    // Fill `mapping` from the back.
    auto in_r_index = static_cast<int64_t>(input_domain.size()) - 1;
    auto out_index = static_cast<int64_t>(out_size) - 1;
    // Map K if K is not a broadcast.
    if (!k_bcast) {
      mapping[out_index] = input_domain[in_r_index];
      out_index--;
    }
    in_r_index--;

    // Skip N because it's not in the input.
    out_index--;

    // Map the rest, i.e., *_i.
    while (in_r_index >= 0) {
      mapping[out_index] = input_domain[in_r_index];
      in_r_index--;
      out_index--;
    }
  } else {
    NVF_ERROR(
        input_position == 1 || input_position == 2,
        "Input position must be 0, 1, or 2. Found ",
        input_position);

    auto in_r_index = static_cast<int64_t>(input_domain.size()) - 1;
    auto out_index = static_cast<int64_t>(out_size) - 1;
    if (k_bcast) {
      // If K is a broadcast, don't map K.
      if (input_position == 1) {
        // Skip K in the weight.
        in_r_index--;
      }
    } else {
      // Otherwise, map K in the weight.
      if (input_position == 1) {
        mapping[out_index] = input_domain[in_r_index];
        in_r_index--;
      }
      out_index--;
    }

    // Fill `N`
    mapping[out_index] = input_domain[in_r_index];

    // Fill *_wb from the front.
    out_index = 0;
    for (auto in_index : arange(in_r_index)) {
      mapping[out_index] = input_domain[in_index];
      out_index++;
    }
  }
  return mapping;
}

namespace {
ParallelType promoteParallelType(ParallelType a, ParallelType b) {
  if (a == b) {
    return a;
  }
  NVF_ERROR(
      a == ParallelType::Serial || b == ParallelType::Serial,
      "Doesn't know how to resolve ",
      a,
      " and ",
      b,
      " at this moment.");
  return a == ParallelType::Serial ? b : a;
}
} // namespace

// Adding these pragmas since gcc-12.2.1
// incorrectly reports a warning with the use of evaluate
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
IterDomain* newOutputIterDomain(
    const std::vector<IterDomain*>& input_ids,
    const std::optional<IterType> force_iter_type) {
  // For the start and stop offsets, take the maximum of input axes.
  // For now, the offsets of both start and stop are always integer
  // constant, so we can statically compute them. It is unclear
  // whether we would need to support dynamic offsetting, e.g.,
  // shifting by a dynamic offset.
  int64_t start_offset = 0;
  int64_t stop_offset = 0;
  Val* extent_val = nullptr;
  bool extent_is_from_symbolic = true;
  Val* expanded_extent_val = nullptr;
  auto parallel_type = ParallelType::Serial;
  std::optional<IterType> iter_type = std::nullopt;

  for (auto id : input_ids) {
    // Filter out any nullptrs
    if (id == nullptr) {
      continue;
    }

    NVF_ERROR(
        id->getParallelType() == ParallelType::Serial ||
            isParallelTypeDeviceDim(id->getParallelType()),
        id->getParallelType(),
        " is not expected when building ops.");
    parallel_type = promoteParallelType(parallel_type, id->getParallelType());

    if (id->isBroadcast()) {
      if (id->hasExpandedExtent()) {
        expanded_extent_val =
            promoteSize(expanded_extent_val, id->expandedExtent());
      }
      continue;
    }

    if (extent_is_from_symbolic && !id->isSymbolic()) {
      // We prefer to use extents from non-Symbolic inputs if there are any
      // because they might indicate a broadcast axis that is resolved in this
      // op.
      extent_val = id->extent();
      extent_is_from_symbolic = false;
    }
    extent_val = promoteSize(extent_val, id->extent());
    if (iter_type.has_value()) {
      iter_type = promoteIterType(iter_type.value(), id->getIterType());
    } else {
      iter_type = id->getIterType();
    }

    auto id_start_offset = id->start();
    auto id_stop_offset = id->stopOffset();
    // Currently, start is always constant
    NVF_ERROR(
        id_start_offset->isConstInt(),
        "Invalid IterDomain start: ",
        id_start_offset);
    NVF_ERROR(
        id_stop_offset->isConstInt(),
        "Invalid IterDomain stop offset: ",
        id_stop_offset);
    start_offset =
        std::max(start_offset, id_start_offset->evaluate().as<int64_t>());
    stop_offset =
        std::max(stop_offset, id_stop_offset->evaluate().as<int64_t>());
  }

  if (force_iter_type.has_value()) {
    // Use forced iter_type instead of the one inferred from the input IDs
    iter_type = force_iter_type.value();
  }

  IterDomain* out_domain = nullptr;
  if (extent_val != nullptr) {
    NVF_ERROR(
        iter_type.has_value(),
        "Could not deduce iter type for new tensor view.");
    out_domain =
        IterDomainBuilder(
            IrBuilder::create<Val>(start_offset, DataType::Index), extent_val)
            .stop_offset(IrBuilder::create<Val>(stop_offset, DataType::Index))
            .parallel_type(parallel_type)
            .iter_type(iter_type.value())
            .build();
  } else {
    out_domain = IterDomainBuilder(
                     FusionGuard::getCurFusion()->zeroVal(),
                     FusionGuard::getCurFusion()->oneVal())
                     .expanded_extent(expanded_extent_val)
                     .parallel_type(parallel_type)
                     .iter_type(IterType::Broadcast)
                     .build();
  }
  return out_domain;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

std::vector<IterDomain*> newOutputDomain(const std::vector<Val*>& vals) {
  std::vector<TensorView*> tvs;
  for (auto val : vals) {
    if (auto* tv = dynamic_cast<TensorView*>(val)) {
      tvs.push_back(tv);
    }
  }
  NVF_CHECK(
      !tvs.empty(),
      "Tried to create new output TensorView but received empty list.");

  std::vector<IterDomain*> out_domain(
      TensorDomain::noReductions(tvs[0]->getLogicalDomain()).size(), nullptr);

  for (const auto dim_i : arange(out_domain.size())) {
    std::vector<IterDomain*> input_ids;
    input_ids.reserve(tvs.size());
    for (auto* tv : tvs) {
      auto dom = TensorDomain::noReductions(tv->getLogicalDomain());
      input_ids.emplace_back(dom[dim_i]);
    }
    out_domain[dim_i] = newOutputIterDomain(input_ids);
  }
  return out_domain;
}

TensorView* newOutputTV(const std::vector<Val*>& vals, DataType dtype) {
  auto out_domain = newOutputDomain(vals);
  auto* new_out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      dtype);

  DeviceMesh new_mesh;
  // Find the first input that has a mesh. This seems arbitrary, but is at this
  // moment safest because it's consistent with PropagateShardingsPass.
  for (auto* tv : ir_utils::filterByType<TensorView>(vals)) {
    if (tv->hasDeviceMesh()) {
      new_mesh = tv->getDeviceMesh();
      break;
    }
  }
  new_out->setDeviceMesh(new_mesh);

  return new_out;
}

std::vector<Val*> maybeBroadcast(const std::vector<Val*>& vals) {
  std::vector<Val*> out_vals(vals.size(), nullptr);
  size_t n_dims = 0;
  for (auto val : vals) {
    if (val->getValType().value() == ValType::TensorView) {
      n_dims = std::max(
          n_dims,
          TensorDomain::noReductions(val->as<TensorView>()->getLogicalDomain())
              .size());
    }
  }

  for (const auto i : arange(vals.size())) {
    if (vals[i]->getValType().value() == ValType::TensorView) {
      auto tv = vals[i]->as<TensorView>();
      out_vals[i] = maybe_broadcast_inner_to_rank(tv, n_dims);
    } else {
      out_vals[i] = vals[i];
    }
  }
  return out_vals;
}

Val* newValLike(Val* val, DataType dtype) {
  NVF_CHECK(
      dtype != DataType::Null, "Invalid datatype provided for new value.");

  if (val->isA<TensorView>()) {
    return newOutputTV({val}, dtype);
  }

  return newScalar(ValType::Others, dtype);
}

// returns the minimum init value for reduction:
//   -inf for floating type;
//   lowest value for integer type;
//   false for bool.
Val* getMinimumValue(DataType v) {
  switch (std::get<PrimDataType>(v.type)) {
    case (DataType::Double):
      return IrBuilder::create<Val>(-std::numeric_limits<double>::infinity());
      break;
    case (DataType::Float):
      return IrBuilder::create<Val>(
          static_cast<double>(-std::numeric_limits<float>::infinity()));
      break;
    case (DataType::Half):
      return IrBuilder::create<Val>(
          static_cast<double>(-std::numeric_limits<c10::Half>::infinity()));
      break;
    case DataType::BFloat16:
      return IrBuilder::create<Val>(
          static_cast<double>(-std::numeric_limits<c10::BFloat16>::infinity()));
      break;
    case DataType::Float8_e4m3fn:
      // e4m3 is finite.
      return IrBuilder::create<Val>(
          static_cast<double>(-std::numeric_limits<c10::Float8_e4m3fn>::max()));
      break;
    case DataType::Float8_e5m2:
      return IrBuilder::create<Val>(static_cast<double>(
          -std::numeric_limits<c10::Float8_e5m2>::infinity()));
      break;
    case DataType::Float8_e8m0fnu:
      // e8m0 is finite.
      return IrBuilder::create<Val>(static_cast<double>(
          -std::numeric_limits<c10::Float8_e8m0fnu>::max()));
      break;
    case (DataType::Int):
      return IrBuilder::create<Val>(std::numeric_limits<int64_t>::lowest());
      break;
    case (DataType::Int32):
      return IrBuilder::create<Val>(
          (int64_t)std::numeric_limits<int32_t>::lowest());
      break;
    case (DataType::Bool):
      return IrBuilder::create<Val>(false);
      break;
    default:
      NVF_CHECK(false, "Could not generate a min op for tensor with type: ", v);
  }
  return nullptr;
}

// returns the maximum init value for reduction:
//   inf for floating type;
//   highest value for integer type;
//   true for bool.
Val* getMaximumValue(DataType v) {
  switch (std::get<PrimDataType>(v.type)) {
    case (DataType::Double):
      return IrBuilder::create<Val>(std::numeric_limits<double>::infinity());
      break;
    case (DataType::Float):
      return IrBuilder::create<Val>(std::numeric_limits<float>::infinity());
      break;
    case (DataType::Half):
      return IrBuilder::create<Val>(
          static_cast<double>(std::numeric_limits<c10::Half>::infinity()));
      break;
    case DataType::BFloat16:
      return IrBuilder::create<Val>(
          static_cast<double>(std::numeric_limits<c10::BFloat16>::infinity()));
      break;
    case DataType::Float8_e4m3fn:
      // e4m3 is finite.
      return IrBuilder::create<Val>(
          static_cast<double>(std::numeric_limits<c10::Float8_e4m3fn>::max()));
      break;
    case DataType::Float8_e5m2:
      return IrBuilder::create<Val>(static_cast<double>(
          std::numeric_limits<c10::Float8_e5m2>::infinity()));
      break;
    case DataType::Float8_e8m0fnu:
      // e8m0 is finite.
      return IrBuilder::create<Val>(
          static_cast<double>(std::numeric_limits<c10::Float8_e8m0fnu>::max()));
      break;
    case DataType::Int:
      return IrBuilder::create<Val>(std::numeric_limits<int64_t>::max());
      break;
    case DataType::Int32:
      return IrBuilder::create<Val>(
          (int64_t)std::numeric_limits<int32_t>::max());
      break;
    case DataType::Bool:
      return IrBuilder::create<Val>(true);
      break;
    default:
      NVF_CHECK(false, "Could not generate a max op for tensor with type: ", v);
  }
  return nullptr;
}

std::vector<unsigned int> canonicalizeAxes(
    const std::vector<int64_t>& axes,
    int64_t ndims) {
  std::vector<unsigned int> uint_axes;
  uint_axes.reserve(axes.size());
  std::transform(
      axes.begin(), axes.end(), std::back_inserter(uint_axes), [&](int axis) {
        return (unsigned int)wrapDim(axis, ndims);
      });
  return uint_axes;
}

Val* binOpIdentity(BinaryOpType op_type, DataType dtype) {
  Fusion* fusion = FusionGuard::getCurFusion();
  switch (op_type) {
    case BinaryOpType::Add:
      return fusion->zeroVal(dtype);
    case BinaryOpType::Mul:
      return fusion->oneVal(dtype);
    case BinaryOpType::Min:
      return getMaximumValue(dtype);
    case BinaryOpType::Max:
      return getMinimumValue(dtype);
    case BinaryOpType::LogicalAnd:
      NVF_ERROR(isBooleanType(dtype));
      return fusion->trueVal();
    case BinaryOpType::LogicalOr:
      NVF_ERROR(isBooleanType(dtype));
      return fusion->falseVal();
    default:
      NVF_THROW("Binary op ", op_type, " has no two-sided inverse");
  }
  return nullptr;
}

} // namespace ops
} // namespace nvfuser

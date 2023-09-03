// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/builder.h>
#include <ops/arith.h>
#include <ops/utils.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <limits>

namespace nvfuser {
namespace ops {

TensorView* maybe_broadcast_inner_to_rank(TensorView* t, size_t rank) {
  size_t t_rank = TensorDomain::noReductions(t->getMaybeRFactorDomain()).size();

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

TensorView* maybe_broadcast_index_tv(TensorView* t, size_t dim, size_t rank) {
  size_t ori_rank =
      TensorDomain::noReductions(t->getMaybeRFactorDomain()).size();
  NVF_ERROR(
      ori_rank == 1,
      "The rank of index tensorview in index_select must be 1, but got ",
      ori_rank);
  NVF_ERROR(
      dim < rank,
      "The dim of index_select must be < rank, but got ",
      dim,
      " >= ",
      rank);
  std::vector<bool> bcast_dims(rank, false);
  // broadcast outter on inp to match rank with other.
  if (dim + 1 < rank) {
    std::fill(bcast_dims.begin() + (int64_t)dim + 1, bcast_dims.end(), true);
  }
  // broadcast inner on inp to match rank with other.
  if (dim > 0) {
    std::fill(bcast_dims.begin(), bcast_dims.begin() + (int64_t)dim, true);
  }
  if (dim + 1 < rank || dim > 0) {
    t = broadcast(t, bcast_dims);
  }
  return t;
}

Val* simplifiedInt(Val* val) {
  NVF_ERROR(val->isConstInt(), "Expecting Const Int's only in this routine.");
  if (val->value().hasValue()) {
    return val;
  }
  return IrBuilder::create<Val>(val->evaluateInt(), val->dtype());
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
        v1->evaluateInt() == v2->evaluateInt(),
        "Expected sizes of, ",
        v1->toString(),
        " and ",
        v2->toString(),
        " to match but found ",
        v1->evaluateInt(),
        " and ",
        v2->evaluateInt(),
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
  // Gather: Converted to Iteration
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

  // Do not propagate Gather and VectorComponent
  if (type1 == IterType::Gather || type1 == IterType::VectorComponent ||
      type1 == IterType::GatherScatter) {
    type1 = IterType::Iteration;
  }
  if (type2 == IterType::Gather || type2 == IterType::VectorComponent ||
      type2 == IterType::GatherScatter) {
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

std::vector<IterDomain*> newOutputDomain(
    const std::vector<Val*>& vals,
    DataType dtype) {
  std::vector<TensorView*> tvs;
  for (auto val : vals) {
    if (val->getValType() == ValType::TensorView) {
      tvs.push_back(val->as<TensorView>());
    }
  }
  NVF_CHECK(
      !tvs.empty(),
      "Tried to create new output TensorView but received empty list.");

  std::vector<IterDomain*> out_domain(
      TensorDomain::noReductions(tvs[0]->getMaybeRFactorDomain()).size(),
      nullptr);

  // For the start and stop offsets, take the maximum of input axes.
  // For now, the offsets of both start and stop are always integer
  // constant, so we can statically compute them. It is unclear
  // whether we would need to support dynamic offsetting, e.g.,
  // shifting by a dynamic offset.
  std::vector<int64_t> start_offsets(out_domain.size(), 0);
  std::vector<int64_t> stop_offsets(out_domain.size(), 0);
  std::vector<Val*> extent_vals(out_domain.size(), nullptr);
  std::vector<Val*> expanded_extent_vals(out_domain.size(), nullptr);
  std::vector<std::optional<IterType>> iter_types(
      out_domain.size(), std::nullopt);

  for (auto tv : tvs) {
    auto dom = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    NVF_ERROR(
        dom.size() == out_domain.size(),
        "Invalid tensor view found while producing an output, it has ",
        dom.size(),
        " dimensions but expected ",
        out_domain.size());
    for (const auto i : c10::irange(dom.size())) {
      if (dom[i]->isBroadcast()) {
        if (dom[i]->hasExpandedExtent()) {
          expanded_extent_vals[i] =
              promoteSize(expanded_extent_vals[i], dom[i]->expandedExtent());
        }
        continue;
      }
      extent_vals[i] = promoteSize(extent_vals[i], dom[i]->extent());
      if (iter_types[i].has_value()) {
        iter_types[i] =
            promoteIterType(iter_types[i].value(), dom[i]->getIterType());
      } else {
        iter_types[i] = dom[i]->getIterType();
      }

      auto start_offset = dom[i]->start();
      auto stop_offset = dom[i]->stopOffset();
      // Currently, start is always constant
      NVF_ERROR(
          start_offset->isConstInt(),
          "Invalid IterDomain start: ",
          start_offset);
      NVF_ERROR(
          stop_offset->isConstInt(),
          "Invalid IterDomain stop offset: ",
          stop_offset);
      start_offsets[i] =
          std::max(start_offsets[i], start_offset->evaluateInt());
      stop_offsets[i] = std::max(stop_offsets[i], stop_offset->evaluateInt());
    }
  }
  for (const auto dim_i : c10::irange(out_domain.size())) {
    if (extent_vals[dim_i] != nullptr) {
      NVF_ERROR(
          iter_types[dim_i].has_value(),
          "Could not deduce iter type for new tensor view.");
      out_domain[dim_i] =
          IterDomainBuilder(
              IrBuilder::create<Val>(start_offsets[dim_i], DataType::Index),
              extent_vals[dim_i])
              .stop_offset(
                  IrBuilder::create<Val>(stop_offsets[dim_i], DataType::Index))
              .iter_type(iter_types[dim_i].value())
              .build();
    } else {
      out_domain[dim_i] = IterDomainBuilder(
                              FusionGuard::getCurFusion()->zeroVal(),
                              FusionGuard::getCurFusion()->oneVal())
                              .expanded_extent(expanded_extent_vals[dim_i])
                              .iter_type(IterType::Broadcast)
                              .build();
    }
    // Set exact mapping for each of the non-broadcast input IDs
    for (auto tv : tvs) {
      auto id =
          TensorDomain::noReductions(tv->getMaybeRFactorDomain()).at(dim_i);
      // TODO: Set up proper exact mapping here
      if (!id->isSymbolic() &&
          id->getIterType() == out_domain[dim_i]->getIterType()) {
        id->setExactMapped(out_domain[dim_i]);
      }
    }
  }

  return out_domain;
}

TensorView* newOutputTV(const std::vector<Val*>& vals, DataType dtype) {
  auto out_domain = newOutputDomain(vals, dtype);
  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      dtype);
}

std::vector<Val*> maybeBroadcast(const std::vector<Val*>& vals) {
  std::vector<Val*> out_vals(vals.size(), nullptr);
  size_t n_dims = 0;
  for (auto val : vals) {
    if (val->getValType().value() == ValType::TensorView) {
      n_dims = std::max(
          n_dims,
          TensorDomain::noReductions(
              val->as<TensorView>()->getMaybeRFactorDomain())
              .size());
    }
  }

  for (const auto i : c10::irange(vals.size())) {
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

  const ValType vtype = val->getValType().value();

  if (vtype == ValType::TensorView) {
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
    case (DataType::Int):
      return IrBuilder::create<Val>(std::numeric_limits<int64_t>::max());
      break;
    case (DataType::Int32):
      return IrBuilder::create<Val>(
          (int64_t)std::numeric_limits<int32_t>::max());
      break;
    case (DataType::Bool):
      return IrBuilder::create<Val>(true);
      break;
    default:
      NVF_CHECK(false, "Could not generate a max op for tensor with type: ", v);
  }
  return nullptr;
}

} // namespace ops
} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>

#include <exceptions.h>
#include <interval_analysis.h>
#include <kernel_ir_dispatch.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace nvfuser {

BoundedInt BoundedInt::operator+(const BoundedInt& other) const {
  return BoundedInt{min + other.min, max + other.max};
}

BoundedInt BoundedInt::operator+(const int64_t other) const {
  return BoundedInt{min + other, max + other};
}

BoundedInt BoundedInt::operator-(const BoundedInt& other) const {
  return BoundedInt{min - other.max, max - other.min};
}

BoundedInt BoundedInt::operator-(const int64_t other) const {
  return BoundedInt{min - other, max - other};
}

BoundedInt BoundedInt::operator*(const BoundedInt& other) const {
  // TODO: How should we handle overflow here?
  std::vector<int64_t> xs{
      min * other.min, min * other.max, max * other.min, max * other.max};
  return BoundedInt{
      *std::min_element(xs.begin(), xs.end()),
      *std::max_element(xs.begin(), xs.end())};
}

BoundedInt BoundedInt::operator*(const int64_t other) const {
  if (other < 0L) {
    return BoundedInt{max * other, min * other};
  }
  return BoundedInt{min * other, max * other};
}

BoundedInt BoundedInt::operator/(const BoundedInt& other) const {
  // Note that division by zero will be a runtime error anyway, so we can
  // ignore it for this analysis. This means that if this or other has
  // negative min and positive max, we should consider the union of up to four
  // different bounds
  auto split_ranges_around_zero = [](const BoundedInt& b) {
    std::vector<BoundedInt> ranges;
    if (b.min < 0L) {
      ranges.push_back({b.min, std::min(b.max, -1L)});
    }
    if (b.max > 0L) {
      ranges.push_back({std::max(b.min, 1L), b.max});
    }
    return ranges;
  };
  const std::vector<BoundedInt> numer_ranges = split_ranges_around_zero(*this);
  const std::vector<BoundedInt> denom_ranges = split_ranges_around_zero(other);

  BoundedInt result;
  bool first = true;
  for (const BoundedInt& numer : numer_ranges) {
    for (const BoundedInt& denom : denom_ranges) {
      BoundedInt simple_range;
      // numer and denom are each either only negative or only positive
      if (numer.min > 0) {
        if (denom.min > 0) {
          // positive over positive
          simple_range = {numer.min / denom.max, numer.max / denom.min};
        } else {
          // positive over negative
          simple_range = {numer.max / denom.max, numer.min / denom.min};
        }
      } else {
        if (denom.min > 0) {
          // negative over positive
          simple_range = {numer.min / denom.min, numer.max / denom.max};
        } else {
          // negative over negative
          simple_range = {numer.max / denom.min, numer.min / denom.max};
        }
      }
      // Result is the union over all of the simple ranges
      if (first) {
        result = simple_range;
      } else {
        result.min = std::min(result.min, simple_range.min);
        result.max = std::max(result.max, simple_range.max);
      }
      first = false;
    }
  }
  return result;
}

BoundedInt BoundedInt::operator/(const int64_t other) const {
  if (other == 0L) {
    // division by zero case. Return unbounded
    return BoundedInt();
  } else if (other < 0) {
    return BoundedInt{max / other, min / other};
  } else {
    return BoundedInt{min / other, max / other};
  }
}

BoundedInt BoundedInt::operator%(const BoundedInt& other) const {
  if (min >= 0L && other.min >= 0L && max < other.min) {
    return {min, max};
  } else {
    // NOTE: this might not be true if this or other is negative
    return BoundedInt{0L, std::min(max, other.max - 1L)};
  }
}

BoundedInt BoundedInt::operator%(const int64_t other) const {
  if (min >= 0L && max < other) {
    return {min, max};
  } else {
    return {0L, other - 1L};
  }
}

//! Returns the number of high bits that must be common among all integers in
//! this interval
//!
//! Example:
//!   min = 0b10101010
//!   max = 0b10101100
//!
//!   All numbers in this range are of the form 0b10101XXX
//!     different_bits = 0b110
//!     num_common_bits = 61
int64_t BoundedInt::countCommonHighBits() const {
  // Reinterpret integers as unsigned, so that bitwise ops and
  // std::countl_zero are well-defined
  uint64_t different_bits = (*reinterpret_cast<const uint64_t*>(&max)) ^
      (*reinterpret_cast<const uint64_t*>(&min));
#if __cplusplus < 202002L
  // TODO: add countl_zero to csrc/C++20/ somewhere for C++17 backward
  // compatibility
  int64_t num_common_bits = 64L;
  while (different_bits != 0L) {
    different_bits >>= 1;
    num_common_bits--;
  }
  return num_common_bits;
#else
  return (int64_t)std::countl_zero(different_bits);
#endif
}

// For bitwise operations, we consider the range of each bit independently.
// Consider a number x=0bABCDE. If min(x)=max(x), then each of the bits A, B,
// C, D, and E are fixed. However, if there is a range of values possible then
// a subset of these bits could take on either 0 or 1. Suppose the range of x
// is [0b01010, 0b01100]. Then we know that A=0, B=1, and C, D, and E can have
// either value. Generally speaking, for numbers lying between two positive
// integers, we know the lower-most K many bits are not fixed, where K is
// PRECISION-(number of high bits in common). We can compute the largest K
// between this and other, then we know that the XOR between these two values
// can have any value for that many lower bits and all the higher bits are
// determined by XORing the two min (or max) bounds with one another.
//
// [Note on twos-complement negative integers]
// Since twos-complement negative integers can be envisioned as simply
// stacking (without flipping) the negative values at the right side of the
// positive values, we can apply the same algorithm regardless of signedness.
BoundedInt BoundedInt::operator^(const BoundedInt& other) const {
  // New interval has this many fixed bits
  int64_t fixed_bits =
      std::min(countCommonHighBits(), other.countCommonHighBits());
  // Mask everything below the higher fixed_bits
  int64_t low_mask = (1 << fixed_bits) - 1; // 0b00111
  int64_t new_min = (min ^ other.min) & (~low_mask); // 0b01000
  int64_t new_max = new_min + low_mask; // 0b01111
  return {new_min, new_max};
}

BoundedInt BoundedInt::operator&(const BoundedInt& other) const {
  // New interval has this many fixed bits
  int64_t fixed_bits =
      std::min(countCommonHighBits(), other.countCommonHighBits());
  // Mask everything below the higher fixed_bits
  int64_t low_mask = (1 << fixed_bits) - 1; // 0b00111
  int64_t new_min = (min & other.min) & (~low_mask); // 0b01000
  int64_t new_max = new_min + low_mask; // 0b01111
  return {new_min, new_max};
}

BoundedInt BoundedInt::operator|(const BoundedInt& other) const {
  // New interval has this many fixed bits
  int64_t fixed_bits =
      std::min(countCommonHighBits(), other.countCommonHighBits());
  // Mask everything below the higher fixed_bits
  int64_t low_mask = (1 << fixed_bits) - 1; // 0b00111
  int64_t new_min = (min | other.min) & (~low_mask); // 0b01000
  int64_t new_max = new_min + low_mask; // 0b01111
  return {new_min, new_max};
}

BoundedInt BoundedInt::operator~() const {
  // New interval has this many fixed bits
  int64_t fixed_bits = countCommonHighBits();
  // Mask everything below the higher fixed_bits
  int64_t low_mask = (1 << fixed_bits) - 1; // 0b00111
  int64_t new_min = (~min) & (~low_mask); // 0b01000
  int64_t new_max = new_min + low_mask; // 0b01111
  return {new_min, new_max};
}

// Index types are always signed (always going to be true?). This means that a
// right shift is _arithmetic_ right shift, so if the argument is negative,
// after the shift it stays negative.
BoundedInt BoundedInt::operator>>(const BoundedInt& other) const {
  NVF_ERROR(other.min >= 0, "Shift operator must not have negative shift");
  // Note: arithmetic right shift makes negative values closer to zero, as it
  // does for positive values
  int64_t new_min = (min < 0L) ? (min >> other.min) : (min >> other.max);
  int64_t new_max = (max < 0L) ? (max >> other.max) : (min >> other.min);
  return {new_min, new_max};
}

BoundedInt ceilDiv(const BoundedInt& numer, const BoundedInt& denom) {
  // NOTE: This is very similar to operator/
  auto split_ranges_around_zero = [](const BoundedInt& b) {
    std::vector<BoundedInt> ranges;
    if (b.min < 0L) {
      ranges.push_back({b.min, std::min(b.max, -1L)});
    }
    if (b.max > 0L) {
      ranges.push_back({std::max(b.min, 1L), b.max});
    }
    return ranges;
  };
  const std::vector<BoundedInt> numer_ranges = split_ranges_around_zero(numer);
  const std::vector<BoundedInt> denom_ranges = split_ranges_around_zero(denom);

  BoundedInt result;
  bool first = true;
  for (const BoundedInt& numer : numer_ranges) {
    for (const BoundedInt& denom : denom_ranges) {
      BoundedInt simple_range;
      // numer and denom are each either only negative or only positive
      if (numer.min > 0) {
        if (denom.min > 0) {
          // positive over positive
          simple_range = {
              ceilDiv(numer.min, denom.max), ceilDiv(numer.max, denom.min)};
        } else {
          // positive over negative
          simple_range = {
              ceilDiv(numer.max, denom.max), ceilDiv(numer.min, denom.min)};
        }
      } else {
        if (denom.min > 0) {
          // negative over positive
          simple_range = {
              ceilDiv(numer.min, denom.min), ceilDiv(numer.max, denom.max)};
        } else {
          // negative over negative
          simple_range = {
              ceilDiv(numer.max, denom.min), ceilDiv(numer.min, denom.max)};
        }
      }
      // Result is the union over all of the simple ranges
      if (first) {
        result = simple_range;
      } else {
        result.min = std::min(result.min, simple_range.min);
        result.max = std::max(result.max, simple_range.max);
      }
      first = false;
    }
  }
  return result;
}

BoundedInt BoundedInt::operator<<(const BoundedInt& other) const {
  NVF_ERROR(
      min >= 0,
      "Left shift must not be applied to number that can be negative");
  NVF_ERROR(other.min >= 0, "Shift operator must not have negative shift");
  return {min << other.min, max << other.max};
}

ScalarBoundsCalculator::ScalarBoundsCalculator(
    kir::Kernel* kernel,
    ExpressionEvaluator& expr_eval,
    const LaunchParams& launch_params)
    : expr_eval_(expr_eval), launch_params_(launch_params) {
  if (kernel != nullptr) {
    // If kernel is given, process all exprs in it
    kir::IrVisitor::handle(kernel->topLevelExprs());
  }
}

//! Return the bounds, computed over all scalars in the fusion with the given
//! data type
BoundedInt ScalarBoundsCalculator::boundByDataType(DataType dtype) {
  BoundedInt ret;
  bool initialized = false;
  for (auto& [val, b] : bounds_) {
    if (val->dtype() != dtype) {
      continue;
    }
    if (!initialized) {
      ret = b;
      initialized = true;
    } else {
      ret.min = std::min(ret.min, b.min);
      ret.max = std::max(ret.max, b.max);
    }
    if (b.min < std::numeric_limits<int32_t>::min() ||
        b.max > std::numeric_limits<int32_t>::max()) {
    }
  }
  return ret;
}

//! Look at all casts (T)x where x is of type nvfuser_index_t, to ensure that
//! these casts are safe i.e. that the bounds of x do not overflow those
//! representable by T.
bool ScalarBoundsCalculator::castsFromIndexAreSafe() const {
  return std::all_of(
      casts_from_index_.begin(), casts_from_index_.end(), [&](UnaryOp* cast) {
        const BoundedInt& bounds = bounds_.at(cast->in());
        DataType out_dtype = cast->out()->dtype();
        NVF_ERROR(
            std::holds_alternative<PrimDataType>(out_dtype.type),
            "Expected PrimDataType but found ",
            out_dtype);
        switch (std::get<PrimDataType>(out_dtype.type)) {
          case DataType::Int:
            return true;
          case DataType::Int32:
            return bounds.min >= std::numeric_limits<int32_t>::min() &&
                bounds.max <= std::numeric_limits<int32_t>::max();
          case DataType::Short:
            return bounds.min >= std::numeric_limits<int16_t>::min() &&
                bounds.max <= std::numeric_limits<int16_t>::max();
          case DataType::Char:
            return bounds.min >= std::numeric_limits<int8_t>::min() &&
                bounds.max <= std::numeric_limits<int8_t>::max();
          case DataType::UInt64:
            // upper limit is above that of int64_t, which is the type of
            // bounds.max
            return bounds.min >= 0L;
          case DataType::UInt32:
            return bounds.min >= std::numeric_limits<uint32_t>::min() &&
                bounds.max <= std::numeric_limits<uint32_t>::max();
          case DataType::UInt16:
            return bounds.min >= std::numeric_limits<uint16_t>::min() &&
                bounds.max <= std::numeric_limits<uint16_t>::max();
          case DataType::Byte:
            return bounds.min >= std::numeric_limits<uint8_t>::min() &&
                bounds.max <= std::numeric_limits<uint8_t>::max();
            return true;
          default:
            NVF_THROW("Unhandled DataType ", out_dtype);
            return false;
        }
      });
}

std::ostream& operator<<(std::ostream& out, const BoundedInt& b) {
  out << "BoundedInt[" << b.min << ", " << b.max << "]";
  return out;
}

void ScalarBoundsCalculator::setBounds(Val* val, const BoundedInt& bounds) {
  bounds_[val] = bounds;
}

void ScalarBoundsCalculator::setBounds(Val* val, int64_t min, int64_t max) {
  setBounds(val, {min, max});
}

void ScalarBoundsCalculator::setAsUnbounded(Val* val) {
  setBounds(
      val,
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max());
}

void ScalarBoundsCalculator::boundNamedScalar(NamedScalar* scalar) {
  if (std::optional<ParallelType> ptype = scalar->getParallelDim();
      ptype.has_value()) {
    // scalar is the extent of a parallel dim, so evaluate it
    int64_t dim_int = launch_params_.getDim(ptype.value());
    setBounds(scalar, dim_int, dim_int);
  } else if (std::optional<ParallelType> ptype = scalar->getParallelIndex();
             ptype.has_value()) {
    // scalar is the index of a parallel dim, so bound it by [0, dim-1]
    int64_t dim_int = launch_params_.getDim(ptype.value());
    setBounds(scalar, 0L, dim_int - 1L);
  } else {
    // We do not know how to bound other NamedScalars
    setAsUnbounded(scalar);
  }
}

// Non-recursive function to look up bounds if they have been recorded
// already. For NamedScalars, also look in parallel dimension map. Finally,
// try and evaluate. If all this fails, return nullopt.
std::optional<BoundedInt> ScalarBoundsCalculator::maybeGetBounds(Val* val) {
  if (auto it = bounds_.find(val); it != bounds_.end()) {
    return it->second;
  } else if (auto* scalar = dynamic_cast<NamedScalar*>(val)) {
    boundNamedScalar(scalar);
    return bounds_.at(val);
  } else if (PolymorphicValue pv = expr_eval_.evaluate(val, known_scalars_);
             pv.hasValue()) {
    setBounds(val, pv.as<int64_t>(), pv.as<int64_t>());
    return bounds_.at(val);
  } else {
    return std::nullopt;
  }
}

void ScalarBoundsCalculator::dispatch(Statement* stmt) {
  kir::IrVisitor::dispatch(stmt);
}
void ScalarBoundsCalculator::dispatch(Val* val) {
  if (val->isIntegralScalar() && val->definition() != nullptr) {
    // This will kick off recursive dispatch
    dispatch(val->definition());
  }
  kir::IrVisitor::dispatch(val);
}

void ScalarBoundsCalculator::dispatch(Expr* expr) {
  if (auto* uop = dynamic_cast<UnaryOp*>(expr)) {
    if (uop->getUnaryOpType() == UnaryOpType::ToUnsignedSmemAddr) {
      // This is a workaround for a limitation in being able to evaluate
      // metadata for tensors with swizzles.
      // TODO: is there a better workaround?
      int64_t max_smem_addr =
          (int64_t)at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock -
          1L;
      known_scalars_[uop->out()] = max_smem_addr;
      setBounds(uop->out(), 0L, max_smem_addr);
      return;
    }
    if (uop->getUnaryOpType() == UnaryOpType::Cast &&
        uop->in()->dtype() == DataType::Index &&
        uop->out()->isIntegralScalar()) {
      // Collect casts _from_ Index scalars, so that we can check that these are
      // safe.
      casts_from_index_.push_back(uop);
    }
  }

  if (!expr->isA<ForLoop>() &&
      std::all_of(
          expr->outputs().begin(), expr->outputs().end(), [](Val* outp) {
            return !outp->isIntegralScalar();
          })) {
    // We don't need to process expressions that do not produce integers.
    // Note that for loops do "produce" their index variables for our
    // purposes.
    // It is possible that the expression outputs are constant scalars, so
    // try and compute them here.
    for (Val* outp : expr->outputs()) {
      if (outp->isIntegralScalar()) {
        PolymorphicValue pv = expr_eval_.evaluate(outp, known_scalars_);
        if (pv.hasValue()) {
          setBounds(outp, pv.as<int64_t>(), pv.as<int64_t>());
        }
      }
    }
    return;
  }
  // Inline scalar expressions might not have their inputs processed yet
  // The following loop ensures that all inputs to expr have recorded bounds.
  std::vector<Val*> immediate_inputs = expr->inputs();
  if (auto* loop = dynamic_cast<ForLoop*>(expr)) {
    immediate_inputs.push_back(loop->start());
    immediate_inputs.push_back(loop->stop());
    immediate_inputs.push_back(loop->step());
  }
  for (Val* inp : immediate_inputs) {
    if (!inp->isIntegralScalar()) {
      continue;
    }
    std::optional<BoundedInt> inp_bounds = maybeGetBounds(inp);
    if (!inp_bounds.has_value()) {
      // If inp is not constant, then we can try bounding its inputs, if
      // they are int scalars. If it has no producers that are int scalars,
      // and it is unbound, then we cannot provide any bounds for it.
      if (Expr* def = inp->definition(); def &&
          std::any_of(def->inputs().begin(),
                      def->inputs().end(),
                      [](Val* definp) { return definp->isIntegralScalar(); })) {
        // Recursively dispatch definitions
        dispatch(def);
      } else {
        setAsUnbounded(inp);
      }
    }
  }
  kir::IrVisitor::dispatch(expr);
}

int64_t ScalarBoundsCalculator::evalInt(Val* val) {
  return expr_eval_.evaluate(val).as<int64_t>();
}

void ScalarBoundsCalculator::handle(ForLoop* loop) {
  // Set bounds for the loop variable
  BoundedInt start = bounds_.at(loop->start());
  BoundedInt stop = bounds_.at(loop->stop());
  setBounds(loop->index(), start.min, stop.max - 1L);
  kir::IrVisitor::handle(loop);
}

void ScalarBoundsCalculator::handle(LoadStoreOp* lsop) {
  setBounds(lsop->out(), bounds_.at(lsop->in()));
}

void ScalarBoundsCalculator::handle(UnaryOp* uop) {
  BoundedInt a = bounds_.at(uop->in());
  BoundedInt result;
  switch (uop->getUnaryOpType()) {
    case UnaryOpType::Abs:
      result = {
          std::min(std::abs(a.min), std::abs(a.max)),
          std::max(std::abs(a.min), std::abs(a.max))};
      break;
    case UnaryOpType::BitwiseNot:
      result = ~a;
      break;
    case UnaryOpType::Cast:
      // This assumes there is no loss or overflow, since those should not
      // occur in our kernels. We can check that later for index types using
      // castsFromIndexAreSafe().
      result = a;
      break;
    case UnaryOpType::Neg:
      result = {-a.max, -a.min};
      break;
    default:
      NVF_THROW(
          "Propagation of integer bounds is not yet implemented for ",
          uop->toString());
  }
  setBounds(uop->out(), result);
}

void ScalarBoundsCalculator::handle(BinaryOp* bop) {
  BoundedInt a = bounds_.at(bop->lhs());
  BoundedInt b = bounds_.at(bop->rhs());
  BoundedInt result;
  switch (bop->getBinaryOpType()) {
    case BinaryOpType::Add:
      result = a + b;
      break;
    case BinaryOpType::BitwiseAnd:
      result = a & b;
      break;
    case BinaryOpType::BitwiseOr:
      result = a | b;
      break;
    case BinaryOpType::BitwiseXor:
      result = a ^ b;
      break;
    case BinaryOpType::CeilDiv:
      result = ceilDiv(a, b);
      break;
    case BinaryOpType::Div:
      result = a / b;
      break;
    case BinaryOpType::Mod:
      result = a % b;
      break;
    case BinaryOpType::Mul:
      result = a * b;
      break;
    case BinaryOpType::Lshift:
      result = a << b;
      break;
    case BinaryOpType::Rshift:
      result = a >> b;
      break;
    case BinaryOpType::Sub:
      result = a - b;
      break;
    default:
      NVF_THROW(
          "Propagation of integer bounds is not yet implemented for ",
          bop->toString());
  }
  setBounds(bop->out(), result);
}

void ScalarBoundsCalculator::handle(TernaryOp* top) {
  switch (top->getTernaryOpType()) {
    default:
      NVF_THROW(
          "Propagation of integer bounds is not yet implemented for ",
          top->toString());
  }
}

ScalarBoundsCalculator::~ScalarBoundsCalculator() {}

// TODO: Use this to set index type
PrimDataType getSmallestIndexTypeByBoundingExpressions(
    kir::Kernel* kernel,
    ExpressionEvaluator& expr_eval,
    const LaunchParams& launch_params) {
  // bind args to expression evaluator
  ScalarBoundsCalculator calc(kernel, expr_eval, launch_params);
  // Compute the range of all nvfuser_index_t values in the fusion
  BoundedInt index_bounds = calc.boundByDataType();
  return (index_bounds.min < (int64_t)std::numeric_limits<int32_t>::min() ||
          index_bounds.max > (int64_t)std::numeric_limits<int32_t>::max())
      ? PrimDataType::Int
      : PrimDataType::Int32;
}

} // namespace nvfuser

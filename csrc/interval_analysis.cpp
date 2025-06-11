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

BoundedInt BoundedInt::operator-() const {
  return BoundedInt{-max, -min};
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

// Division ranges are computed differently based on whether the numerator and
// denominator are positive or negative. Because of this, we split the numerator
// into a negative and a non-positive range and we split the denominator into a
// negative and a positive range. Then we compute the bounds for every non-empty
// combination of ranges, of which there are at most four. The final bound is
// the union of those intervals.
//
// For example, if we have -2 <= a <= 1 and -1 <= b <= 2 and we want to compute
// bounds for a / b, we have the following cases to handle
//
//   -2 / -1 =  2
//   -1 / -1 =  1
//    0 / -1 =  0
//    1 / -1 = -1
//   -2 /  0 = ERROR
//   -1 /  0 = ERROR
//    0 /  0 = ERROR
//    1 /  0 = ERROR
//   -2 /  1 = -2
//   -1 /  1 = -1
//    0 /  1 =  0
//    1 /  1 =  1
//   -2 /  2 = -1
//   -1 /  2 =  0
//    0 /  2 =  0
//    1 /  2 =  0
//
// We split a into intervals -2 <= a <= -1 and 0 <= a <= 1 which includes zero.
// We split b, on the other hand, into -1 <= b <= -1 and 1 <= b <= 2, excluding
// the error cases. Then for all four combinations we compute a single interval
// before computing the union of those four intervals.
//
//   -2 / -1 =  2
//   -1 / -1 =  1     =>    [1, 2]
//
//   -2 /  1 = -2
//   -1 /  1 = -1
//   -2 /  2 = -1
//   -1 /  2 =  0     =>    [-2, 0]
//
//    0 / -1 =  0
//    1 / -1 = -1     =>    [-1, 0]
//
//    0 /  1 =  0
//    1 /  1 =  1
//    0 /  2 =  0
//    1 /  2 =  0     =>    [0, 1]
//
// The result we return in this case is the union of these four intervals which
// is [-2, 2]
#define DEFINE_DIVISION_LIKE_OP(a, b, pospos, posneg, negpos, negneg)      \
  NVF_ERROR(                                                               \
      b.min != 0L || b.max != 0L,                                          \
      "Found denominator that cannot be non-zero: ",                       \
      b);                                                                  \
  const auto split_ranges_around_zero = [](const BoundedInt& b,            \
                                           bool include_zero) {            \
    std::vector<BoundedInt> ranges;                                        \
    if (b.min < 0L) {                                                      \
      ranges.push_back({b.min, std::min(b.max, -1L)});                     \
    }                                                                      \
    int64_t min_nonneg_val = include_zero ? 0L : 1L;                       \
    if (b.max >= min_nonneg_val) {                                         \
      ranges.push_back({std::max(b.min, min_nonneg_val), b.max});          \
    }                                                                      \
    return ranges;                                                         \
  };                                                                       \
  const std::vector<BoundedInt> numer_ranges =                             \
      split_ranges_around_zero(a, /*include_zero=*/true);                  \
  const std::vector<BoundedInt> denom_ranges =                             \
      split_ranges_around_zero(b, /*include_zero=*/false);                 \
                                                                           \
  BoundedInt result;                                                       \
  bool first = true;                                                       \
  for (const BoundedInt& numer : numer_ranges) {                           \
    for (const BoundedInt& denom : denom_ranges) {                         \
      BoundedInt simple_range;                                             \
      /* numer and denom are each either only negative or only positive */ \
      if (numer.min >= 0) {                                                \
        if (denom.min > 0) {                                               \
          simple_range = pospos(numer, denom);                             \
        } else {                                                           \
          simple_range = posneg(numer, denom);                             \
        }                                                                  \
      } else {                                                             \
        if (denom.min > 0) {                                               \
          simple_range = negpos(numer, denom);                             \
        } else {                                                           \
          simple_range = negneg(numer, denom);                             \
        }                                                                  \
      }                                                                    \
      /* Result is the union over all of the simple ranges */              \
      if (first) {                                                         \
        result = simple_range;                                             \
      } else {                                                             \
        result.min = std::min(result.min, simple_range.min);               \
        result.max = std::max(result.max, simple_range.max);               \
      }                                                                    \
      first = false;                                                       \
    }                                                                      \
  }                                                                        \
  return result;
BoundedInt BoundedInt::operator/(const BoundedInt& other) const {
  // positive over positive
  const auto pospos = [](const BoundedInt& numer, const BoundedInt& denom) {
    return BoundedInt{numer.min / denom.max, numer.max / denom.min};
  };
  // positive over negative
  const auto posneg = [](const BoundedInt& numer, const BoundedInt& denom) {
    return BoundedInt{numer.max / denom.max, numer.min / denom.min};
  };
  // negative over positive
  const auto negpos = [](const BoundedInt& numer, const BoundedInt& denom) {
    return BoundedInt{numer.min / denom.min, numer.max / denom.max};
  };
  // negative over negative
  const auto negneg = [](const BoundedInt& numer, const BoundedInt& denom) {
    return BoundedInt{numer.max / denom.min, numer.min / denom.max};
  };
  DEFINE_DIVISION_LIKE_OP(*this, other, pospos, posneg, negpos, negneg);
}

BoundedInt ceilDiv(const BoundedInt& a, const BoundedInt& b) {
  // positive over positive
  const auto pospos = [](const BoundedInt& numer, const BoundedInt& denom) {
    return BoundedInt{
        ceilDiv(numer.min, denom.max), ceilDiv(numer.max, denom.min)};
  };
  // positive over negative
  const auto posneg = [](const BoundedInt& numer, const BoundedInt& denom) {
    return BoundedInt{
        ceilDiv(numer.max, denom.max), ceilDiv(numer.min, denom.min)};
  };
  // negative over positive
  const auto negpos = [](const BoundedInt& numer, const BoundedInt& denom) {
    return BoundedInt{
        ceilDiv(numer.min, denom.min), ceilDiv(numer.max, denom.max)};
  };
  // negative over negative
  const auto negneg = [](const BoundedInt& numer, const BoundedInt& denom) {
    return BoundedInt{
        ceilDiv(numer.max, denom.min), ceilDiv(numer.min, denom.max)};
  };
  DEFINE_DIVISION_LIKE_OP(a, b, pospos, posneg, negpos, negneg);
}

// Modulo is the remainder op and satisfies
//
//   a % b = a - (a / b) * b
//
// for any a and b. Since division in C++ is truncdiv and rounds toward zero,
// (a / b) * b is the same as (a / (-b)) * (-b) and never maps a negative value
// to a more negative value. This means the remainder is negative when a is
// negative and non-negative when a is non-negative.
//
// Note that like for division, we ignore b==0. Additionally, if we can
// guarantee 0 <= a < b then a % b = a so we can just use a's bounds. This
// is also the case if b < a < 0.
BoundedInt BoundedInt::operator%(const BoundedInt& other) const {
  // positive mod positive
  const auto pospos = [](const BoundedInt& numer, const BoundedInt& denom) {
    if (numer.max < denom.min) {
      // mod op is trivial
      return numer;
    } else {
      return BoundedInt{0L, denom.max - 1L};
    }
  };
  // positive mod negative
  const auto posneg = [&pospos](
                          const BoundedInt& numer, const BoundedInt& denom) {
    return pospos(numer, -denom);
  };
  // negative mod positive
  const auto negpos = [](const BoundedInt& numer, const BoundedInt& denom) {
    if (numer.min > -denom.min) {
      // mod op is trivial
      return numer;
    } else {
      return BoundedInt{1L - denom.max, 0};
    }
    return BoundedInt{
        ceilDiv(numer.min, denom.min), ceilDiv(numer.max, denom.max)};
  };
  // negative mod negative
  const auto negneg = [&negpos](
                          const BoundedInt& numer, const BoundedInt& denom) {
    return negpos(numer, -denom);
  };
  DEFINE_DIVISION_LIKE_OP(*this, other, pospos, posneg, negpos, negneg);
}
#undef DEFINE_DIVISION_LIKE_OP

BoundedInt BoundedInt::operator/(const int64_t other) const {
  return *this / BoundedInt{other, other};
}

BoundedInt BoundedInt::operator%(const int64_t other) const {
  return *this % BoundedInt{other, other};
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
#define DEFINE_BITWISE_BINARY_OP(op)                                        \
  BoundedInt BoundedInt::operator op(const BoundedInt & other) const {      \
    /* New interval has this many fixed bits */                             \
    int64_t var_bits =                                                      \
        64L - std::min(countCommonHighBits(), other.countCommonHighBits()); \
    /* Mask everything below the higher fixed_bits */                       \
    int64_t low_mask = (1 << var_bits) - 1;                                 \
    int64_t new_min = (min op other.min) & (~low_mask);                     \
    int64_t new_max = new_min + low_mask;                                   \
    return {new_min, new_max};                                              \
  }
DEFINE_BITWISE_BINARY_OP(&)
DEFINE_BITWISE_BINARY_OP(|)
DEFINE_BITWISE_BINARY_OP(^)
#undef DEFINE_BITWISE_BINARY_OP

BoundedInt BoundedInt::operator~() const {
  // New interval has this many fixed bits
  int64_t var_bits = 64L - countCommonHighBits();
  // Mask everything below the higher fixed_bits
  int64_t low_mask = (1 << var_bits) - 1; // 0b00111
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
  int64_t new_max = (max < 0L) ? (max >> other.max) : (max >> other.min);
  return {new_min, new_max};
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

void ScalarBoundsCalculator::setBoundsForNamedScalar(NamedScalar* scalar) {
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
    setBoundsForNamedScalar(scalar);
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
  if (lsop->in()->isIntegralScalar()) {
    setBounds(lsop->out(), bounds_.at(lsop->in()));
  }
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

} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <visibility.h>

#include <expr_evaluator.h>
#include <ir/internal_nodes.h>
#include <kernel.h>
#include <kernel_ir_dispatch.h>
#include <polymorphic_value.h>
#include <runtime/executor_params.h>
#include <type.h>

#include <optional>
#include <unordered_map>
#include <vector>

namespace nvfuser {

//! This holds inclusive bounds for a particular integer Val. We will propagate
//! one of these for each integer scalar in the lowered kernel. That propagation
//! makes use of the operators defined in this class.
//!
//! Note that this class does not necessarily represent tight bounds on
//! complicated expressions. For example:
//!
//!   for a in iS0{n}
//!     b = a * 2
//!     c = b % 8
//!
//! In our analysis, we will define the following ranges as BoundedInt values:
//!
//!   a \in [0, n-1]
//!   b \in [0, (n-1) * 2]
//!   c \in [0, 7]  (assuming 7 is not in the range of n)
//!
//! These bounds are correct even though we could use a tighter bound for c of
//! [0, 6] since we know that b must be a multiple of 2, so c must be 0, 2, 4,
//! or 6 only. This kind of analysis is not provided by the simplistic
//! propagation using a BoundedInt interval at each stage.
struct NVF_API BoundedInt {
  int64_t min;
  int64_t max;

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
  NVF_API int64_t countCommonHighBits() const;

  NVF_API BoundedInt operator+(const BoundedInt& other) const;
  NVF_API BoundedInt operator+(const int64_t other) const;
  NVF_API BoundedInt operator-(const BoundedInt& other) const;
  NVF_API BoundedInt operator-(const int64_t other) const;
  NVF_API BoundedInt operator*(const BoundedInt& other) const;
  NVF_API BoundedInt operator*(const int64_t other) const;
  NVF_API BoundedInt operator/(const BoundedInt& other) const;
  NVF_API BoundedInt operator/(const int64_t other) const;
  NVF_API BoundedInt operator%(const BoundedInt& other) const;
  NVF_API BoundedInt operator%(const int64_t other) const;

  NVF_API BoundedInt operator^(const BoundedInt& other) const;
  NVF_API BoundedInt operator&(const BoundedInt& other) const;
  NVF_API BoundedInt operator|(const BoundedInt& other) const;
  NVF_API BoundedInt operator~() const;
  NVF_API BoundedInt operator>>(const BoundedInt& other) const;
  NVF_API BoundedInt operator<<(const BoundedInt& other) const;

  NVF_API bool operator==(const BoundedInt& other) const {
    return min == other.min && max == other.max;
  }
  NVF_API bool operator!=(const BoundedInt& other) const {
    return !(*this == other);
  }
};

NVF_API std::ostream& operator<<(std::ostream& out, const BoundedInt& b);

//! This class traverses the expressions in a kir::Kernel and defines a
//! BoundedInt for each integer scalar encountered. The range is determined by
//! the scalar's definition along with the rules defined in BoundedInt.
class NVF_API ScalarBoundsCalculator : kir::IrVisitor {
 public:
  NVF_API ScalarBoundsCalculator(
      kir::Kernel* kernel,
      ExpressionEvaluator& expr_eval,
      const LaunchParams& launch_params);

  NVF_API virtual ~ScalarBoundsCalculator();

  //! Return the bounds, computed over all scalars in the fusion with the given
  //! data type
  NVF_API BoundedInt boundByDataType(DataType dtype = DataType::Index);

  //! Look at all casts (T)x where x is of type nvfuser_index_t, to ensure that
  //! these casts are safe i.e. that the bounds of x do not overflow those
  //! representable by T.
  NVF_API bool castsFromIndexAreSafe() const;

  //! NamedScalar bounds are set using the launch_params_. For example
  //! `blockDim.x` is set to the [blockDim.x, blockDim.x] and `threadIdx.x` is
  //! set to [0, blockDim.x - 1].
  NVF_API void boundNamedScalar(NamedScalar* scalar);

  using kir::IrVisitor::dispatch;
  //! These public methods are useful for processing an individual statement to
  //! get bounds of all its producers
  NVF_API void dispatch(Statement* statement) final;
  NVF_API void dispatch(Val* val) final;

  NVF_API void setBounds(Val* val, const BoundedInt& bounds);
  NVF_API void setBounds(Val* val, int64_t min, int64_t max);
  NVF_API void setAsUnbounded(Val* val);

  //! Non-recursive function to look up bounds if they have been recorded
  //! already. For NamedScalars, also look in parallel dimension map and bound
  //! if it has not already been bounded. Finally, try and evaluate constants.
  //! If all this fails, return nullopt.
  NVF_API std::optional<BoundedInt> maybeGetBounds(Val* val);

 private:
  NVF_API void dispatch(Expr* expr) final;

  //! Evaluate val using our ExpressionEvaluator
  int64_t evalInt(Val* val);

  using kir::IrVisitor::handle;

  NVF_API void handle(ForLoop* loop) final;
  NVF_API void handle(LoadStoreOp* lsop) final;
  NVF_API void handle(UnaryOp* uop) final;
  NVF_API void handle(BinaryOp* bop) final;
  NVF_API void handle(TernaryOp* top) final;

 private:
  ExpressionEvaluator& expr_eval_;
  const LaunchParams& launch_params_;
  std::unordered_map<const Val*, BoundedInt> bounds_;
  std::unordered_map<const Val*, PolymorphicValue> known_scalars_;
  std::vector<UnaryOp*> casts_from_index_;
};

//! This function uses ScalarBoundsCalculator to determine the union of all the
//! ranges of DataType::Index scalars in the fusion. Then it returns the
//! smallest type that can represent that range, out of PrimDataType::Int or
//! PrimDataType::Int32.
PrimDataType getSmallestIndexTypeByBoundingExpressions(
    kir::Kernel* kernel,
    ExpressionEvaluator& expr_eval,
    const LaunchParams& launch_params);

} // namespace nvfuser

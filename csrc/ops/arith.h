// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

#include <ir/base_nodes.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ops/utils.h>
#include <type.h>
#include <type_promotion.h>

/*
 * The operations defined in this header is intended as user facing functions.
 * Generally users should not directly instantiate temporary TensorViews they
 * should instead use the functions below which will automatically create IR
 * nodes, and return a resulting TensorView of correctly tracked shapes.
 */

namespace nvfuser {

// Insertion of casting op to dtype, returns new resulting val
Val* castOp(DataType dtype, Val* v1);
TensorView* castOp(DataType dtype, TensorView* v1);
// If v1 is not dtype, insert a cast op, otherwise return v1
Val* maybeCastOp(DataType dtype, Val* v1);
TensorView* maybeCastOp(DataType dtype, TensorView* v1);

Val* bitCastOp(DataType dtype, Val* v1);
TensorView* bitCastOp(DataType dtype, TensorView* v1);

// Perform unary op type and return the output
Val* unaryOp(UnaryOpType type, Val* v1);
TensorView* unaryOp(UnaryOpType type, TensorView* v1);
Val* unaryIsOp(UnaryOpType type, Val* v1);
TensorView* unaryIsOp(UnaryOpType type, TensorView* v1);
Val* unaryOp(UnaryOpType type, Val* v1, const TypePromotionConfig& config);
TensorView* unaryOp(
    UnaryOpType type,
    TensorView* v1,
    const TypePromotionConfig& config);

// Perform binary op type on v1 and v2 and return a type promoted output.
// Mod, CeilDiv, and LT are considered Int only output operations for now.
Val* binaryOp(
    BinaryOpType type,
    Val* v1,
    Val* v2,
    DataType out_dtype = DataType::Null);
TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2,
    DataType out_dtype = DataType::Null);
TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2,
    DataType out_dtype = DataType::Null);
TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2,
    DataType out_dtype = DataType::Null);

Val* binaryOp(
    BinaryOpType type,
    Val* v1,
    Val* v2,
    const TypePromotionConfig& config);
TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2,
    const TypePromotionConfig& config);
TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2,
    const TypePromotionConfig& config);
TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2,
    const TypePromotionConfig& config);

// Return a new TensorView consistent with reducing `tv` on specified `axes`
TensorView* newForReduction(
    TensorView* tv,
    const std::vector<unsigned int>& axes,
    DataType data_type = DataType::Null);

// Perform a reduction operation on v1, initial value for reduction is init,
// reduces across axes, and reduction operation defined by BinaryOp. Reduction
// of size-1 dimension is automatically converted to squeeze.
TensorView* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int64_t>& axes,
    Val* init,
    TensorView* v1,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

// Just create a ReductionOp, don't try to simplify it. Don't convert size-1
// reduction into squeeze and don't convert size-0 reduction into full.
TensorView* reductionOpRaw(
    BinaryOpType reduction_op_type,
    const std::vector<int64_t>& axes,
    Val* init,
    TensorView* v1,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

//! Auxiliary Struct holding result of
//! a single welford op in ternsorview
struct WelfordResult {
  TensorView* avg;
  TensorView* var_sum;
  TensorView* n;

  explicit WelfordResult(
      TensorView* in_avg,
      TensorView* in_var_sum,
      TensorView* in_n,
      const bool check_definition = true);
};

//! Auxiliary struct holding the result of a topk operation.
//!
//! Contains two TensorViews:
//! - values: tensor containing the k largest/smallest values
//! - indices: tensor containing the indices of those values in the original
//! tensor
//!
//! Both tensors have the same shape as the input tensor, except the dimension
//! along which topk was performed has size k.
struct TopKResult {
 public:
  TensorView* values = nullptr; //!< The k largest/smallest values
  TensorView* indices; //!< Indices of the values in the original tensor

  //! Constructor ensuring both outputs come from the same TopK operation
  explicit TopKResult(TensorView* in_values, TensorView* in_indices);
};

//! Welford operator on specified axes. This is currently the only scan op with
//! multiple outputs that is supported. May consider generalization if more scan
//! ops are added.
WelfordResult Welford(
    TensorView* tv,
    const std::vector<int64_t>& axes,
    TensorView* init_avg = nullptr,
    TensorView* init_var = nullptr,
    // Initializes to 0 in function definition, doing this so we don't have to
    // import IrBuilder just for this one interface.
    Val* init_N = nullptr);

//! Create a raw WelfordOp. Don't convert size-1 or size-0 reduction into
//! squeeze/full.
WelfordResult WelfordRaw(
    TensorView* tv,
    const std::vector<int64_t>& axes,
    TensorView* init_avg = nullptr,
    TensorView* init_var = nullptr,
    // Initializes to 0 in function definition, doing this so we don't have to
    // import IrBuilder just for this one interface.
    Val* init_N = nullptr);

// RNG OPERATIONS
TensorView* rand(
    const std::vector<Val*>& shape,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr,
    bool maybe_symbolic = true);
TensorView* rand_like(TensorView*, Val* philox_seed, Val* philox_offset);
// Note that overloading these would be convenient, but overloaded functions are
// difficult to cast correctly. In the serde method
// RecordFunctorFactory::setupFunctionMaps(), the op is cast to, for example
// nvfuser::Val* (*)(nvfuser::Val*). In order to avoid errors due to that
// static_cast, we just implement the unary and ternary versions of the random
// *_like operators as separate functions.
Val* rand_like(Val*, Val* philox_seed, Val* philox_offset);
TensorView* rand_like(TensorView* tv);
Val* rand_like(Val* val);

TensorView* randn(
    const std::vector<Val*>& shape,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr,
    bool maybe_symbolic = true);
TensorView* randn_like(TensorView*, Val* philox_seed, Val* philox_offset);
Val* randn_like(Val*, Val* philox_seed, Val* philox_offset);
TensorView* randn_like(TensorView* tv);
Val* randn_like(Val* val);

TensorView* uniform(
    const std::vector<Val*>& shape,
    Val* low,
    Val* high,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr,
    bool maybe_symbolic = true);
TensorView* normal(
    const std::vector<Val*>& shape,
    Val* mean,
    Val* std,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr,
    bool maybe_symbolic = true);

// TENSOR FACTORIES
TensorView* full(
    const std::vector<Val*>& shape,
    Val* fill_value,
    DataType dtype,
    bool maybe_symbolic = true);
TensorView* full_like(TensorView* tv, Val* fill_value, DataType dtype);
TensorView* full_like(TensorView* tv, Val* fill_value);
Val* full_like(Val* tv, Val* fill_value);
TensorView* zeros(
    const std::vector<Val*>& shape,
    DataType dtype,
    bool maybe_symbolic = true);
TensorView* zeros_like(TensorView*);
Val* zeros_like(Val*);
TensorView* ones(
    const std::vector<Val*>& shape,
    DataType dtype,
    bool maybe_symbolic = true);
TensorView* ones_like(TensorView*);
Val* ones_like(Val*);
TensorView* iota(
    Val* length,
    Val* start = nullptr,
    Val* step = nullptr,
    DataType dtype = DataType::Int);
//! WARNING: giving invalid combinations of the start, end and step
//! arguments can result in undefined behavior. Specifically, the
//! signs of `end - start` and step must be the same.
TensorView* arange(Val* end, DataType dtype = DataType::Int);
TensorView* arange(Val* start, Val* end, DataType dtype = DataType::Int);
TensorView* arange(
    Val* start,
    Val* end,
    Val* step,
    DataType dtype = DataType::Int);
TensorView* eye(Val* size, DataType dtype);
TensorView* eye(Val* rows, Val* cols, DataType dtype);

// UNARY OPERATIONS
// abs
Val* abs(Val*);
TensorView* abs(TensorView*);
// acos
Val* acos(Val*);
TensorView* acos(TensorView*);
// acosh
Val* acosh(Val*);
TensorView* acosh(TensorView*);
// asin
Val* asin(Val*);
TensorView* asin(TensorView*);
// asinh
Val* asinh(Val*);
TensorView* asinh(TensorView*);
// atan
Val* atan(Val*);
TensorView* atan(TensorView*);
// atanh
Val* atanh(Val*);
TensorView* atanh(TensorView*);
// ceil
Val* ceil(Val*);
TensorView* ceil(TensorView*);
// cos
Val* cos(Val*);
TensorView* cos(TensorView*);
// cosh
Val* cosh(Val*);
TensorView* cosh(TensorView*);
// exp
Val* exp(Val*);
TensorView* exp(TensorView*);
// exp2
Val* exp2(Val*);
TensorView* exp2(TensorView*);
// expm1
Val* expm1(Val*);
TensorView* expm1(TensorView*);
// erf
Val* erf(Val*);
TensorView* erf(TensorView*);
// erfc
Val* erfc(Val*);
TensorView* erfc(TensorView*);
// erfinv
Val* erfinv(Val*);
TensorView* erfinv(TensorView*);
// erfcinv
Val* erfcinv(Val*);
TensorView* erfcinv(TensorView*);
// floor
Val* floor(Val*);
TensorView* floor(TensorView*);
// frac
Val* frac(Val*);
TensorView* frac(TensorView*);
// silu
Val* silu(Val*);
TensorView* silu(TensorView*);
// lgamma
Val* lgamma(Val*);
TensorView* lgamma(TensorView*);
// log
Val* log(Val*);
TensorView* log(TensorView*);
// log10
Val* log10(Val*);
TensorView* log10(TensorView*);
// log1p
Val* log1p(Val*);
TensorView* log1p(TensorView*);
// log2
Val* log2(Val*);
TensorView* log2(TensorView*);
// neg
Val* neg(Val*);
TensorView* neg(TensorView*);
// logical_not
Val* logical_not(Val*);
TensorView* logical_not(TensorView*);
// bitwise_not
Val* bitwise_not(Val*);
TensorView* bitwise_not(TensorView*);
// real
Val* real(Val*);
TensorView* real(TensorView*);
// reciprocal
Val* reciprocal(Val*);
TensorView* reciprocal(TensorView*);
// relu
Val* relu(Val*);
TensorView* relu(TensorView*);
// rsqrt
Val* rsqrt(Val*);
TensorView* rsqrt(TensorView*);
// round
Val* round(Val*);
TensorView* round(TensorView*);
// sigmoid
Val* sigmoid(Val*);
TensorView* sigmoid(TensorView*);
// signbit
Val* signbit(Val*);
TensorView* signbit(TensorView*);
// sin
Val* sin(Val*);
TensorView* sin(TensorView*);
// sinh
Val* sinh(Val*);
TensorView* sinh(TensorView*);
// sqrt
Val* sqrt(Val*);
TensorView* sqrt(TensorView*);
// tan
Val* tan(Val*);
TensorView* tan(TensorView*);
// tanh
Val* tanh(Val*);
TensorView* tanh(TensorView*);
// trunc
Val* trunc(Val*);
TensorView* trunc(TensorView*);
// bitwise_not
Val* bitwise_not(Val*);
TensorView* bitwise_not(TensorView*);
// bitceil
Val* bitceil(Val*);
TensorView* bitceil(TensorView*);
// imag
Val* imag(Val*);
TensorView* imag(TensorView*);
// isfinite
Val* isfinite(Val*);
TensorView* isfinite(TensorView*);
// isinf
Val* isinf(Val*);
TensorView* isinf(TensorView*);
// isnan
Val* isnan(Val*);
TensorView* isnan(TensorView*);
// isneginf
Val* isneginf(Val*);
TensorView* isneginf(TensorView*);
// isposinf
Val* isposinf(Val*);
TensorView* isposinf(TensorView*);
// isreal
Val* isreal(Val*);
TensorView* isreal(TensorView*);
// print
Val* print(Val*);
TensorView* print(TensorView*);

// This is a function used to give the symbolic shape of a tensor for use
// with functions like broadcast_in_dim that take a shape vector
// to use to expand an input tensor
std::vector<Val*> shape(TensorView* inp);
// Get the symbolic size of a specific dimension of a tensor
Val* size(TensorView* inp, int64_t dim);
Val* at(const std::vector<Val*>& inp, int64_t index);

// BINARY OPERATIONS
// add
Val* add(Val* v1, Val* v2);
TensorView* add(TensorView* v1, Val* v2);
TensorView* add(Val* v1, TensorView* v2);
TensorView* add(TensorView* v1, TensorView* v2);
// atan2
Val* atan2(Val* v1, Val* v2);
TensorView* atan2(TensorView* v1, Val* v2);
TensorView* atan2(Val* v1, TensorView* v2);
TensorView* atan2(TensorView* v1, TensorView* v2);
// truediv: promote to float for integer division, has the same semantics as the
// python's operator /
Val* truediv(Val* v1, Val* v2);
TensorView* truediv(TensorView* v1, Val* v2);
TensorView* truediv(Val* v1, TensorView* v2);
TensorView* truediv(TensorView* v1, TensorView* v2);
// div: don't promote to float, instead, truncate the result, this has the same
// semantics as the C++'s operator /
Val* div(Val* v1, Val* v2);
TensorView* div(TensorView* v1, Val* v2);
TensorView* div(Val* v1, TensorView* v2);
TensorView* div(TensorView* v1, TensorView* v2);
// fmod
Val* fmod(Val* v1, Val* v2);
TensorView* fmod(TensorView* v1, Val* v2);
TensorView* fmod(Val* v1, TensorView* v2);
TensorView* fmod(TensorView* v1, TensorView* v2);
// mul
Val* mul(Val* v1, Val* v2);
TensorView* mul(TensorView* v1, Val* v2);
TensorView* mul(Val* v1, TensorView* v2);
TensorView* mul(TensorView* v1, TensorView* v2);
// pow
Val* pow(Val* v1, Val* v2);
TensorView* pow(TensorView* v1, Val* v2);
TensorView* pow(Val* v1, TensorView* v2);
TensorView* pow(TensorView* v1, TensorView* v2);
// remainder
Val* remainder(Val* v1, Val* v2);
TensorView* remainder(TensorView* v1, Val* v2);
TensorView* remainder(Val* v1, TensorView* v2);
TensorView* remainder(TensorView* v1, TensorView* v2);
// sub
Val* sub(Val* v1, Val* v2);
TensorView* sub(TensorView* v1, Val* v2);
TensorView* sub(Val* v1, TensorView* v2);
TensorView* sub(TensorView* v1, TensorView* v2);
// maximum
Val* maximum(Val* v1, Val* v2);
TensorView* maximum(TensorView* v1, Val* v2);
TensorView* maximum(Val* v1, TensorView* v2);
TensorView* maximum(TensorView* v1, TensorView* v2);
// minimum
Val* minimum(Val* v1, Val* v2);
TensorView* minimum(TensorView* v1, Val* v2);
TensorView* minimum(Val* v1, TensorView* v2);
TensorView* minimum(TensorView* v1, TensorView* v2);
// nextafter: Only single- or double-precision
// floating point types (after promotion) are supported.
Val* nextafter(Val* v1, Val* v2);
TensorView* nextafter(TensorView* v1, Val* v2);
TensorView* nextafter(Val* v1, TensorView* v2);
TensorView* nextafter(TensorView* v1, TensorView* v2);
// Integer binary ops
// mod
Val* mod(Val* v1, Val* v2);
TensorView* mod(TensorView* v1, Val* v2);
TensorView* mod(Val* v1, TensorView* v2);
TensorView* mod(TensorView* v1, TensorView* v2);
// ceilDiv
Val* ceilDiv(Val* v1, Val* v2);
TensorView* ceilDiv(TensorView* v1, Val* v2);
TensorView* ceilDiv(Val* v1, TensorView* v2);
TensorView* ceilDiv(TensorView* v1, TensorView* v2);
// Bitwise and logical binary ops
// bitwise_and
Val* bitwise_and(Val* v1, Val* v2);
TensorView* bitwise_and(TensorView* v1, Val* v2);
TensorView* bitwise_and(Val* v1, TensorView* v2);
TensorView* bitwise_and(TensorView* v1, TensorView* v2);
// logical_and
Val* logical_and(Val* v1, Val* v2);
TensorView* logical_and(TensorView* v1, Val* v2);
TensorView* logical_and(Val* v1, TensorView* v2);
TensorView* logical_and(TensorView* v1, TensorView* v2);
// bitwise_left_shift
Val* bitwise_left_shift(Val* v1, Val* v2);
TensorView* bitwise_left_shift(TensorView* v1, Val* v2);
TensorView* bitwise_left_shift(Val* v1, TensorView* v2);
TensorView* bitwise_left_shift(TensorView* v1, TensorView* v2);
// bitwise_right_shift
Val* bitwise_right_shift(Val* v1, Val* v2);
TensorView* bitwise_right_shift(TensorView* v1, Val* v2);
TensorView* bitwise_right_shift(Val* v1, TensorView* v2);
TensorView* bitwise_right_shift(TensorView* v1, TensorView* v2);
// logical_right_shift
TensorView* logical_right_shift(TensorView* x, TensorView* shift);
TensorView* logical_right_shift(TensorView* x, Val* shift);
TensorView* logical_right_shift(Val* x, TensorView* shift);
Val* logical_right_shift(Val* x, Val* shift);
// bitwise_or
Val* bitwise_or(Val* v1, Val* v2);
TensorView* bitwise_or(TensorView* v1, Val* v2);
TensorView* bitwise_or(Val* v1, TensorView* v2);
TensorView* bitwise_or(TensorView* v1, TensorView* v2);
// logical_or
Val* logical_or(Val* v1, Val* v2);
TensorView* logical_or(TensorView* v1, Val* v2);
TensorView* logical_or(Val* v1, TensorView* v2);
TensorView* logical_or(TensorView* v1, TensorView* v2);
// bitwise_xor
Val* bitwise_xor(Val* v1, Val* v2);
TensorView* bitwise_xor(TensorView* v1, Val* v2);
TensorView* bitwise_xor(Val* v1, TensorView* v2);
TensorView* bitwise_xor(TensorView* v1, TensorView* v2);
// gcd
Val* gcd(Val* v1, Val* v2);
TensorView* gcd(TensorView* v1, Val* v2);
TensorView* gcd(Val* v1, TensorView* v2);
TensorView* gcd(TensorView* v1, TensorView* v2);
// Logical binary ops
// eq
Val* eq(Val* v1, Val* v2);
TensorView* eq(TensorView* v1, Val* v2);
TensorView* eq(Val* v1, TensorView* v2);
TensorView* eq(TensorView* v1, TensorView* v2);
// ge
Val* ge(Val* v1, Val* v2);
TensorView* ge(TensorView* v1, Val* v2);
TensorView* ge(Val* v1, TensorView* v2);
TensorView* ge(TensorView* v1, TensorView* v2);
// gt
Val* gt(Val* v1, Val* v2);
TensorView* gt(TensorView* v1, Val* v2);
TensorView* gt(Val* v1, TensorView* v2);
TensorView* gt(TensorView* v1, TensorView* v2);
// le
Val* le(Val* v1, Val* v2);
TensorView* le(TensorView* v1, Val* v2);
TensorView* le(Val* v1, TensorView* v2);
TensorView* le(TensorView* v1, TensorView* v2);
// lt
Val* lt(Val* v1, Val* v2);
TensorView* lt(TensorView* v1, Val* v2);
TensorView* lt(Val* v1, TensorView* v2);
TensorView* lt(TensorView* v1, TensorView* v2);
// ne
Val* ne(Val* v1, Val* v2);
TensorView* ne(TensorView* v1, Val* v2);
TensorView* ne(Val* v1, TensorView* v2);
TensorView* ne(TensorView* v1, TensorView* v2);

// complex
Val* complex(Val* v1, Val* v2);
TensorView* complex(TensorView* v1, Val* v2);
TensorView* complex(Val* v1, TensorView* v2);
TensorView* complex(TensorView* v1, TensorView* v2);

// REDUCTION OPERATIONS
TensorView* sum(
    TensorView* v1,
    const std::vector<int64_t>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

TensorView* prod(
    TensorView* v1,
    const std::vector<int64_t>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

TensorView* max(
    TensorView* v1,
    const std::vector<int64_t>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

TensorView* min(
    TensorView* v1,
    const std::vector<int64_t>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

// COMPOUND OPERATIONS
// add_alpha
Val* add_alpha(Val* v1, Val* v2, Val* s);
TensorView* add_alpha(TensorView* v1, Val* v2, Val* s);
TensorView* add_alpha(Val* v1, TensorView* v2, Val* s);
TensorView* add_alpha(TensorView* v1, TensorView* v2, Val* s);
// sub_alpha
Val* sub_alpha(Val* v1, Val* v2, Val* s);
TensorView* sub_alpha(TensorView* v1, Val* v2, Val* s);
TensorView* sub_alpha(Val* v1, TensorView* v2, Val* s);
TensorView* sub_alpha(TensorView* v1, TensorView* v2, Val* s);
// lerp
Val* lerp(Val* start, Val* end, Val* weight);
TensorView* lerp(TensorView* start, Val* end, Val* weight);
TensorView* lerp(Val* start, TensorView* end, Val* weight);
TensorView* lerp(Val* start, Val* end, TensorView* weight);
TensorView* lerp(TensorView* start, TensorView* end, Val* weight);
TensorView* lerp(TensorView* start, Val* end, TensorView* weight);
TensorView* lerp(Val* start, TensorView* end, TensorView* weight);
TensorView* lerp(TensorView* start, TensorView* end, TensorView* weight);

// addcmul
Val* addcmul(Val* v1, Val* v2, Val* v3, Val* s);
TensorView* addcmul(TensorView* v1, Val* v2, Val* v3, Val* s);
TensorView* addcmul(Val* v1, TensorView* v2, Val* v3, Val* s);
TensorView* addcmul(Val* v1, Val* v2, TensorView* v3, Val* s);
TensorView* addcmul(TensorView* v1, TensorView* v2, Val* v3, Val* s);
TensorView* addcmul(TensorView* v1, Val* v2, TensorView* v3, Val* s);
TensorView* addcmul(Val* v1, TensorView* v2, TensorView* v3, Val* s);
TensorView* addcmul(TensorView* v1, TensorView* v2, TensorView* v3, Val* s);

// TERNARY OPERATIONS
// where
Val* where(Val* c, Val* v1, Val* v2);
TensorView* where(TensorView* c, Val* v1, Val* v2);
TensorView* where(Val* c, TensorView* v1, Val* v2);
TensorView* where(Val* c, Val* v1, TensorView* v2);
TensorView* where(TensorView* c, TensorView* v1, Val* v2);
TensorView* where(TensorView* c, Val* v1, TensorView* v2);
TensorView* where(Val* c, TensorView* v1, TensorView* v2);
TensorView* where(TensorView* c, TensorView* v1, TensorView* v2);
// threshold
Val* threshold(Val* in, Val* thresh, Val* value);
TensorView* threshold(TensorView* in, Val* thresh, Val* value);
// clamp
Val* clamp(Val* in, Val* min_val, Val* max_val);
TensorView* clamp(TensorView* in, Val* min_val, Val* max_val);

//! Internal operator for supporting backward graphs
//!
//! example:
//!   v1 = T1 [I0(10),I1(20),I2(30),I3(40)]
//!   v2 = sum_to(v1,{30,1}) ------> v2 = T2[I2,R3 (keep_dim)]
//!
//!  This operator will return v1* directly if sizes of v1 root domain
//!  is already the same as shape.
//!
//!  Name of sum_to is different from NV fuser naming,
//!  this is to align with the operator name of at::sum_to.

TensorView* sum_to(TensorView* v1, const std::vector<Val*>& sum_to_size);

TensorView* sum_to(TensorView* v1, const std::vector<int64_t>& sum_to_size);

// Append a new IterDomain to the end of a TenorView to allow
// iterating on a vector type. The input tensor must have
// vector dtype.
TensorView* viewAsScalar(TensorView* inp);

//! A fused pointwise multiply and sum
//!  operator that instantiates the following
//!  fused pattern:
//!     c = mul(tv_a, tv_b);
//!     return sum(c, axes)
//!
//! \param tv_a first multiply operand
//! \param tv_b second multiply operand
//! \param axes axes to sum over, relative to output loop domain
//! \param init sum initial value
//!
//! Note & TODO:
//!   currently only support lowering to a mma op
//!   through this interface and only support fp16 inputs.
//!   will support converting back to multiply and reduce in
//!   a follow up.
TensorView* fusedMultiplySum(
    TensorView* tv_a,
    TensorView* tv_b,
    const std::vector<int64_t>& axes,
    Val* init = nullptr);

// Create a tensor view from the given value. The given value can be a single
// scalar, an array of scalars, or a nested array of scalars.
TensorView* tensor(Val* val);

template <typename T>
TensorView* tensor(const std::vector<T>& vals) {
  return tensor(IrBuilder::arrayExpr(vals));
}

TensorView* argsort(
    TensorView* v1,
    int64_t dim,
    bool descending = false,
    bool stable = false);

//! Grouped matrix multiplication
//!
//! Performs matrix multiplication on grouped sets of matrices using offsets
//! to define variable-sized groups. This op computes:
//!
//! alpha * grouped_mm((mat1 * scale1), (mat2 * scale2), offsets) + bias * beta
//!
//! \param mat1 First set of matrices
//! \param mat2 Second set of matrices
//! \param offsets Offsets tensor defining group boundaries
//! \param scale1 Scale tensor for mat1
//! \param scale2 Scale tensor for mat2
//! \param alpha Global Scaling factor for mat1@mat2
//! \param bias Bias tensor
//! \param beta Scale tensor for bias
//! \param dtype Output dtype, if empty, the output dtype will be the same as
//! the input dtype
//! \param out_block_scale_size Output block scaling factor size, if 0, the
//! output block scaling factor will not be computed
//! \param block_scaling_factor_dtype Block scaling factor dtype. This argument
//! is needed when out_block_scale_size is not 0.
//! \param out_gamma Output gamma flag, if true, the output gamma will be
//! computed
//! \return Result of grouped matrix multiplication
ScaledTensorView grouped_mm(
    TensorView* mat1,
    TensorView* mat2,
    TensorView* offsets,
    TensorView* scale1 = nullptr,
    TensorView* scale2 = nullptr,
    TensorView* alpha = nullptr,
    TensorView* bias = nullptr,
    TensorView* beta = nullptr,
    DataType dtype = DataType::Null,
    int64_t out_block_scale_size = 0,
    DataType block_scaling_factor_dtype = DataType::Null,
    bool out_gamma = false);

//! TopK operation: find the k largest or smallest elements along a dimension
//!
//! Returns the k largest (if largest=true) or smallest (if largest=false)
//! elements of the input tensor along the given dimension.
//!
//! \param v1 Input tensor
//! \param k Number of elements to return (must be non-negative integer)
//! \param dim Dimension along which to find top-k elements (default: -1, last
//! dim)
//! \param largest If true, return largest elements; if false, return smallest
//! (default: true)
//! \param sorted If true, return elements in sorted order (default: false)
//! \param maybe_symbolic If true, this would set the output on the top k
//! IterDomain as IterType::Symbolic, instead of inheriting the iter type from
//! inputs. (default: true)
//! \return TopKResult containing values and indices tensors
//!
//! \note The output tensors have the same shape as the input, except the
//!       specified dimension has size k instead of its original size.
TopKResult topk(
    TensorView* v1,
    Val* k,
    int64_t dim = -1,
    bool largest = true,
    bool sorted = false,
    bool maybe_symbolic = true);

//! Computes an inclusive scan of a tensor in a single dimension.
//!
//! Given a 1D input tensor x, this computes the output
//! recursively via
//!
//!   y = scan(x, 0, Add, zeroVal())
//!
//!   y[0] = x[0]
//!   y[i] = y[i-1] + x[i] for 0 < i < n
//!
//! If the dimension being scanned is an expanded broadcast, we throw an error.
TensorView* scan(
    TensorView* in_tv,
    int64_t dim,
    BinaryOpType op_type,
    Val* init = nullptr);

//! This is an alias for scan(tv, dim, BinaryOpType::Add, zeroVal())
TensorView* prefixSum(TensorView* tv, int64_t dim);

//! Another alias for PyTorch's cumsum
inline TensorView* cumsum(TensorView* tv, int64_t dim) {
  return prefixSum(tv, dim);
}

} // namespace nvfuser

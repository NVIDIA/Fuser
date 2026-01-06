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
NVF_API Val* castOp(DataType dtype, Val* v1);
NVF_API TensorView* castOp(DataType dtype, TensorView* v1);
// If v1 is not dtype, insert a cast op, otherwise return v1
NVF_API Val* maybeCastOp(DataType dtype, Val* v1);
NVF_API TensorView* maybeCastOp(DataType dtype, TensorView* v1);

NVF_API Val* bitCastOp(DataType dtype, Val* v1);
NVF_API TensorView* bitCastOp(DataType dtype, TensorView* v1);

// Perform unary op type and return the output
NVF_API Val* unaryOp(UnaryOpType type, Val* v1);
NVF_API TensorView* unaryOp(UnaryOpType type, TensorView* v1);
NVF_API Val* unaryIsOp(UnaryOpType type, Val* v1);
TensorView* unaryIsOp(UnaryOpType type, TensorView* v1);
NVF_API Val* unaryOp(
    UnaryOpType type,
    Val* v1,
    const TypePromotionConfig& config);
NVF_API TensorView* unaryOp(
    UnaryOpType type,
    TensorView* v1,
    const TypePromotionConfig& config);

// Perform binary op type on v1 and v2 and return a type promoted output.
// Mod, CeilDiv, and LT are considered Int only output operations for now.
NVF_API Val* binaryOp(
    BinaryOpType type,
    Val* v1,
    Val* v2,
    DataType out_dtype = DataType::Null);
NVF_API TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2,
    DataType out_dtype = DataType::Null);
NVF_API TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2,
    DataType out_dtype = DataType::Null);
NVF_API TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2,
    DataType out_dtype = DataType::Null);

NVF_API Val* binaryOp(
    BinaryOpType type,
    Val* v1,
    Val* v2,
    const TypePromotionConfig& config);
NVF_API TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2,
    const TypePromotionConfig& config);
NVF_API TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2,
    const TypePromotionConfig& config);
NVF_API TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2,
    const TypePromotionConfig& config);

// Return a new TensorView consistent with reducing `tv` on specified `axes`
NVF_API TensorView* newForReduction(
    TensorView* tv,
    const std::vector<int64_t>& axes,
    DataType data_type = DataType::Null);

// Perform a reduction operation on v1, initial value for reduction is init,
// reduces across axes, and reduction operation defined by BinaryOp. Reduction
// of size-1 dimension is automatically converted to squeeze.
NVF_API TensorView* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int64_t>& axes,
    Val* init,
    TensorView* v1,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

// Just create a ReductionOp, don't try to simplify it. Don't convert size-1
// reduction into squeeze and don't convert size-0 reduction into full.
NVF_API TensorView* reductionOpRaw(
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
  TensorView* indices =
      nullptr; //!< Indices of the values in the original tensor

  explicit TopKResult(TensorView* in_values, TensorView* in_indices)
      : values(in_values), indices(in_indices) {}
};

//! Welford operator on specified axes. This is currently the only scan op with
//! multiple outputs that is supported. May consider generalization if more scan
//! ops are added.
NVF_API WelfordResult Welford(
    TensorView* tv,
    const std::vector<int64_t>& axes,
    TensorView* init_avg = nullptr,
    TensorView* init_var = nullptr,
    // Initializes to 0 in function definition, doing this so we don't have to
    // import IrBuilder just for this one interface.
    Val* init_N = nullptr);

//! Create a raw WelfordOp. Don't convert size-1 or size-0 reduction into
//! squeeze/full.
NVF_API WelfordResult WelfordRaw(
    TensorView* tv,
    const std::vector<int64_t>& axes,
    TensorView* init_avg = nullptr,
    TensorView* init_var = nullptr,
    // Initializes to 0 in function definition, doing this so we don't have to
    // import IrBuilder just for this one interface.
    Val* init_N = nullptr);

// RNG OPERATIONS
NVF_API TensorView* rand(
    const std::vector<Val*>& shape,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr,
    bool maybe_symbolic = true);
NVF_API TensorView* rand_like(
    TensorView*,
    Val* philox_seed,
    Val* philox_offset);
// Note that overloading these would be convenient, but overloaded functions are
// difficult to cast correctly. In the serde method
// RecordFunctorFactory::setupFunctionMaps(), the op is cast to, for example
// nvfuser::Val* (*)(nvfuser::Val*). In order to avoid errors due to that
// static_cast, we just implement the unary and ternary versions of the random
// *_like operators as separate functions.
NVF_API Val* rand_like(Val*, Val* philox_seed, Val* philox_offset);
NVF_API TensorView* rand_like(TensorView* tv);
NVF_API Val* rand_like(Val* val);

NVF_API TensorView* randn(
    const std::vector<Val*>& shape,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr,
    bool maybe_symbolic = true);
NVF_API TensorView* randn_like(
    TensorView*,
    Val* philox_seed,
    Val* philox_offset);
NVF_API Val* randn_like(Val*, Val* philox_seed, Val* philox_offset);
NVF_API TensorView* randn_like(TensorView* tv);
NVF_API Val* randn_like(Val* val);

NVF_API TensorView* uniform(
    const std::vector<Val*>& shape,
    Val* low,
    Val* high,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr,
    bool maybe_symbolic = true);
NVF_API TensorView* normal(
    const std::vector<Val*>& shape,
    Val* mean,
    Val* std,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr,
    bool maybe_symbolic = true);

// TENSOR FACTORIES
NVF_API TensorView* full(
    const std::vector<Val*>& shape,
    Val* fill_value,
    DataType dtype,
    bool maybe_symbolic = true);
NVF_API TensorView* full_like(TensorView* tv, Val* fill_value, DataType dtype);
NVF_API TensorView* full_like(TensorView* tv, Val* fill_value);
Val* full_like(Val* tv, Val* fill_value);
NVF_API TensorView* zeros(
    const std::vector<Val*>& shape,
    DataType dtype,
    bool maybe_symbolic = true);
NVF_API TensorView* zeros_like(TensorView*);
Val* zeros_like(Val*);
NVF_API TensorView* ones(
    const std::vector<Val*>& shape,
    DataType dtype,
    bool maybe_symbolic = true);
NVF_API TensorView* ones_like(TensorView*);
Val* ones_like(Val*);
NVF_API TensorView* iota(
    Val* length,
    Val* start = nullptr,
    Val* step = nullptr,
    DataType dtype = DataType::Int);
//! WARNING: giving invalid combinations of the start, end and step
//! arguments can result in undefined behavior. Specifically, the
//! signs of `end - start` and step must be the same.
NVF_API TensorView* arange(Val* end, DataType dtype = DataType::Int);
NVF_API TensorView* arange(
    Val* start,
    Val* end,
    DataType dtype = DataType::Int);
NVF_API TensorView* arange(
    Val* start,
    Val* end,
    Val* step,
    DataType dtype = DataType::Int);
NVF_API TensorView* eye(Val* size, DataType dtype);
NVF_API TensorView* eye(Val* rows, Val* cols, DataType dtype);

// UNARY OPERATIONS
// abs
NVF_API Val* abs(Val*);
NVF_API TensorView* abs(TensorView*);
// acos
NVF_API Val* acos(Val*);
NVF_API TensorView* acos(TensorView*);
// acosh
NVF_API Val* acosh(Val*);
NVF_API TensorView* acosh(TensorView*);
// asin
NVF_API Val* asin(Val*);
NVF_API TensorView* asin(TensorView*);
// asinh
NVF_API Val* asinh(Val*);
NVF_API TensorView* asinh(TensorView*);
// atan
NVF_API Val* atan(Val*);
NVF_API TensorView* atan(TensorView*);
// atanh
NVF_API Val* atanh(Val*);
NVF_API TensorView* atanh(TensorView*);
// ceil
NVF_API Val* ceil(Val*);
NVF_API TensorView* ceil(TensorView*);
// cos
NVF_API Val* cos(Val*);
NVF_API TensorView* cos(TensorView*);
// cosh
NVF_API Val* cosh(Val*);
NVF_API TensorView* cosh(TensorView*);
// exp
NVF_API Val* exp(Val*);
NVF_API TensorView* exp(TensorView*);
// exp2
NVF_API Val* exp2(Val*);
NVF_API TensorView* exp2(TensorView*);
// expm1
NVF_API Val* expm1(Val*);
NVF_API TensorView* expm1(TensorView*);
// erf
NVF_API Val* erf(Val*);
NVF_API TensorView* erf(TensorView*);
// erfc
NVF_API Val* erfc(Val*);
NVF_API TensorView* erfc(TensorView*);
// erfinv
NVF_API Val* erfinv(Val*);
NVF_API TensorView* erfinv(TensorView*);
// erfcinv
NVF_API Val* erfcinv(Val*);
NVF_API TensorView* erfcinv(TensorView*);
// floor
NVF_API Val* floor(Val*);
NVF_API TensorView* floor(TensorView*);
// frac
NVF_API Val* frac(Val*);
NVF_API TensorView* frac(TensorView*);
// silu
NVF_API Val* silu(Val*);
NVF_API TensorView* silu(TensorView*);
// lgamma
NVF_API Val* lgamma(Val*);
NVF_API TensorView* lgamma(TensorView*);
// log
NVF_API Val* log(Val*);
NVF_API TensorView* log(TensorView*);
// log10
NVF_API Val* log10(Val*);
NVF_API TensorView* log10(TensorView*);
// log1p
NVF_API Val* log1p(Val*);
NVF_API TensorView* log1p(TensorView*);
// log2
NVF_API Val* log2(Val*);
NVF_API TensorView* log2(TensorView*);
// neg
NVF_API Val* neg(Val*);
NVF_API TensorView* neg(TensorView*);
// logical_not
NVF_API Val* logical_not(Val*);
NVF_API TensorView* logical_not(TensorView*);
// bitwise_not
NVF_API Val* bitwise_not(Val*);
NVF_API TensorView* bitwise_not(TensorView*);
// real
NVF_API Val* real(Val*);
NVF_API TensorView* real(TensorView*);
// reciprocal
NVF_API Val* reciprocal(Val*);
NVF_API TensorView* reciprocal(TensorView*);
// relu
NVF_API Val* relu(Val*);
NVF_API TensorView* relu(TensorView*);
// rsqrt
NVF_API Val* rsqrt(Val*);
NVF_API TensorView* rsqrt(TensorView*);
// round
NVF_API Val* round(Val*);
NVF_API TensorView* round(TensorView*);
// sigmoid
NVF_API Val* sigmoid(Val*);
NVF_API TensorView* sigmoid(TensorView*);
// signbit
NVF_API Val* signbit(Val*);
NVF_API TensorView* signbit(TensorView*);
// sin
NVF_API Val* sin(Val*);
NVF_API TensorView* sin(TensorView*);
// sinh
NVF_API Val* sinh(Val*);
NVF_API TensorView* sinh(TensorView*);
// sqrt
NVF_API Val* sqrt(Val*);
NVF_API TensorView* sqrt(TensorView*);
// tan
NVF_API Val* tan(Val*);
NVF_API TensorView* tan(TensorView*);
// tanh
NVF_API Val* tanh(Val*);
NVF_API TensorView* tanh(TensorView*);
// trunc
NVF_API Val* trunc(Val*);
NVF_API TensorView* trunc(TensorView*);
// bitwise_not
NVF_API Val* bitwise_not(Val*);
NVF_API TensorView* bitwise_not(TensorView*);
// bitceil
NVF_API Val* bitceil(Val*);
NVF_API TensorView* bitceil(TensorView*);
// imag
NVF_API Val* imag(Val*);
NVF_API TensorView* imag(TensorView*);
// isfinite
NVF_API Val* isfinite(Val*);
NVF_API TensorView* isfinite(TensorView*);
// isinf
NVF_API Val* isinf(Val*);
NVF_API TensorView* isinf(TensorView*);
// isnan
NVF_API Val* isnan(Val*);
NVF_API TensorView* isnan(TensorView*);
// isneginf
NVF_API Val* isneginf(Val*);
NVF_API TensorView* isneginf(TensorView*);
// isposinf
NVF_API Val* isposinf(Val*);
NVF_API TensorView* isposinf(TensorView*);
// isreal
NVF_API Val* isreal(Val*);
NVF_API TensorView* isreal(TensorView*);
// print
NVF_API Val* print(Val*);
NVF_API TensorView* print(TensorView*);

// This is a function used to give the symbolic shape of a tensor for use
// with functions like broadcast_in_dim that take a shape vector
// to use to expand an input tensor
NVF_API std::vector<Val*> shape(TensorView* inp);
// Get the symbolic size of a specific dimension of a tensor
NVF_API Val* size(TensorView* inp, int64_t dim);
NVF_API Val* at(const std::vector<Val*>& inp, int64_t index);

// BINARY OPERATIONS
// add
NVF_API Val* add(Val* v1, Val* v2);
NVF_API TensorView* add(TensorView* v1, Val* v2);
NVF_API TensorView* add(Val* v1, TensorView* v2);
NVF_API TensorView* add(TensorView* v1, TensorView* v2);
// atan2
NVF_API Val* atan2(Val* v1, Val* v2);
NVF_API TensorView* atan2(TensorView* v1, Val* v2);
NVF_API TensorView* atan2(Val* v1, TensorView* v2);
NVF_API TensorView* atan2(TensorView* v1, TensorView* v2);
// truediv: promote to float for integer division, has the same semantics as the
// python's operator /
NVF_API Val* truediv(Val* v1, Val* v2);
NVF_API TensorView* truediv(TensorView* v1, Val* v2);
NVF_API TensorView* truediv(Val* v1, TensorView* v2);
NVF_API TensorView* truediv(TensorView* v1, TensorView* v2);
// div: don't promote to float, instead, truncate the result, this has the same
// semantics as the C++'s operator /
NVF_API Val* div(Val* v1, Val* v2);
NVF_API TensorView* div(TensorView* v1, Val* v2);
NVF_API TensorView* div(Val* v1, TensorView* v2);
NVF_API TensorView* div(TensorView* v1, TensorView* v2);
// fmod
NVF_API Val* fmod(Val* v1, Val* v2);
NVF_API TensorView* fmod(TensorView* v1, Val* v2);
NVF_API TensorView* fmod(Val* v1, TensorView* v2);
NVF_API TensorView* fmod(TensorView* v1, TensorView* v2);
// mul
NVF_API Val* mul(Val* v1, Val* v2);
NVF_API TensorView* mul(TensorView* v1, Val* v2);
NVF_API TensorView* mul(Val* v1, TensorView* v2);
NVF_API TensorView* mul(TensorView* v1, TensorView* v2);
// pow
NVF_API Val* pow(Val* v1, Val* v2);
NVF_API TensorView* pow(TensorView* v1, Val* v2);
NVF_API TensorView* pow(Val* v1, TensorView* v2);
NVF_API TensorView* pow(TensorView* v1, TensorView* v2);
// remainder
NVF_API Val* remainder(Val* v1, Val* v2);
NVF_API TensorView* remainder(TensorView* v1, Val* v2);
NVF_API TensorView* remainder(Val* v1, TensorView* v2);
NVF_API TensorView* remainder(TensorView* v1, TensorView* v2);
// sub
NVF_API Val* sub(Val* v1, Val* v2);
NVF_API TensorView* sub(TensorView* v1, Val* v2);
NVF_API TensorView* sub(Val* v1, TensorView* v2);
NVF_API TensorView* sub(TensorView* v1, TensorView* v2);
// maximum
NVF_API Val* maximum(Val* v1, Val* v2);
NVF_API TensorView* maximum(TensorView* v1, Val* v2);
NVF_API TensorView* maximum(Val* v1, TensorView* v2);
NVF_API TensorView* maximum(TensorView* v1, TensorView* v2);
// minimum
NVF_API Val* minimum(Val* v1, Val* v2);
NVF_API TensorView* minimum(TensorView* v1, Val* v2);
NVF_API TensorView* minimum(Val* v1, TensorView* v2);
NVF_API TensorView* minimum(TensorView* v1, TensorView* v2);
// nextafter: Only single- or double-precision
// floating point types (after promotion) are supported.
NVF_API Val* nextafter(Val* v1, Val* v2);
NVF_API TensorView* nextafter(TensorView* v1, Val* v2);
NVF_API TensorView* nextafter(Val* v1, TensorView* v2);
NVF_API TensorView* nextafter(TensorView* v1, TensorView* v2);
// Integer binary ops
// mod
NVF_API Val* mod(Val* v1, Val* v2);
NVF_API TensorView* mod(TensorView* v1, Val* v2);
NVF_API TensorView* mod(Val* v1, TensorView* v2);
NVF_API TensorView* mod(TensorView* v1, TensorView* v2);
// ceilDiv
NVF_API Val* ceilDiv(Val* v1, Val* v2);
TensorView* ceilDiv(TensorView* v1, Val* v2);
TensorView* ceilDiv(Val* v1, TensorView* v2);
TensorView* ceilDiv(TensorView* v1, TensorView* v2);
// Bitwise and logical binary ops
// bitwise_and
NVF_API Val* bitwise_and(Val* v1, Val* v2);
NVF_API TensorView* bitwise_and(TensorView* v1, Val* v2);
NVF_API TensorView* bitwise_and(Val* v1, TensorView* v2);
NVF_API TensorView* bitwise_and(TensorView* v1, TensorView* v2);
// logical_and
NVF_API Val* logical_and(Val* v1, Val* v2);
NVF_API TensorView* logical_and(TensorView* v1, Val* v2);
NVF_API TensorView* logical_and(Val* v1, TensorView* v2);
NVF_API TensorView* logical_and(TensorView* v1, TensorView* v2);
// bitwise_left_shift
NVF_API Val* bitwise_left_shift(Val* v1, Val* v2);
NVF_API TensorView* bitwise_left_shift(TensorView* v1, Val* v2);
NVF_API TensorView* bitwise_left_shift(Val* v1, TensorView* v2);
NVF_API TensorView* bitwise_left_shift(TensorView* v1, TensorView* v2);
// bitwise_right_shift
NVF_API Val* bitwise_right_shift(Val* v1, Val* v2);
NVF_API TensorView* bitwise_right_shift(TensorView* v1, Val* v2);
NVF_API TensorView* bitwise_right_shift(Val* v1, TensorView* v2);
NVF_API TensorView* bitwise_right_shift(TensorView* v1, TensorView* v2);
// logical_right_shift
NVF_API TensorView* logical_right_shift(TensorView* x, TensorView* shift);
NVF_API TensorView* logical_right_shift(TensorView* x, Val* shift);
NVF_API TensorView* logical_right_shift(Val* x, TensorView* shift);
NVF_API Val* logical_right_shift(Val* x, Val* shift);
// bitwise_or
NVF_API Val* bitwise_or(Val* v1, Val* v2);
NVF_API TensorView* bitwise_or(TensorView* v1, Val* v2);
NVF_API TensorView* bitwise_or(Val* v1, TensorView* v2);
NVF_API TensorView* bitwise_or(TensorView* v1, TensorView* v2);
// logical_or
NVF_API Val* logical_or(Val* v1, Val* v2);
NVF_API TensorView* logical_or(TensorView* v1, Val* v2);
NVF_API TensorView* logical_or(Val* v1, TensorView* v2);
NVF_API TensorView* logical_or(TensorView* v1, TensorView* v2);
// bitwise_xor
NVF_API Val* bitwise_xor(Val* v1, Val* v2);
NVF_API TensorView* bitwise_xor(TensorView* v1, Val* v2);
NVF_API TensorView* bitwise_xor(Val* v1, TensorView* v2);
NVF_API TensorView* bitwise_xor(TensorView* v1, TensorView* v2);
// gcd
NVF_API Val* gcd(Val* v1, Val* v2);
NVF_API TensorView* gcd(TensorView* v1, Val* v2);
NVF_API TensorView* gcd(Val* v1, TensorView* v2);
NVF_API TensorView* gcd(TensorView* v1, TensorView* v2);
// Logical binary ops
// eq
NVF_API Val* eq(Val* v1, Val* v2);
NVF_API TensorView* eq(TensorView* v1, Val* v2);
NVF_API TensorView* eq(Val* v1, TensorView* v2);
NVF_API TensorView* eq(TensorView* v1, TensorView* v2);
// ge
NVF_API Val* ge(Val* v1, Val* v2);
NVF_API TensorView* ge(TensorView* v1, Val* v2);
NVF_API TensorView* ge(Val* v1, TensorView* v2);
NVF_API TensorView* ge(TensorView* v1, TensorView* v2);
// gt
NVF_API Val* gt(Val* v1, Val* v2);
NVF_API TensorView* gt(TensorView* v1, Val* v2);
NVF_API TensorView* gt(Val* v1, TensorView* v2);
NVF_API TensorView* gt(TensorView* v1, TensorView* v2);
// le
NVF_API Val* le(Val* v1, Val* v2);
NVF_API TensorView* le(TensorView* v1, Val* v2);
NVF_API TensorView* le(Val* v1, TensorView* v2);
NVF_API TensorView* le(TensorView* v1, TensorView* v2);
// lt
NVF_API Val* lt(Val* v1, Val* v2);
NVF_API NVF_API TensorView* lt(TensorView* v1, Val* v2);
NVF_API TensorView* lt(Val* v1, TensorView* v2);
NVF_API TensorView* lt(TensorView* v1, TensorView* v2);
// ne
NVF_API Val* ne(Val* v1, Val* v2);
NVF_API TensorView* ne(TensorView* v1, Val* v2);
NVF_API TensorView* ne(Val* v1, TensorView* v2);
NVF_API TensorView* ne(TensorView* v1, TensorView* v2);

// complex
NVF_API Val* complex(Val* v1, Val* v2);
NVF_API TensorView* complex(TensorView* v1, TensorView* v2);

// REDUCTION OPERATIONS
NVF_API TensorView* sum(
    TensorView* v1,
    const std::vector<int64_t>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

NVF_API TensorView* prod(
    TensorView* v1,
    const std::vector<int64_t>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

NVF_API TensorView* max(
    TensorView* v1,
    const std::vector<int64_t>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

NVF_API TensorView* min(
    TensorView* v1,
    const std::vector<int64_t>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

// COMPOUND OPERATIONS
// add_alpha
NVF_API Val* add_alpha(Val* v1, Val* v2, Val* s);
NVF_API TensorView* add_alpha(TensorView* v1, Val* v2, Val* s);
NVF_API TensorView* add_alpha(Val* v1, TensorView* v2, Val* s);
NVF_API TensorView* add_alpha(TensorView* v1, TensorView* v2, Val* s);
// sub_alpha
NVF_API Val* sub_alpha(Val* v1, Val* v2, Val* s);
NVF_API TensorView* sub_alpha(TensorView* v1, Val* v2, Val* s);
NVF_API TensorView* sub_alpha(Val* v1, TensorView* v2, Val* s);
NVF_API TensorView* sub_alpha(TensorView* v1, TensorView* v2, Val* s);
// lerp
NVF_API Val* lerp(Val* start, Val* end, Val* weight);
NVF_API TensorView* lerp(TensorView* start, Val* end, Val* weight);
NVF_API TensorView* lerp(Val* start, TensorView* end, Val* weight);
NVF_API TensorView* lerp(Val* start, Val* end, TensorView* weight);
NVF_API TensorView* lerp(TensorView* start, TensorView* end, Val* weight);
NVF_API TensorView* lerp(TensorView* start, Val* end, TensorView* weight);
NVF_API TensorView* lerp(Val* start, TensorView* end, TensorView* weight);
NVF_API TensorView* lerp(
    TensorView* start,
    TensorView* end,
    TensorView* weight);

// addcmul
NVF_API Val* addcmul(Val* v1, Val* v2, Val* v3, Val* s);
NVF_API TensorView* addcmul(TensorView* v1, Val* v2, Val* v3, Val* s);
NVF_API TensorView* addcmul(Val* v1, TensorView* v2, Val* v3, Val* s);
NVF_API TensorView* addcmul(Val* v1, Val* v2, TensorView* v3, Val* s);
NVF_API TensorView* addcmul(TensorView* v1, TensorView* v2, Val* v3, Val* s);
NVF_API TensorView* addcmul(TensorView* v1, Val* v2, TensorView* v3, Val* s);
NVF_API TensorView* addcmul(Val* v1, TensorView* v2, TensorView* v3, Val* s);
NVF_API TensorView* addcmul(
    TensorView* v1,
    TensorView* v2,
    TensorView* v3,
    Val* s);

// TERNARY OPERATIONS
// where
NVF_API Val* where(Val* c, Val* v1, Val* v2);
NVF_API TensorView* where(TensorView* c, Val* v1, Val* v2);
NVF_API TensorView* where(Val* c, TensorView* v1, Val* v2);
NVF_API TensorView* where(Val* c, Val* v1, TensorView* v2);
NVF_API TensorView* where(TensorView* c, TensorView* v1, Val* v2);
NVF_API TensorView* where(TensorView* c, Val* v1, TensorView* v2);
NVF_API TensorView* where(Val* c, TensorView* v1, TensorView* v2);
NVF_API TensorView* where(TensorView* c, TensorView* v1, TensorView* v2);
// threshold
NVF_API Val* threshold(Val* in, Val* thresh, Val* value);
NVF_API TensorView* threshold(TensorView* in, Val* thresh, Val* value);
// clamp
NVF_API Val* clamp(Val* in, Val* min_val, Val* max_val);
NVF_API TensorView* clamp(TensorView* in, Val* min_val, Val* max_val);

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

NVF_API TensorView* sum_to(
    TensorView* v1,
    const std::vector<Val*>& sum_to_size);

NVF_API TensorView* sum_to(
    TensorView* v1,
    const std::vector<int64_t>& sum_to_size);

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
NVF_API TensorView* fusedMultiplySum(
    TensorView* tv_a,
    TensorView* tv_b,
    const std::vector<int64_t>& axes,
    Val* init = nullptr);

// Create a tensor view from the given value. The given value can be a single
// scalar, an array of scalars, or a nested array of scalars.
NVF_API TensorView* tensor(Val* val);

template <typename T>
NVF_API TensorView* tensor(const std::vector<T>& vals) {
  return tensor(IrBuilder::arrayExpr(vals));
}

NVF_API TensorView* argsort(
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
NVF_API ScaledTensorView grouped_mm(
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
NVF_API TopKResult topk(
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
//! If the dimension being scanned is an expanded broadcast, we throw
//! an error.
//!
//! Note that unlike reductions, low precision inputs are not
//! automatically upcast to float, as that is the PyTorch convention.
NVF_API TensorView* scan(
    TensorView* in_tv,
    int64_t dim,
    BinaryOpType op_type,
    Val* init = nullptr);

//! This is an alias for scan(tv, dim, BinaryOpType::Add, zeroVal())
NVF_API TensorView* prefixSum(TensorView* tv, int64_t dim);

//! Another alias for PyTorch's cumsum
NVF_API inline TensorView* cumsum(TensorView* tv, int64_t dim) {
  return prefixSum(tv, dim);
}

struct BlockQuantizationResults {
 public:
  TensorView* quantized_tensor = nullptr;
  TensorView* block_scales = nullptr;

  explicit BlockQuantizationResults(
      TensorView* in_quantized_tensor,
      TensorView* in_block_scales)
      : quantized_tensor(in_quantized_tensor), block_scales(in_block_scales) {}
};

// API for block quantization.
// Currently We take FP32 or BF16/FP16 inputs and produce two outputs,
// quantized ouptuts and block scales.
// Quantized outputs can be nvFP4(DataType::Float4_e2m1fn) or mxFP8
// (DataType::Float8_e4m3fn).
// Block scales for nvFP4 is DataType::Float8_e4m3fn and
// for mxFP8 is DataType::Float8_e8m0fnu.
// We optionally take a block size as an input but currenlty just support 16 (32
// for mxFP8). The flag swizzle_scales which generates swizzled block scales is
// only supported when quantizing to nvFP4.
NVF_API BlockQuantizationResults blockQuantize(
    TensorView* input,
    TensorView* global_scaling_factor = nullptr,
    int64_t block_size = 16,
    bool swizzle_scales = false,
    DataType out_dtype = DataType::Float4_e2m1fn);

} // namespace nvfuser

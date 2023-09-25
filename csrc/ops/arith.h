// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <ir/base_nodes.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
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

// Perform a reduction operation on v1, initial value for reduction is init,
// reduces across axes, and reduction operation defined by BinaryOp. Reduction
// of size-1 dimension is automatically converted to squeeze.
TensorView* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    TensorView* v1,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

// Just create a ReductionOp, don't try to simplify it. Don't convert size-1
// reduction into squeeze and don't convert size-0 reduction into full.
TensorView* reductionOpRaw(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    TensorView* v1,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

//! Auxiliary Struct holding result of
//! a single welford op in ternsorview
class WelfordResult {
 public:
  TensorView* avg;
  TensorView* var_sum;
  TensorView* n;

  explicit WelfordResult(
      TensorView* in_avg,
      TensorView* in_var_sum,
      TensorView* in_n,
      const bool check_definition = true);
};

//! Welford operator on specified axes. This is currently the only scan op with
//! multiple outputs that is supported. May consider generalization if more scan
//! ops are added.
WelfordResult Welford(
    TensorView* tv,
    const std::vector<int>& axes,
    TensorView* init_avg = nullptr,
    TensorView* init_var = nullptr,
    // Initializes to 0 in function definition, doing this so we don't have to
    // import IrBuilder just for this one interface.
    Val* init_N = nullptr);

//! Create a raw WelfordOp. Don't convert size-1 or size-0 reduction into
//! squeeze/full.
WelfordResult WelfordRaw(
    TensorView* tv,
    const std::vector<int>& axes,
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
    Val* philox_offset = nullptr);
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
    Val* philox_offset = nullptr);
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
    Val* philox_offset = nullptr);
TensorView* normal(
    const std::vector<Val*>& shape,
    Val* mean,
    Val* std,
    DataType dtype,
    Val* philox_seed = nullptr,
    Val* philox_offset = nullptr);

// TENSOR FACTORIES
TensorView* full(
    const std::vector<Val*>& shape,
    Val* fill_value,
    DataType dtype);
TensorView* full_like(TensorView* tv, Val* fill_value, DataType dtype);
TensorView* full_like(TensorView* tv, Val* fill_value);
Val* full_like(Val* tv, Val* fill_value);
TensorView* zeros(const std::vector<Val*>& shape, DataType dtype);
TensorView* zeros_like(TensorView*);
Val* zeros_like(Val*);
TensorView* ones(const std::vector<Val*>& shape, DataType dtype);
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

// Broadcasts inp based on bool vector. Size of broadcast bool vector should be
// the number of dims desired in the broadcasted tensor. This vector should be
// true if output dim should be a broadcasted dim, and false if it is not a
// broadcasted dim. Number of false entires must match the number of input dims.
TensorView* broadcast(
    TensorView* inp,
    const std::vector<bool>& is_broadcast_dim);

// Expands input based on provided sizes. expand_sizes should be larger than
// the input's root domain (really rfactor) and will broadcast on inner
// dimensions. expand_sizes should be -1 for any dimension that should remain a
// symbolic size. For dimensions that remain broadcast after the expand should
// be set to 1, any dimension being expanded must be marked as a broadcast in
// the input and will be expanded to the provided constant size. Any dimension
// that's symbolic in the input but specified as a non -1 value will be set to
// that constant value.
TensorView* expand(TensorView* inp, const std::vector<Val*>& expanded_sizes);

// Expands input based on other. For dimensions in inp that are broadcast with a
// matching entry in other that's either a broadcast with expanded extent or a
// non broadcasted iter domain, inp will be expanded to other's size.
TensorView* expand_as(TensorView* inp, TensorView* other);

// This is a function used to give the symbolic sizes of a tensor for use
// with functions like broadcast_in_size that take in a vector of sizes
// to use to expand an input tensor
std::vector<Val*> tensor_sizes(TensorView* inp);
// This is a function used to give the symbolic shape of a tensor for use
// with functions like broadcast_in_dim that take a shape vector
// to use to expand an input tensor
std::vector<Val*> shape(TensorView* inp);
// Get the symbolic size of a specific dimension of a tensor
Val* size(TensorView* inp, int64_t dim);
Val* at(std::vector<Val*>& inp, int64_t index);

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
    const std::vector<int>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

TensorView* prod(
    TensorView* v1,
    const std::vector<int>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

TensorView* max(
    TensorView* v1,
    const std::vector<int>& reduction_axes,
    bool keep_dim = false,
    DataType dtype = DataType::Null);

TensorView* min(
    TensorView* v1,
    const std::vector<int>& reduction_axes,
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

//! Shift a tensor to a direction specified by offsets.
//!
//! Example:
//!   t0: 2D tensor of size N by M
//!   t1 = shift(t0, {1, -1});
//!
//!   then:
//!     t1[i, j] = t0[i-1, j+1] for 1 <= i < N and 0 <= j < M-1.
//!     t1[i, j] = 0, otherwise
//!
//! The pad option controls how out-of-boundary accesses are
//! handled. It specifies how many zeros are logically padded. If no
//! pad option is given, it automatically pads the input tensor so
//! that the output tensor has the same extent for each axis.
//!
//! When a padding value is smaller than the absolute value of a shift
//! offset, the output axis still has the same extent but its start or
//! stop offset is moved inward to signify those outside of the offset
//! are invalid.
//!
//! It is not allowed to use padding values that are larger than shift
//! offsets, which would mean output extentes would be larger than
//! input extents
TensorView* shift(
    TensorView* inp,
    const std::vector<int>& offsets,
    const std::vector<int>& pad_width = {});

TensorView* shift(TensorView* inp, const std::vector<int>& offsets, bool pad);

//! Gather a window of nearby elements for each element.
//!
//! Each window of size window_shape is stored as a additional
//! innermost domain, meaning that the number of dimensions of the
//! output tensor doubles. The pad_width parameter specifies the
//! padding width of each side of each axis. The strides parameter
//! specifies striding of the operation. Non-unit striding is
//! implemented with strided split, whose outer output domain becomes
//! the root domain for subsequent consumers. The inner output domain
//! becomes a Stride domain, which is ignored by subsequent consumers.
//! Only valid input ranges are fed into strided splits.
//!
//! When trim_out_of_bounds is true, the values at the first and last
//! ends that are outside of the start and stop offsets are
//! effetively trimmed by partial split by 1.
//!
//! Example 1:
//!   t0: 2D tensor of [N, M]
//!   t1 = gather(t0, {1, 3}, {{0, 0}, {1, 1}});
//!
//!   then:
//!     t1: [N, M, 1, 3]
//!     t1[i, j, k, l] = The value at the window position of [k, l]
//!                      for t0[i, j]
//!
//! Example 2.1 (without trimming):
//!   t0: 2D tensor of [N, M]
//!   t1 = gather(t0, {2, 2}, {{0, 0}, {0, 0}});
//!
//!   then:
//!     t1: [N (stop offset: 1), M (stop offset: 1, 2, 2)]
//!
//! Example 2.1 (with trimming)
//!   t0: 2D tensor of [N, M]
//!   t1 = gather(t0, {2, 2}, {{0, 0}, {0, 0}}, true);
//!
//!   then:
//!     t1: [ceilDiv(N - 1, 1), ceilDiv(M - 1, 1), 2, 2]
//!
//! Example 3:
//!   t0: 2D tensor of [N, M]
//!   t1 = gather(t0, {3, 3}, {{0, 0}, {0, 0}}, {3, 3});
//!
//!   then:
//!     t1: [ceilDiv(N - 2, 3), ceilDiv(M - 2, 3), 2, 2]
//!
TensorView* gather(
    TensorView* inp,
    const std::vector<int>& window_shape,
    const std::vector<std::vector<int>>& pad_width,
    const std::vector<int>& strides = {},
    bool trim_out_of_bounds = false);

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
//! \param axes axes to sum over
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
    const std::vector<int>& axes,
    Val* init = nullptr);

// Create a tensor view from the given value. The given value can be a single
// scalar, an array of scalars, or a nested array of scalars.
TensorView* tensor(Val* val);

template <typename T>
TensorView* tensor(const std::vector<T>& vals) {
  return tensor(IrBuilder::arrayExpr(vals));
}

} // namespace nvfuser

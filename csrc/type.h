// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <c10/macros/Export.h>

#include <array>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
#include <variant>

namespace nvfuser {

// https://stackoverflow.com/questions/18837857/cant-use-enum-class-as-unordered-map-key
struct TypeHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

// Order of strength
enum class ValType {
  TensorDomain,
  IterDomain,
  TensorView,
  Scalar,
  NamedScalar,
  Predicate,
  TensorIndex,
  AggregateVal,
  Attribute
};

// Manual - The user provides the Bool value. Predicate generation is bypassed.
// Inline corresponds with PredicateCompute::getInlinePredicate
// Unswitch corresponds with UnswitchPredicate::get
// Misaligned - PredicateCompute::getInlinePredicate + Misaligned flag
// Shift - ShiftPredicateInserter::getShiftPredicate
// Padding - ShiftPredicateInserter::getPaddingPredicate
// ReductionWrite - Same as Inline but without reduction axes
// LoopRotation - Predicate added by loop rotation, currently always true.
enum class PredicateType {
  Manual,
  Inline,
  Unswitch,
  Vectorize,
  Misaligned,
  Shift,
  Padding,
  ReductionWrite,
  LoopRotation
};

// Index type is a convenience type that may be a 64 or 32 signed integer.
// This is helpful for math on indexing/size when we don't know what the index
// type might be. This allows us to prevent assuming the welford count must be
// int64_t which is relatively heavy to carry around. Index will be resolved
// at compile time with KernelIndexMode.
enum class PrimDataType {
  Double,
  Float,
  Half,
  Int,
  Index,
  Int32,
  Bool,
  BFloat16,
  ComplexFloat,
  ComplexDouble,
  // Pointers
  SMemAddress,
  // Null
  Null
};

struct DataType;

struct ArrayOf {
  std::shared_ptr<DataType> type;
  size_t size;
  inline bool operator==(const ArrayOf& other) const;
};

struct PointerOf {
  std::shared_ptr<DataType> type;
  inline bool operator==(const PointerOf& other) const;
};

struct DataType {
  using VariantOfSupportedTypes =
      std::variant<PrimDataType, ArrayOf, PointerOf>;
  VariantOfSupportedTypes type = PrimDataType::Null;

  DataType() = default;
  DataType(const VariantOfSupportedTypes& type) : type(type) {}
  DataType(const PrimDataType& type) : type(type) {}
  DataType(const ArrayOf& type) : type(type) {}
  DataType(const PointerOf& type) : type(type) {}

  static constexpr PrimDataType Double = PrimDataType::Double;
  static constexpr PrimDataType Float = PrimDataType::Float;
  static constexpr PrimDataType Half = PrimDataType::Half;
  static constexpr PrimDataType Int = PrimDataType::Int;
  static constexpr PrimDataType Index = PrimDataType::Index;
  static constexpr PrimDataType Int32 = PrimDataType::Int32;
  static constexpr PrimDataType Bool = PrimDataType::Bool;
  static constexpr PrimDataType BFloat16 = PrimDataType::BFloat16;
  static constexpr PrimDataType ComplexFloat = PrimDataType::ComplexFloat;
  static constexpr PrimDataType ComplexDouble = PrimDataType::ComplexDouble;
  static constexpr PrimDataType SMemAddress = PrimDataType::SMemAddress;
  static constexpr PrimDataType Null = PrimDataType::Null;
};

inline bool operator==(const DataType& lhs, const DataType& rhs) {
  return lhs.type == rhs.type;
}

inline bool operator!=(const DataType& lhs, const DataType& rhs) {
  return !operator==(lhs, rhs);
}

bool ArrayOf::operator==(const ArrayOf& other) const {
  return *type == *other.type && size == other.size;
}

bool PointerOf::operator==(const PointerOf& other) const {
  return *type == *other.type;
}

enum class KernelIndexMode { INT32, INT64 };

PrimDataType indexModeToDtype(KernelIndexMode index_mode);
KernelIndexMode indexTypeToMode(DataType index_type);

// Returns if the datatype is a floating point type
TORCH_CUDA_CU_API bool isFloatingPointType(DataType dtype);
// Returns if the datatype is an integer type
TORCH_CUDA_CU_API bool isIntegralType(DataType dtype);
// Returns if the datatype is a pointer type
TORCH_CUDA_CU_API bool isPointerType(DataType dtype);
// Returns if the datatype is an boolean type
TORCH_CUDA_CU_API bool isBooleanType(DataType dtype);
// Returns if the datatype is a complex type
TORCH_CUDA_CU_API bool isComplexType(DataType dtype);
// Return the corresponding scalar of a complex type
DataType getTypeFromComplexType(DataType dtype);
// Return if the datatype is supported on the current device
TORCH_CUDA_CU_API bool isSupportedTypeByDevice(DataType dtype);

template <PrimDataType DT>
struct DataTypeToNativeType;

template <typename NativeType>
struct NativeTypeToDataType;

#define DEFINE_DATATYPE_TO_NATIVE_TYPE(data_type, native_type) \
  template <>                                                  \
  struct DataTypeToNativeType<data_type> {                     \
    using type = native_type;                                  \
  };                                                           \
  template <>                                                  \
  struct NativeTypeToDataType<native_type> {                   \
    static constexpr PrimDataType type = data_type;            \
  };

// TODO: Add more type specializations
DEFINE_DATATYPE_TO_NATIVE_TYPE(DataType::Float, float);
DEFINE_DATATYPE_TO_NATIVE_TYPE(DataType::Double, double);
DEFINE_DATATYPE_TO_NATIVE_TYPE(DataType::Int, int64_t);
DEFINE_DATATYPE_TO_NATIVE_TYPE(DataType::Int32, int);
DEFINE_DATATYPE_TO_NATIVE_TYPE(DataType::Bool, bool);
DEFINE_DATATYPE_TO_NATIVE_TYPE(DataType::ComplexFloat, std::complex<float>);
DEFINE_DATATYPE_TO_NATIVE_TYPE(DataType::ComplexDouble, std::complex<double>);

#undef DEFINE_DATATYPE_TO_NATIVE_TYPE

enum class UnaryOpType {
  Abs,
  Acos,
  Acosh,
  Address,
  Asin,
  Asinh,
  Atan,
  Atanh,
  Cast,
  Ceil,
  Cos,
  Cosh,
  Exp,
  Exp2,
  Expm1,
  Erf,
  Erfc,
  Erfinv,
  Erfcinv,
  Floor,
  Frac,
  Gelu,
  Imag,
  Silu,
  Lgamma,
  Log,
  Log10,
  Log1p,
  Log2,
  BitCast,
  Neg,
  Real,
  Reciprocal,
  Relu,
  Rsqrt,
  Round,
  Set,
  Sigmoid,
  Sin,
  Sinh,
  Sqrt,
  Tan,
  Tanh,
  Trunc,

  // Tools to help debugging
  Print,

  // Might be a bitwise operator or boolean operator.
  Not,

  // Operators returning boolean values
  IsFinite,
  IsInf,
  IsNan,
  IsNegInf,
  IsPosInf,
  IsReal,
};

// Primarily for Not, which could be Not a boolean, or a bitwise not.
bool alsoBooleanOperator(const UnaryOpType uopt);

// TODO: Order of this list is important as it affects type promotion. it's not
// in the right order now.
enum class BinaryOpType {
  // Math Ops
  Add,
  Atan2,
  Div,
  Fmod,
  Max,
  Min,
  Mul,
  Pow,
  Remainder,
  Sub,
  // TypeAs,

  // Integer output ops. If changing modify isIntegerOp
  Mod,
  CeilDiv,
  Lshift,
  Rshift,

  // Logical Ops
  // Int operations, leave position of Mod as first logical op see
  // isLogicalOp(BinaryOpType bopt)
  Eq,
  GE,
  GT,
  LE,
  LT,
  NE,

  // Maybe bitwise or boolean op, leave position of and as first bool/int
  // op. These are ops that have different operators based on output type. See
  // is boolean op. These ops also don't work on floating point inputs.
  And,
  Or,
  Xor
};

enum class ScatterOpType { Set };

enum class RNGOpType {
  Uniform, // Uniform in [0, 1)
  UniformRange, // Uniform in [low, high]
  NormalStandard, // Normal with mean 0, std 1
  NormalGeneral, // Normal with given mean and std
};

// Return if output of operator should be a boolean
bool isIntegerOp(const BinaryOpType bopt);

// Return if output of operator should be a boolean
bool isLogicalOp(const BinaryOpType bopt);

// Operations that could be a bitwise operation or a boolean operation depending
// on input, for example bitwise_and is also used for boolean and in the jit
bool alsoBooleanOperator(const BinaryOpType bopt);

enum class TernaryOpType { Clamp, Lerp, Threshold, Where };

enum class ParallelType {
  BIDz,
  BIDy,
  BIDx,
  TIDz,
  TIDy,
  TIDx,
  Vectorize,
  MisalignedVectorize,
  Unroll,
  Unswitch,
  Mma,
  Group,
  Serial
};

TORCH_CUDA_CU_API std::unordered_set<ParallelType> allParallelTypesExcept(
    const std::unordered_set<ParallelType>& except);

static constexpr std::array<ParallelType, 6> kParallelTypeThreads = {
    ParallelType::BIDx,
    ParallelType::BIDy,
    ParallelType::BIDz,
    ParallelType::TIDx,
    ParallelType::TIDy,
    ParallelType::TIDz};

static constexpr std::array<ParallelType, 3> kParallelTypeBIDs = {
    ParallelType::BIDx,
    ParallelType::BIDy,
    ParallelType::BIDz};

static constexpr std::array<ParallelType, 3> kParallelTypeTIDs = {
    ParallelType::TIDx,
    ParallelType::TIDy,
    ParallelType::TIDz};

enum class MemoryType { Local, Shared, Global };

// sometimes broadcasted tensors may be inputed in the kernel with an explicit 1
// size. If that size is there, we need to account that there's also a stride
// there, even if the stride = 0. If we don't account for that stride when
// accessing a tensor like: [b2{1}, i0, i1] we would linearize the access like:
// [i0*stride[0] + i1*stride[1]] when it should be: [i0*stride[1] +
// i1*stride[2]]. Broadcasts that translate to a physical memory dim we consider
// "with stride", Broadcasts only through our broadcast op we consider "without
// stride"
enum class IterType {
  Iteration,
  Reduction,
  Broadcast,
  Gather,
  Stride,
  GatherScatter,
  VectorComponent
};

// Used for Iteration Domain mapping modes in ComputeAtMap
enum class IdMappingMode { EXACT, ALMOSTEXACT, LOOP, PERMISSIVE };

static constexpr std::array<IdMappingMode, 4> kIdMappingModes = {
    IdMappingMode::EXACT,
    IdMappingMode::ALMOSTEXACT,
    IdMappingMode::LOOP,
    IdMappingMode::PERMISSIVE};

// Used to annotate the special memory intrinsics that a loadstore
//  op will be lowered to.
enum class LoadStoreOpType {
  LdMatrix,
  LdMatrixTranspose,
  CpAsyncCa,
  CpAsyncCg
};

// Used to label what part of the double buffered iterdomain
//  a for loop is materializing.
enum class DoubleBufferLoopStage { NotApplicable, Prolog, Main, Epilog };

//! Supported swizzle types,
//!  corresponds to swizzles functions on the runtime cuda
//!  naming it swizzle_2d to reserve the options to have a swizzle_1d.
//!
//!  TODO: unify with existing swizzle logic, currently
//!    doesn't have the same type.
enum class Swizzle2DType {
  NoSwizzle = 0,
  ZShape,
  Transpose,
  XOR,
  Scatter,
  CyclicShift
};

//! Modes of swizzle, see [Note on swizzle mode].
enum class SwizzleMode { NoSwizzle = 0, Data, Loop };

// Returns if function needs an f suffix on the operator when operating on a
// float value i.e. sin->sinf
bool needFloatSuffix(UnaryOpType t);
bool needFloatSuffix(BinaryOpType t);
bool needFloatSuffix(RNGOpType t);

ValType promote_type(const ValType& t1, const ValType& t2);
DataType promote_type(const DataType& t1, const DataType& t2);

// If type cannot be found (i.e. codegen does not support provided type) returns
// DataType::Null
TORCH_CUDA_CU_API DataType aten_to_data_type(const at::ScalarType& scalar_type);
TORCH_CUDA_CU_API at::ScalarType data_type_to_aten(const DataType& data_type);

TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ValType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const PredicateType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const DataType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const UnaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const BinaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const TernaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ScatterOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const RNGOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ParallelType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const MemoryType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const IterType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const IdMappingMode);
TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream&,
    const LoadStoreOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream&,
    const DoubleBufferLoopStage);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const Swizzle2DType&);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const SwizzleMode&);
TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream&,
    const KernelIndexMode&);

std::string stringifyBooleanOp(const UnaryOpType);
std::string stringifyBooleanOp(const BinaryOpType);

std::string stringifyThreadSize(const ParallelType);
std::string stringifyThread(const ParallelType);
TORCH_CUDA_CU_API std::string typePrefix(const DataType);

// TODO: ThreadDim should be BlockDim and BlockDim should be GridDim
// Returns if parallel type is TID[x, y, z]
TORCH_CUDA_CU_API bool isParallelTypeThreadDim(ParallelType);
// Returns if parallel type is BID[x, y, z]
TORCH_CUDA_CU_API bool isParallelTypeBlockDim(ParallelType);
// Returns if parallel type is a grid or block parallelization dimension
TORCH_CUDA_CU_API bool isParallelTypeThread(ParallelType);

TORCH_CUDA_CU_API bool isParallelTypeVectorize(ParallelType);

TORCH_CUDA_CU_API c10::optional<std::string> inline_op_str(const UnaryOpType);
TORCH_CUDA_CU_API c10::optional<std::string> inline_op_str(const BinaryOpType);
TORCH_CUDA_CU_API c10::optional<std::string> inline_op_str(const RNGOpType);
TORCH_CUDA_CU_API c10::optional<std::string> integer_op_str(const BinaryOpType);
TORCH_CUDA_CU_API c10::optional<std::string> bool_op_str(const BinaryOpType);
TORCH_CUDA_CU_API const char* predicate_type2string(PredicateType t);

TORCH_CUDA_CU_API c10::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>&);

TORCH_CUDA_CU_API size_t dataTypeSize(DataType type);

// If the index type is known it will be automatically used here
TORCH_CUDA_CU_API size_t dataTypeSize(DataType type, DataType index_type);

enum class LaunchConfigType {
  Compatible,
  SharedMemory,
  BIDz,
  BIDy,
  BIDx,
  TIDz,
  TIDy,
  TIDx
};

const char* const kMagicZeroName = "nvfuser_zero";

//! Maximum number of reductions that can be grouped together. The
//! limit can be increased by extending struct Tuple define in tuple.cu.
static constexpr int kMaxNumGroupedReductions = 16;

} // namespace nvfuser

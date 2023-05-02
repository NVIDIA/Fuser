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
TORCH_CUDA_CU_API inline bool isFloatingPointType(DataType dtype) {
  TORCH_CHECK(
      dtype != DataType::Null,
      "Null type is not a valid argument to isFloatingPointType");
  return dtype == DataType::Double || dtype == DataType::Float ||
      dtype == DataType::Half || dtype == DataType::BFloat16;
}

// Returns if the datatype is an integer type
TORCH_CUDA_CU_API inline bool isIntegralType(DataType dtype) {
  return std::visit(
      [](auto&& dtype) {
        using T = std::decay_t<decltype(dtype)>;
        if constexpr (std::is_same_v<T, PrimDataType>) {
          switch (dtype) {
            case DataType::Index:
            case DataType::Int:
            case DataType::Int32:
              return true;
            case DataType::Null:
              TORCH_CHECK(
                  false, "Null type is not a valid argument to isIntegralType");
            default:
              return false;
          }
        }
        return false;
      },
      dtype.type);
}

// Returns if the datatype is a pointer type
TORCH_CUDA_CU_API inline bool isPointerType(DataType dtype) {
  return std::holds_alternative<PointerOf>(dtype.type) ||
      dtype == DataType::SMemAddress;
}

// Returns if the datatype is an integer or pointer type
TORCH_CUDA_CU_API inline bool isIntegralOrPointerType(DataType dtype) {
  return isIntegralType(dtype) || isPointerType(dtype);
}

// Returns if the datatype is a boolean type
TORCH_CUDA_CU_API inline bool isBooleanType(DataType dtype) {
  TORCH_CHECK(
      dtype != DataType::Null,
      "Null type is not a valid argument to isBooleanType");
  return dtype == DataType::Bool;
}

// Returns if the datatype is a complex type
TORCH_CUDA_CU_API inline bool isComplexType(DataType dtype) {
  TORCH_CHECK(
      dtype != DataType::Null,
      "Null type is not a valid argument to isComplexType");
  return dtype == DataType::ComplexFloat || dtype == DataType::ComplexDouble;
}

// Return the corresponding scalar of a complex type
DataType getTypeFromComplexType(DataType dtype);
// Return the corresponding complex type of a scalar
DataType getComplexTypeFromType(DataType dtype);
// Return if the datatype is supported on the current device
TORCH_CUDA_CU_API bool isSupportedTypeByDevice(DataType dtype);

template <PrimDataType DT>
struct DataTypeToNativeType;

template <PrimDataType DT>
struct DataTypeToNativeTypeWithC10Complex;

template <PrimDataType DT>
struct DataTypeToAtenType;

template <typename NativeType>
struct NativeTypeToDataType;

template <typename NativeType>
struct NativeTypeWithC10ComplexToDataType;

template <at::ScalarType aten_type>
struct AtenTypeToDataType;

template <at::ScalarType aten_type>
struct AtenTypeToNativeType;

template <at::ScalarType aten_type>
struct AtenTypeToNativeTypeWithC10Complex;

#define DEFINE_DATATYPE_TO_NATIVE_TYPE(                                     \
    data_type, at_type, native_type, native_type_with_c10_complex)          \
  template <>                                                               \
  struct DataTypeToNativeType<data_type> {                                  \
    using type = native_type;                                               \
  };                                                                        \
  template <>                                                               \
  struct DataTypeToNativeTypeWithC10Complex<data_type> {                    \
    using type = native_type_with_c10_complex;                              \
  };                                                                        \
  template <>                                                               \
  struct DataTypeToAtenType<data_type> {                                    \
    static constexpr at::ScalarType type = at_type;                         \
  };                                                                        \
  template <>                                                               \
  struct NativeTypeToDataType<native_type> {                                \
    static constexpr PrimDataType type = data_type;                         \
  };                                                                        \
  template <>                                                               \
  struct NativeTypeWithC10ComplexToDataType<native_type_with_c10_complex> { \
    static constexpr PrimDataType type = data_type;                         \
  };                                                                        \
  template <>                                                               \
  struct AtenTypeToDataType<at_type> {                                      \
    static constexpr PrimDataType type = data_type;                         \
  };                                                                        \
  template <>                                                               \
  struct AtenTypeToNativeType<at_type> {                                    \
    using type = native_type;                                               \
  };                                                                        \
  template <>                                                               \
  struct AtenTypeToNativeTypeWithC10Complex<at_type> {                      \
    using type = native_type_with_c10_complex;                              \
  };

DEFINE_DATATYPE_TO_NATIVE_TYPE(
    DataType::Float,
    at::ScalarType::Float,
    float,
    float);
DEFINE_DATATYPE_TO_NATIVE_TYPE(
    DataType::Double,
    at::ScalarType::Double,
    double,
    double);
DEFINE_DATATYPE_TO_NATIVE_TYPE(
    DataType::Half,
    at::ScalarType::Half,
    at::Half,
    at::Half);
DEFINE_DATATYPE_TO_NATIVE_TYPE(
    DataType::BFloat16,
    at::ScalarType::BFloat16,
    at::BFloat16,
    at::BFloat16);
DEFINE_DATATYPE_TO_NATIVE_TYPE(
    DataType::Int,
    at::ScalarType::Long,
    int64_t,
    int64_t);
DEFINE_DATATYPE_TO_NATIVE_TYPE(DataType::Int32, at::ScalarType::Int, int, int);
DEFINE_DATATYPE_TO_NATIVE_TYPE(
    DataType::Bool,
    at::ScalarType::Bool,
    bool,
    bool);
DEFINE_DATATYPE_TO_NATIVE_TYPE(
    DataType::ComplexFloat,
    at::ScalarType::ComplexFloat,
    std::complex<float>,
    c10::complex<float>);
DEFINE_DATATYPE_TO_NATIVE_TYPE(
    DataType::ComplexDouble,
    at::ScalarType::ComplexDouble,
    std::complex<double>,
    c10::complex<double>);

#undef DEFINE_DATATYPE_TO_NATIVE_TYPE

//! Returns the number of base-10 digits required to guarantee a lossless
//! binary->text->binary round-trip. For exact types, this function returns 0.
int max_digits10(DataType dtype);

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
  Nextafter,
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
  Xor,

  // generate complex from real and imaginary parts
  Complex
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

// Symbolic: Undetermined between Iteration or Broadcast
enum class IterType {
  Iteration,
  Reduction,
  Broadcast,
  Gather,
  Stride,
  GatherScatter,
  VectorComponent,
  Symbolic
};

// Used for Iteration Domain mapping modes in ComputeAtMap
enum class IdMappingMode {
  EXACT,
  ALMOSTEXACT,
  LOOP,
  PERMISSIVE,
  PERMISSIVE_RESIZE
};

static constexpr std::array<IdMappingMode, 5> kIdMappingModes = {
    IdMappingMode::EXACT,
    IdMappingMode::ALMOSTEXACT,
    IdMappingMode::LOOP,
    IdMappingMode::PERMISSIVE,
    IdMappingMode::PERMISSIVE_RESIZE};

// Used to annotate the special memory intrinsics that a loadstore
//  op will be lowered to.
enum class LoadStoreOpType {
  Set,
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
enum class Swizzle2DType { NoSwizzle = 0, ZShape, XOR, CyclicShift };

//! Modes of swizzle, see [Note on swizzle mode].
enum class SwizzleMode { NoSwizzle = 0, Data, Loop };

// Returns if function needs an f suffix on the operator when operating on a
// float value i.e. sin->sinf
bool needFloatSuffix(UnaryOpType t);
bool needFloatSuffix(BinaryOpType t);
bool needFloatSuffix(RNGOpType t);

ValType promoteType(const ValType& t1, const ValType& t2);

#define HANDLE_TYPE_PROMOTION(Type1, Type2)                              \
  if (t1 == NativeTypeToDataType<Type1>::type &&                         \
      t2 == NativeTypeToDataType<Type2>::type) {                         \
    return NativeTypeToDataType<std::common_type_t<Type1, Type2>>::type; \
  }

#define HANDLE_TYPE_PROMOTION1(Type1)                \
  HANDLE_TYPE_PROMOTION(Type1, float);               \
  HANDLE_TYPE_PROMOTION(Type1, double);              \
  HANDLE_TYPE_PROMOTION(Type1, int64_t);             \
  HANDLE_TYPE_PROMOTION(Type1, int);                 \
  HANDLE_TYPE_PROMOTION(Type1, bool);                \
  HANDLE_TYPE_PROMOTION(Type1, std::complex<float>); \
  HANDLE_TYPE_PROMOTION(Type1, std::complex<double>)

inline DataType promoteType(const DataType& t1, const DataType& t2) {
  if (t1 == t2) {
    return t1;
  }
  // pointer +- integer = pointer
  if (isPointerType(t1) && isIntegralType(t2)) {
    return t1;
  }
  if (isPointerType(t2) && isIntegralType(t1)) {
    return t2;
  }
  // When seeing DataType::Index, assuming we are computing index, so propagate
  // DataType::Index
  if ((t1 == DataType::Index && isIntegralType(t2)) ||
      (t2 == DataType::Index && isIntegralType(t1))) {
    return DataType::Index;
  }
  // Workaround a case where C++ and ATen have different type promotion rules
  if ((t1 == DataType::Double && t2 == DataType::ComplexFloat) ||
      (t2 == DataType::Double && t1 == DataType::ComplexFloat)) {
    // WARNING: ATen and C++ behave differently for this case. ATen returns
    // DataType::ComplexDouble but C++ returns DataType::ComplexFloat. Right now
    // we choose to be consistent with ATen.
    // TODO: I am pretty sure that for some cases we would need C++'s promotion
    // rule, for example, when we are simplifying scalar expressions, and for
    // other cases, we need ATen's promotion rule, for example, when we define
    // fusion from ATen graph. Fortunately, right now this is the only case to
    // worry about, and I don't think in practice, using ATen's rule would cause
    // any trouble.
    return DataType::ComplexDouble;
  }
  // Use C++ promotion rule when dtype has a native C++ type
  HANDLE_TYPE_PROMOTION1(float);
  HANDLE_TYPE_PROMOTION1(double);
  HANDLE_TYPE_PROMOTION1(int64_t);
  HANDLE_TYPE_PROMOTION1(int);
  HANDLE_TYPE_PROMOTION1(bool);
  HANDLE_TYPE_PROMOTION1(std::complex<float>);
  HANDLE_TYPE_PROMOTION1(std::complex<double>);
  // double + half/bfloat16 = double
  if ((t1 == DataType::Double && isFloatingPointType(t2)) ||
      (t2 == DataType::Double && isFloatingPointType(t1))) {
    return DataType::Double;
  }
  // float + half/bfloat16 = float
  // half + bfloat16 = float
  if (isFloatingPointType(t1) && isFloatingPointType(t2)) {
    return DataType::Float;
  }
  // complex + half/bfloat16 = complex
  if (isComplexType(t1)) {
    return t1;
  }
  if (isComplexType(t2)) {
    return t2;
  }
  // half + integers/bool = half
  // bfloat16 + integers/bool = bfloat16
  if (isFloatingPointType(t1)) {
    return t1;
  }
  if (isFloatingPointType(t2)) {
    return t2;
  }
  TORCH_CHECK(
      false, "Expected promotable DataTypes but got: ", t1, " and ", t2);
}

#undef HANDLE_TYPE_PROMOTION
#undef HANDLE_TYPE_PROMOTION1

template <typename... Args>
inline DataType promoteType(
    const DataType& t1,
    const DataType& t2,
    const Args&... args) {
  return promoteType(t1, promoteType(t2, promoteType(args...)));
}

inline DataType promoteType(const std::vector<DataType>& types) {
  TORCH_CHECK(types.size() > 0, "Can not promote empty type vector")
  DataType result = types.at(0);
  for (auto t : types) {
    result = promoteType(result, t);
  }
  return result;
}

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
TORCH_CUDA_CU_API const char* load_store_type2string(LoadStoreOpType t);

TORCH_CUDA_CU_API c10::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>&);

constexpr inline size_t primDataTypeSize(PrimDataType type) {
  switch (type) {
    case DataType::Bool:
      return sizeof(bool);
    case DataType::ComplexDouble:
      return sizeof(std::complex<double>);
    case DataType::ComplexFloat:
      return sizeof(std::complex<float>);
    case DataType::Double:
      return sizeof(double);
    case DataType::Float:
      return sizeof(float);
    case DataType::Half:
      return sizeof(at::Half);
    case DataType::BFloat16:
      return sizeof(at::BFloat16);
    case DataType::Index:
      TORCH_INTERNAL_ASSERT(
          false, "The actual type of Index is only known at compile time.");
    case DataType::Int:
      return sizeof(uint64_t);
    case DataType::Int32:
      return sizeof(uint32_t);
    case DataType::SMemAddress:
      return sizeof(unsigned);
    default:
      TORCH_INTERNAL_ASSERT(false, "Size undefined for data type.");
  }
}

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

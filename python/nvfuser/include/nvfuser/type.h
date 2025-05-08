// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <macros.h>
#include <visibility.h>

#include <c10/core/ScalarType.h>

#include <polymorphic_value.h>

#include <array>
#include <complex>
#include <cstdint>
#include <iostream>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_set>
#include <variant>

namespace nvfuser {

// Order of strength
enum class ValType {
  TensorDomain,
  IterDomain,
  TensorView,
  NamedScalar,
  Predicate,
  TensorIndex,
  Stream,
  Others
};

// Manual - The user provides the Bool value. Predicate generation is bypassed.
// Inline corresponds with PredicateCompute::getInlinePredicate
// Unswitch corresponds with UnswitchPredicate::get
// Misaligned - PredicateCompute::getInlinePredicate + Misaligned flag
// ReductionWrite - Same as Inline but without reduction axes
// LoopRotation - Predicate added by loop rotation, currently always true.
// ElectSync - Select a single thread to launch asynchronous operations.
enum class PredicateType {
  Manual,
  Inline,
  Unswitch,
  Vectorize,
  Misaligned,
  ReductionWrite,
  LoopRotation,
  ElectSync
};

// Index type is a convenience type that may be a 64 or 32 signed integer.
// This is helpful for math on indexing/size when we don't know what the index
// type might be. This allows us to prevent assuming the welford count must be
// int64_t which is relatively heavy to carry around. Index will be resolved
// at compile time with KernelIndexMode.
enum class PrimDataType {
  // Floating point types
  Double,
  Float,
  Half,
  BFloat16,
  Float8_e4m3fn,
  Float8_e5m2,
  // Integral types
  Char,
  Short,
  Int32,
  Int,
  Byte, // Following ATen convention
  UInt16, // Following ATen convention
  UInt32,
  UInt64,
  Index,
  // Boolean types
  Bool,
  // Complex types
  ComplexDouble,
  ComplexFloat,
  // Pointers
  SMemAddress,
  TMemAddress,
  // Null
  Null
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

struct DataType;

struct ArrayType {
  std::shared_ptr<DataType> type;
  size_t size;
  inline bool operator==(const ArrayType& other) const;
};

struct PointerType {
  std::shared_ptr<DataType> type;
  inline bool operator==(const PointerType& other) const;
};

struct StructType {
  std::string name;
  std::function<std::shared_ptr<Struct>()> create;

  struct FieldInfo {
    std::string name;
    std::shared_ptr<DataType> type;
    bool used_in_kernel = true;
  };

  std::vector<FieldInfo> fields;

  template <typename T>
  static StructType make(std::vector<FieldInfo> fields, std::string name = "") {
    static_assert(
        std::is_base_of<Struct, T>::value,
        "StructType::make only accepts Struct types");
    return StructType{
        .name = std::move(name),
        .create =
            []() {
              return std::static_pointer_cast<Struct>(std::make_shared<T>());
            },
        .fields = std::move(fields)};
  }

  inline const DataType& fieldDataType(const std::string& name) const {
    for (const auto& field : fields) {
      if (field.name == name) {
        return *field.type;
      }
    }
    NVF_THROW("Field ", name, " not found in struct ", this->name);
  }

  inline bool operator==(const StructType& other) const;
};

struct OpaqueType {
  std::string name;
  std::reference_wrapper<const std::type_info> type_info;
  size_t size;

  template <typename T>
  static OpaqueType make(std::string name = "") {
    return OpaqueType{
        .name = std::move(name), .type_info = typeid(T), .size = sizeof(T)};
  }

  inline bool operator==(const OpaqueType& other) const {
    return type_info.get() == other.type_info.get();
  }
};

struct DataType {
  using VariantOfSupportedTypes = std::
      variant<PrimDataType, ArrayType, PointerType, StructType, OpaqueType>;
  VariantOfSupportedTypes type = PrimDataType::Null;

  DataType() = default;
  DataType(VariantOfSupportedTypes type) : type(std::move(type)) {}
  DataType(PrimDataType type) : type(type) {}
  DataType(ArrayType type) : type(std::move(type)) {}
  DataType(PointerType type) : type(std::move(type)) {}
  DataType(StructType type) : type(std::move(type)) {}
  DataType(OpaqueType type) : type(std::move(type)) {}

  static constexpr PrimDataType Double = PrimDataType::Double;
  static constexpr PrimDataType Float = PrimDataType::Float;
  static constexpr PrimDataType Half = PrimDataType::Half;
  static constexpr PrimDataType Float8_e4m3fn = PrimDataType::Float8_e4m3fn;
  static constexpr PrimDataType Float8_e5m2 = PrimDataType::Float8_e5m2;
  static constexpr PrimDataType Index = PrimDataType::Index;
  static constexpr PrimDataType Char = PrimDataType::Char;
  static constexpr PrimDataType Short = PrimDataType::Short;
  static constexpr PrimDataType Int32 = PrimDataType::Int32;
  static constexpr PrimDataType Int = PrimDataType::Int;
  static constexpr PrimDataType Byte = PrimDataType::Byte;
  static constexpr PrimDataType UInt16 = PrimDataType::UInt16;
  static constexpr PrimDataType UInt32 = PrimDataType::UInt32;
  static constexpr PrimDataType UInt64 = PrimDataType::UInt64;
  static constexpr PrimDataType Bool = PrimDataType::Bool;
  static constexpr PrimDataType BFloat16 = PrimDataType::BFloat16;
  static constexpr PrimDataType ComplexFloat = PrimDataType::ComplexFloat;
  static constexpr PrimDataType ComplexDouble = PrimDataType::ComplexDouble;
  static constexpr PrimDataType SMemAddress = PrimDataType::SMemAddress;
  static constexpr PrimDataType TMemAddress = PrimDataType::TMemAddress;
  static constexpr PrimDataType Null = PrimDataType::Null;
};

inline bool operator==(const DataType& lhs, const DataType& rhs) {
  return lhs.type == rhs.type;
}

inline bool operator!=(const DataType& lhs, const DataType& rhs) {
  return !operator==(lhs, rhs);
}

bool ArrayType::operator==(const ArrayType& other) const {
  return *type == *other.type && size == other.size;
}

bool PointerType::operator==(const PointerType& other) const {
  return *type == *other.type;
}

bool StructType::operator==(const StructType& other) const {
  if (fields.size() != other.fields.size()) {
    return false;
  }
  for (auto i : std::ranges::iota_view(0u, fields.size())) {
    if (fields[i].name != other.fields[i].name ||
        *fields[i].type != *other.fields[i].type ||
        fields[i].used_in_kernel != other.fields[i].used_in_kernel) {
      return false;
    }
  }
  return true;
}

inline StructType StructHandle::type() const {
  return struct_ptr_->type();
}

StructType globalTensorMetaData(
    const PrimDataType& dtype,
    size_t dim,
    size_t alloc_dim);

inline StructType globalTensorMetaData(const PrimDataType& dtype, size_t dim) {
  return globalTensorMetaData(dtype, dim, dim);
}

class Val;
//! Get the type of a Val's metadata, currently only supporting tensors
NVF_API DataType metaDataTypeOf(const Val* tv);

enum class KernelIndexMode { INT32, INT64 };

PrimDataType indexModeToDtype(KernelIndexMode index_mode);
KernelIndexMode indexTypeToMode(DataType index_type);

// check if type preserves all information from base_type. Which indicates a
// cast from base_type -> type -> base_type should be bit-wise identical
bool isInclusiveType(const DataType& base_type, const DataType& type);

// Returns if the datatype is a floating point type
inline bool isFloatingPointType(DataType dtype) {
  return dtype == DataType::Double || dtype == DataType::Float ||
      dtype == DataType::Half || dtype == DataType::BFloat16 ||
      dtype == DataType::Float8_e4m3fn || dtype == DataType::Float8_e5m2;
}

// Returns if the datatype is an integer type
inline bool isIntegralType(DataType dtype) {
  return std::visit(
      [](auto&& dtype) {
        using T = std::decay_t<decltype(dtype)>;
        if constexpr (std::is_same_v<T, PrimDataType>) {
          switch (dtype) {
            case DataType::Index:
            case DataType::Char:
            case DataType::Short:
            case DataType::Int:
            case DataType::Int32:
            case DataType::Byte:
            case DataType::UInt16:
            case DataType::UInt32:
            case DataType::UInt64:
              return true;
            default:
              return false;
          }
        }
        return false;
      },
      dtype.type);
}

// Returns if the datatype is an unsigned integer type
inline bool isUnsignedIntegralType(DataType dtype) {
  return dtype == DataType::Byte || dtype == DataType::UInt16 ||
      dtype == DataType::UInt32 || dtype == DataType::UInt64;
}

// Returns if the datatype is a pointer type
inline bool isPointerType(DataType dtype) {
  return std::holds_alternative<PointerType>(dtype.type) ||
      dtype == DataType::SMemAddress || dtype == DataType::TMemAddress;
}

// Returns if the datatype is an integer or pointer type
inline bool isIntegralOrPointerType(DataType dtype) {
  return isIntegralType(dtype) || isPointerType(dtype);
}

// Returns if the datatype is a boolean type
inline bool isBooleanType(DataType dtype) {
  return dtype == DataType::Bool;
}

// Returns if the datatype is a complex type
inline bool isComplexType(DataType dtype) {
  return dtype == DataType::ComplexFloat || dtype == DataType::ComplexDouble;
}

// Returns if the datatype is a complex type
inline bool isStructType(DataType dtype) {
  return std::holds_alternative<StructType>(dtype.type);
}

// Return the corresponding scalar of a complex type
DataType getTypeFromComplexType(DataType dtype);
// Return the corresponding complex type of a scalar
DataType getComplexTypeFromType(DataType dtype);
// Return if the datatype is supported on the current device
NVF_API bool isSupportedTypeByDevice(DataType dtype);

NVF_API int64_t dataTypeSize(DataType type);

// If the index type is known it will be automatically used here
int64_t dataTypeSize(DataType type, DataType index_type);

template <PrimDataType DT>
struct DataTypeToNativeType;

template <PrimDataType DT>
struct DataTypeToAtenType;

template <typename NativeType>
struct NativeTypeToDataType;

template <at::ScalarType aten_type>
struct AtenTypeToDataType;

template <at::ScalarType aten_type>
struct AtenTypeToNativeType;

template <typename NativeType>
struct IsPrimitiveNativeType : std::false_type {};

#define DEFINE_DATATYPE_TO_NATIVE_TYPE(data_type, native_type) \
  template <>                                                  \
  struct DataTypeToNativeType<data_type> {                     \
    using type = native_type;                                  \
  };                                                           \
  template <>                                                  \
  struct NativeTypeToDataType<native_type> {                   \
    static constexpr PrimDataType type = data_type;            \
  };                                                           \
  template <>                                                  \
  struct IsPrimitiveNativeType<native_type> : std::true_type {}

#define DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(          \
    data_type, at_type, native_type)                      \
  DEFINE_DATATYPE_TO_NATIVE_TYPE(data_type, native_type); \
  template <>                                             \
  struct AtenTypeToDataType<at_type> {                    \
    static constexpr PrimDataType type = data_type;       \
  };                                                      \
  template <>                                             \
  struct AtenTypeToNativeType<at_type> {                  \
    using type = native_type;                             \
  }

DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Float,
    at::ScalarType::Float,
    float);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Double,
    at::ScalarType::Double,
    double);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Half,
    at::ScalarType::Half,
    at::Half);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::BFloat16,
    at::ScalarType::BFloat16,
    at::BFloat16);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Float8_e4m3fn,
    at::ScalarType::Float8_e4m3fn,
    at::Float8_e4m3fn);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Float8_e5m2,
    at::ScalarType::Float8_e5m2,
    at::Float8_e5m2);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Char,
    at::ScalarType::Char,
    int8_t);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Short,
    at::ScalarType::Short,
    int16_t);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Int32,
    at::ScalarType::Int,
    int);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Int,
    at::ScalarType::Long,
    int64_t);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Byte,
    at::ScalarType::Byte,
    uint8_t);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::UInt16,
    at::ScalarType::UInt16,
    uint16_t);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::UInt32,
    at::ScalarType::UInt32,
    uint32_t);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::UInt64,
    at::ScalarType::UInt64,
    uint64_t);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::Bool,
    at::ScalarType::Bool,
    bool);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::ComplexFloat,
    at::ScalarType::ComplexFloat,
    std::complex<float>);
DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE(
    DataType::ComplexDouble,
    at::ScalarType::ComplexDouble,
    std::complex<double>);

#undef DEFINE_DATATYPE_TO_NATIVE_TYPE
#undef DEFINE_DATATYPE_TO_ATEN_AND_NATIVE_TYPE

inline DataType getDataType(const PolymorphicValue& value) {
  std::optional<DataType> dtype = std::nullopt;
  PolymorphicValue::for_all_types([&value, &dtype](auto _) {
    using T = typename decltype(_)::type;
    if constexpr (IsPrimitiveNativeType<T>::value) {
      if (value.is<T>()) {
        dtype = NativeTypeToDataType<T>::type;
      }
    } else if constexpr (std::is_same_v<T, std::vector<PolymorphicValue>>) {
      if (value.is<T>()) {
        const auto& vec = value.as<T>();
        size_t size = vec.size();
        NVF_CHECK(size > 0, "Empty array is not supported");
        dtype =
            ArrayType{std::make_shared<DataType>(getDataType(vec[0])), size};
      }
    } else if constexpr (std::is_same_v<T, Pointer>) {
      // For pointers in polymorphic value, we only store the data size of the
      // pointee, so it is impossible to infer the pointer type.
      NVF_CHECK(!value.is<T>(), "Can not infer pointer type.");
    } else if constexpr (std::is_same_v<T, StructHandle>) {
      if (value.is<T>()) {
        dtype = value.as<T>().type();
      }
    } else if constexpr (std::is_same_v<T, Opaque>) {
      if (value.is<T>()) {
        const auto& opaque = value.as<T>();
        dtype = DataType(OpaqueType{
            .type_info = opaque.any().type(), .size = opaque.size()});
      }
    }
  });
  NVF_CHECK(dtype.has_value(), "Unknown dtype for ", value.type().name());
  return dtype.value();
}

inline bool isCompatibleDataType(DataType dtype, DataType dtype2) {
  if (dtype == dtype2) {
    return true;
  }
  if (isIntegralType(dtype) && isIntegralType(dtype2)) {
    return true;
  }
  if (isFloatingPointType(dtype) && isFloatingPointType(dtype2)) {
    return true;
  }
  if (isComplexType(dtype) && isComplexType(dtype2)) {
    return true;
  }
  if (std::holds_alternative<ArrayType>(dtype.type) &&
      std::holds_alternative<ArrayType>(dtype2.type)) {
    const auto& array_type = std::get<ArrayType>(dtype.type);
    const auto& array_type2 = std::get<ArrayType>(dtype2.type);
    return array_type.size == array_type2.size &&
        isCompatibleDataType(*array_type.type, *array_type2.type);
  }
  if (std::holds_alternative<StructType>(dtype.type) &&
      std::holds_alternative<StructType>(dtype2.type)) {
    const auto& struct_type = std::get<StructType>(dtype.type);
    const auto& struct_type2 = std::get<StructType>(dtype2.type);
    if (struct_type.fields.size() != struct_type2.fields.size()) {
      return false;
    }
    for (auto i : std::ranges::iota_view(0u, struct_type.fields.size())) {
      if (struct_type.fields[i].name != struct_type2.fields[i].name ||
          !isCompatibleDataType(
              *struct_type.fields[i].type, *struct_type2.fields[i].type)) {
        return false;
      }
    }
    return true;
  }
  if (std::holds_alternative<OpaqueType>(dtype.type) &&
      std::holds_alternative<OpaqueType>(dtype2.type)) {
    const auto& opaque_type = std::get<OpaqueType>(dtype.type);
    const auto& opaque_type2 = std::get<OpaqueType>(dtype2.type);
    return opaque_type.type_info.get() == opaque_type2.type_info.get();
  }
  return false;
}

inline bool hasCompatibleDataType(
    const PolymorphicValue& value,
    DataType dtype) {
  // We can not always completely infer data type from value, so we need some
  // special handling here.
  if (std::holds_alternative<PointerType>(dtype.type)) {
    if (!value.is<Pointer>()) {
      return false;
    }
    auto ptr = std::get<PointerType>(dtype.type);
    return dataTypeSize(*ptr.type) == value.as<Pointer>().size();
  } else if (std::holds_alternative<ArrayType>(dtype.type)) {
    if (!value.is<std::vector>()) {
      return false;
    }
    const auto& array_type = std::get<ArrayType>(dtype.type);
    if (array_type.size != value.as<std::vector>().size()) {
      return false;
    }
    if (array_type.size == 0) {
      return true;
    }
  }
  return isCompatibleDataType(getDataType(value), dtype);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

//! Returns the number of base-10 digits required to guarantee a lossless
//! binary->text->binary round-trip. For exact types, this function returns 0.
int max_digits10(DataType dtype);

enum class UnaryOpType {
  Cast,
  BitCast,
  RefCast,

  Abs,
  Acos,
  Acosh,
  Address,
  Asin,
  Asinh,
  Atan,
  Atanh,
  Ceil,
  Cos,
  Cosh,
  Dereference,
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
  Neg,
  Real,
  Reciprocal,
  Relu,
  Rsqrt,
  Round,
  Sigmoid,
  Signbit,
  Sin,
  Sinh,
  Sqrt,
  Tan,
  Tanh,
  Trunc,
  BitCeil,

  // Tools to help debugging
  Print,

  // Logical and bitwise negation
  LogicalNot,
  BitwiseNot,

  // Operators returning boolean values
  IsFinite,
  IsInf,
  IsNan,
  IsNegInf,
  IsPosInf,
  IsReal,

  // Special unary ops
  ElectSync,
  ToUnsignedSmemAddr,
  AdjustPartialLdMatrixAddrInTuring8,
  AdjustPartialLdMatrixAddrInTuring16
};

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

  // Integer output ops.
  Mod,
  CeilDiv,
  Lshift,
  Rshift,
  Gcd,

  // Bitwise Ops
  // These always return integers, as if each arg is first cast to int
  //  If changing modify isIntegerOp.
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,

  // Logical Ops
  // Int operations, leave position of Mod as first logical op see
  // isLogicalOp(BinaryOpType bopt)
  Eq,
  GE,
  GT,
  LE,
  LT,
  NE,

  // These ops compare as if each arg is first cast to bool
  LogicalAnd,
  LogicalOr,

  // generate complex from real and imaginary parts
  Complex
};

enum class ScatterOpType { Set };

enum class RNGOpType {
  Uniform, // Uniform in [0, 1)
  UniformRange, // Uniform in [low, high]
  NormalStandard, // Normal with mean 0, std 1
  NormalGeneral, // Normal with given mean and std
  Undefined,
};

// Return if output of operator should be a boolean
bool isIntegerOp(const BinaryOpType bopt);

// Return if output of operator should be a boolean
bool isLogicalOp(const BinaryOpType bopt);

enum class TernaryOpType { Clamp, Lerp, Threshold, Where, Philox };

enum class ParallelType {
  DIDx,
  DIDy,
  DIDz,
  BIDz,
  BIDy,
  BIDx,
  TIDz,
  TIDy,
  TIDx,
  Stream,
  Vectorize,
  Unroll,
  Unswitch,
  Mma,
  Group,
  Bulk,
  Serial
};

std::unordered_set<ParallelType> allParallelTypesExcept(
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

static constexpr std::array<ParallelType, 3> kParallelTypeDIDs = {
    ParallelType::DIDx,
    ParallelType::DIDy,
    ParallelType::DIDz};

enum class MemoryType { Local, Shared, Global, Tensor };

// Symbolic: Undetermined between Iteration or Broadcast
enum class IterType {
  Iteration,
  Reduction,
  Broadcast,
  Stride,
  GatherScatter,
  VectorComponent,
  Symbolic
};

// Used for Iteration Domain mapping modes in ComputeAtMap
enum class IdMappingMode {
  EXACT,
  ALMOSTEXACT,
  BROADCAST,
  PERMISSIVE,
  LOOP,
  // TODO: Reconsider if this graph is really necessary
  PERMISSIVE_RESIZE,
  // TODO: Reconsider if this graph is really necessary
  INNERMOST
};

static constexpr std::array<IdMappingMode, 7> kIdMappingModes = {
    IdMappingMode::EXACT,
    IdMappingMode::ALMOSTEXACT,
    IdMappingMode::BROADCAST,
    IdMappingMode::PERMISSIVE,
    IdMappingMode::LOOP,
    IdMappingMode::PERMISSIVE_RESIZE,
    IdMappingMode::INNERMOST};

// See
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
// for what each option means. Will also consider .L1::no_allocate because .cs
// still pollutes cache to some extent.
enum class CacheOp {
  Unspecified, // Opt in for the default cache operator or when the LoadStoreOp
               // doesn't take a cache operator.
  AllLevels,
  Streaming,
  Global,
};

//! Used to annotate the special memory intrinsics that a loadstore op will be
//!  lowered to.
//!
//!  SegmenterSet here is used to hint segmenter to break kernel on the output
//!  of the node
enum class LoadStoreOpType {
  Set,
  SegmenterSet,
  LdMatrix,
  CpAsync,
  CpAsyncBulk,
  CpAsyncBulkTensorTile,
  StMatrix,
  LdTMem,
  StTMem
};

// Used to label what part of the circular buffered iterdomain
//  a for loop is materializing.
enum class CircularBufferLoopStage {
  Prolog = 0,
  Main,
  Epilog,
  AsyncWarp,
  ComputeWarp,
  EndOfStages, // A special placeholder used to iterate over all stages
  NotApplicable
};

// The circular buffer load expressions are cloned for these circular buffer
// loop types.
// e.g., No additional loads are required for the Epilogue stage.
inline bool hasCircularBufferLoad(CircularBufferLoopStage stage) {
  return stage == CircularBufferLoopStage::Prolog ||
      stage == CircularBufferLoopStage::Main ||
      stage == CircularBufferLoopStage::AsyncWarp;
}

// The consuming expressions of circular buffer are cloned for these circular
// buffer loop types.
// e.g., No actual computation occurs in the Prologue stage.
inline bool hasCircularBufferConsume(CircularBufferLoopStage stage) {
  return stage == CircularBufferLoopStage::Main ||
      stage == CircularBufferLoopStage::Epilog ||
      stage == CircularBufferLoopStage::ComputeWarp;
}

// A loop type may have WAR hazard if any of the following is true:
// - The load *in this loop type* may overwrite a buffer being read by a
//   compute somewhere (*may or may not be in this loop*)
// - The compute *in this loop type* reads circular buffer TVs that, if not
//   properly handled, could be overwriten by a circular buffer loading
//   somewhere (*may or may not be in this loop*)
inline bool mayHaveWarHazard(CircularBufferLoopStage stage) {
  return stage == CircularBufferLoopStage::Main ||
      stage == CircularBufferLoopStage::AsyncWarp ||
      stage == CircularBufferLoopStage::ComputeWarp;
}

//! Supported swizzle types,
//!  corresponds to swizzles functions on the runtime cuda
//!  naming it swizzle_2d to reserve the options to have a swizzle_1d.
//!
//!  TODO: unify with existing swizzle logic, currently
//!    doesn't have the same type.
enum class SwizzleType { NoSwizzle = 0, XOR, CyclicShift };
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
  NVF_CHECK(false, "Expected promotable DataTypes but got: ", t1, " and ", t2);
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
  NVF_CHECK(!types.empty(), "Can not promote empty type vector")
  DataType result = types.at(0);
  for (const auto& t : types) {
    result = promoteType(result, t);
  }
  return result;
}

// If type cannot be found (i.e. codegen does not support provided type) returns
// DataType::Null
NVF_API DataType aten_to_data_type(const at::ScalarType& scalar_type);
NVF_API at::ScalarType data_type_to_aten(const DataType& data_type);

NVF_API std::ostream& operator<<(std::ostream&, const ValType);
std::ostream& operator<<(std::ostream&, const PredicateType);
NVF_API std::ostream& operator<<(std::ostream&, const DataType);
std::ostream& operator<<(std::ostream&, const UnaryOpType);
NVF_API std::ostream& operator<<(std::ostream&, const BinaryOpType);
std::ostream& operator<<(std::ostream&, const TernaryOpType);
std::ostream& operator<<(std::ostream&, const ScatterOpType);
std::ostream& operator<<(std::ostream&, const RNGOpType);
NVF_API std::ostream& operator<<(std::ostream&, const ParallelType);
NVF_API std::ostream& operator<<(std::ostream&, const MemoryType);
NVF_API std::ostream& operator<<(std::ostream&, const IterType);
std::ostream& operator<<(std::ostream&, const IdMappingMode);
NVF_API std::ostream& operator<<(std::ostream&, const LoadStoreOpType);
std::ostream& operator<<(std::ostream&, const CircularBufferLoopStage);
std::ostream& operator<<(std::ostream&, const SwizzleType&);
std::ostream& operator<<(std::ostream&, const Swizzle2DType&);
std::ostream& operator<<(std::ostream&, const SwizzleMode&);
std::ostream& operator<<(std::ostream&, const KernelIndexMode&);
NVF_API std::ostream& operator<<(std::ostream&, const CacheOp&);
std::ostream& operator<<(std::ostream& os, const std::optional<bool>&);

std::string stringifyThreadSize(const ParallelType);
std::string stringifyThread(const ParallelType);
std::string typePrefix(const DataType);

// TODO: ThreadDim should be BlockDim and BlockDim should be GridDim
// Returns if parallel type is TID[x, y, z]
NVF_API bool isParallelTypeThreadDim(ParallelType);
// Returns if parallel type is BID[x, y, z]
NVF_API bool isParallelTypeBlockDim(ParallelType);
// Returns if parallel type is a grid or block parallelization dimension
NVF_API bool isParallelTypeThread(ParallelType);
// Returns if parallel type is DIDx
NVF_API bool isParallelTypeDeviceDim(ParallelType);

NVF_API bool isParallelTypeVectorize(ParallelType);

std::optional<std::string> inline_op_str(const UnaryOpType);
std::optional<std::string> inline_op_str(const BinaryOpType);
std::optional<std::string> inline_op_str(const RNGOpType);
std::optional<std::string> integer_op_str(const BinaryOpType);
std::optional<std::string> bool_op_str(const BinaryOpType);
const char* predicate_type2string(PredicateType t);
const char* load_store_type2string(LoadStoreOpType t);

std::optional<std::string> cast_func_str(const std::pair<DataType, DataType>&);

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
    case DataType::Float8_e4m3fn:
      return sizeof(at::Float8_e4m3fn);
    case DataType::Float8_e5m2:
      return sizeof(at::Float8_e5m2);
    case DataType::Index:
      NVF_THROW("The actual type of Index is only known at compile time.");
    case DataType::Char:
      return sizeof(int8_t);
    case DataType::Short:
      return sizeof(int16_t);
    case DataType::Int32:
      return sizeof(int32_t);
    case DataType::Int:
      return sizeof(int64_t);
    case DataType::Byte:
      return sizeof(uint8_t);
    case DataType::UInt16:
      return sizeof(uint16_t);
    case DataType::UInt32:
    case DataType::SMemAddress:
    case DataType::TMemAddress:
      return sizeof(uint32_t);
    case DataType::UInt64:
      return sizeof(uint64_t);
    default:
      NVF_THROW("Size undefined for data type.");
  }
}

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

Pointer::Pointer(void* ptr, DataType dtype)
    : ptr_(reinterpret_cast<std::byte*>(ptr)), size_(dataTypeSize(dtype)) {}

inline PolymorphicValue castToDtype(
    PolymorphicValue value,
    const DataType& dtype) {
  if (!value.hasValue()) {
    return value;
  }
  // Cast the given value to the given data type. This enables interface
  // like: IrBuilder::create<Val>(0, DataType::Double) where value is
  // an integer but the desired data type is double.
  if (!hasCompatibleDataType(value, dtype)) {
    PolymorphicValue::for_all_types([&](auto _) {
      using T = typename decltype(_)::type;
      if constexpr (IsPrimitiveNativeType<T>::value) {
        if (isCompatibleDataType(NativeTypeToDataType<T>::type, dtype)) {
          value = PolymorphicValue(static_cast<T>(value));
        }
      }
      // TODO: support arrays and pointers
    });
  }
  return value;
}

// Converts an enum to its underlying type.
// It corresponds with std::to_underlying introduced in c++23
// https://en.cppreference.com/w/cpp/utility/to_underlying
template <typename E>
constexpr auto toUnderlying(E e) noexcept {
  return static_cast<std::underlying_type_t<E>>(e);
}

enum class AsyncOpType { NotAsync, CpAsync, CpAsyncBulk, WgMma };

// Data path between TMem and register file. Tensor memory is not a general
// byte-addressable memory like other memory types. The register <-> TMem
// data transfer must follow one of the following specific patterns which has
// well-defined specification about which thread's which register access to
// which part of TMem. See:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-layout
enum class TMemRegisterDataPath {
  Path32x32b,
  Path16x64b,
  Path16x128b,
  Path16x256b,
  Path16x32bx2,
};

std::ostream& operator<<(std::ostream&, TMemRegisterDataPath);

} // namespace nvfuser

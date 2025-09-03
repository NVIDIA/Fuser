// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <type.h>

#include <ATen/cuda/CUDAContext.h>

#include <sstream>
#include <unordered_set>

#include <ir/all_nodes.h>
#include <tensor_metadata.h>

namespace nvfuser {

StructType NotImplementedStruct::type() const {
  NVF_THROW("Not implemented");
}

StructType globalTensorMetaData(
    const DataType& dtype,
    size_t dim,
    size_t alloc_dim) {
  std::stringstream ss;
  ss << "Tensor<" << dtype << ", " << dim << ", " << alloc_dim << ">";

  StructType::FieldInfo data_field;
  data_field.name = "data";
  data_field.type = std::make_shared<DataType>(
      PointerType{std::make_shared<DataType>(dtype)});
  data_field.used_in_kernel = true;

  StructType::FieldInfo logical_size_field;
  logical_size_field.name = "logical_size";
  logical_size_field.type = std::make_shared<DataType>(
      ArrayType{std::make_shared<DataType>(DataType::Index), dim});
  logical_size_field.used_in_kernel = true;

  StructType::FieldInfo logical_stride_field;
  logical_stride_field.name = "logical_stride";
  logical_stride_field.type = std::make_shared<DataType>(
      ArrayType{std::make_shared<DataType>(DataType::Index), dim});
  logical_stride_field.used_in_kernel = false;

  StructType::FieldInfo alloc_size_field;
  alloc_size_field.name = "alloc_size";
  alloc_size_field.type = std::make_shared<DataType>(
      ArrayType{std::make_shared<DataType>(DataType::Index), alloc_dim});
  alloc_size_field.used_in_kernel = false;

  StructType::FieldInfo alloc_stride_field;
  alloc_stride_field.name = "alloc_stride";
  alloc_stride_field.type = std::make_shared<DataType>(
      ArrayType{std::make_shared<DataType>(DataType::Index), alloc_dim});
  alloc_stride_field.used_in_kernel = true;

  return StructType::make<TensorMetaData>(
      {data_field,
       logical_size_field,
       logical_stride_field,
       alloc_size_field,
       alloc_stride_field},
      ss.str());
}

DataType metaDataTypeOf(const Val* v) {
  auto tv = dynamic_cast<const TensorView*>(v);
  NVF_ERROR(
      tv != nullptr, "Currently, only supports getting metadata of TensorView");
  if (tv->getMemoryType() == MemoryType::Shared) {
    // Smem tensor is defined locally as a pointer
    return PointerType{std::make_shared<DataType>(tv->dtype())};
  }

  size_t dim = TensorDomain::noReductions(tv->getLogicalDomain()).size();
  size_t alloc_dim =
      TensorDomain::noReductions(tv->getMaybeAllocationDomain()).size();
  return globalTensorMetaData(tv->dtype().type, dim, alloc_dim);
}

PrimDataType indexModeToDtype(KernelIndexMode index_mode) {
  switch (index_mode) {
    case KernelIndexMode::INT32:
      return DataType::Int32;
    case KernelIndexMode::INT64:
      return DataType::Int;
  }
  std::unreachable();
}

KernelIndexMode indexTypeToMode(DataType index_type) {
  return index_type == indexModeToDtype(KernelIndexMode::INT32)
      ? KernelIndexMode::INT32
      : KernelIndexMode::INT64;
}

bool isInclusiveType(const DataType& base_type, const DataType& wider_type) {
  if (base_type == wider_type) {
    return true;
  }
  if (base_type == DataType::Bool) {
    return true;
  }
  if ((wider_type == DataType::Double ||
       wider_type == DataType::ComplexDouble) &&
      (base_type == DataType::Double || base_type == DataType::Float ||
       base_type == DataType::Half || base_type == DataType::BFloat16 ||
       base_type == DataType::Float8_e4m3fn ||
       base_type == DataType::Float8_e5m2 ||
       base_type == DataType::Float8_e8m0fnu ||
       base_type == DataType::Float4_e2m1fn)) {
    return true;
  }
  if ((wider_type == DataType::Float || wider_type == DataType::ComplexFloat) &&
      (base_type == DataType::Float || base_type == DataType::Half ||
       base_type == DataType::BFloat16 ||
       base_type == DataType::Float8_e4m3fn ||
       base_type == DataType::Float8_e5m2 ||
       base_type == DataType::Float8_e8m0fnu ||
       base_type == DataType::Float4_e2m1fn)) {
    return true;
  }
  if ((wider_type == DataType::Half || wider_type == DataType::BFloat16) &&
      (base_type == DataType::Float8_e4m3fn ||
       base_type == DataType::Float8_e5m2 ||
       base_type == DataType::Float4_e2m1fn)) {
    return true;
  }
  if (wider_type == DataType::BFloat16 &&
      (base_type == DataType::Float8_e8m0fnu ||
       base_type == DataType::Float4_e2m1fn)) {
    return true;
  }
  if (wider_type == DataType::Float8_e4m3fn &&
      (base_type == DataType::Float4_e2m1fn)) {
    return true;
  }
  if (wider_type == DataType::Float8_e5m2 &&
      (base_type == DataType::Float4_e2m1fn)) {
    return true;
  }
  if ((wider_type == DataType::Int || wider_type == DataType::Double ||
       wider_type == DataType::ComplexDouble) &&
      base_type == DataType::Int32) {
    return true;
  }
  if (wider_type == DataType::ComplexDouble &&
      base_type == DataType::ComplexFloat) {
    return true;
  }
  return false;
}

DataType getTypeFromComplexType(DataType dtype) {
  switch (std::get<PrimDataType>(dtype.type)) {
    case DataType::ComplexFloat:
      return DataType::Float;
    case DataType::ComplexDouble:
      return DataType::Double;
    default:
      NVF_THROW(
          "Only support ComplexFloat and ComplexDouble, current type:", dtype);
  }
}

DataType getComplexTypeFromType(DataType dtype) {
  switch (std::get<PrimDataType>(dtype.type)) {
    case DataType::Float:
    case DataType::ComplexFloat:
      return DataType::ComplexFloat;
    case DataType::Double:
    case DataType::ComplexDouble:
      return DataType::ComplexDouble;
    default:
      NVF_THROW("Only support Float and Double, current type:", dtype);
  }
}

bool isSupportedTypeByDevice(DataType dtype) {
  auto prop = at::cuda::getCurrentDeviceProperties();
  auto major_ver = prop->major;
  if (dtype == DataType::BFloat16) {
    return major_ver >= 8;
  }
  if (dtype == DataType::Float8_e4m3fn || dtype == DataType::Float8_e5m2) {
    return major_ver >= 9;
  }
  if (dtype == DataType::Float8_e8m0fnu || dtype == DataType::Float4_e2m1fn ||
      dtype == DataType::Float4_e2m1fn_x2) {
    return major_ver >= 10;
  }
  return true;
}

bool isIntegerOp(const BinaryOpType bopt) {
  return bopt >= BinaryOpType::Mod && bopt <= BinaryOpType::BitwiseXor;
}

bool isLogicalOp(const BinaryOpType bopt) {
  return bopt >= BinaryOpType::Eq && bopt <= BinaryOpType::LogicalOr;
}

// Return highest on list (smallest enum val)
ValType promoteType(const ValType& t1, const ValType& t2) {
  if (t1 == ValType::TensorView || t2 == ValType::TensorView) {
    return ValType::TensorView;
  }
  if (t1 == ValType::Others &&
      (t2 == ValType::Others || t2 == ValType::NamedScalar)) {
    return ValType::Others;
  }
  if (t2 == ValType::Others &&
      (t1 == ValType::Others || t1 == ValType::NamedScalar)) {
    return ValType::Others;
  }
  if (t1 == ValType::NamedScalar && t2 == ValType::NamedScalar) {
    return ValType::Others;
  }
  NVF_CHECK(false, "Expected promotable ValTypes but got: ", t1, " and ", t2);
}

static std::string data_type2string(DataType t) {
  return std::visit(
      [](auto&& dtype) -> std::string {
        using T = std::decay_t<decltype(dtype)>;
        if constexpr (std::is_same_v<T, PrimDataType>) {
          switch (dtype) {
            case DataType::Null:
              // This is not a real C++ type, but being able to print a string
              // for it is convenient for debugging.
              return "null_type";
            case DataType::Bool:
              return "bool";
            case DataType::Double:
              return "double";
            case DataType::Float:
              return "float";
            case DataType::Half:
              return "__half";
            case DataType::BFloat16:
              return "__bfloat";
            case DataType::Float8_e4m3fn:
              return "__e4m3";
            case DataType::Float8_e5m2:
              return "__e5m2";
            case DataType::Float8_e8m0fnu:
              return "__e8m0";
            case DataType::Float4_e2m1fn:
              return "__e2m1";
            case DataType::Float4_e2m1fn_x2:
              return "__e2m1_x2";
            case DataType::Index:
              return "nvfuser_index_t";
            case DataType::Char:
              return "int8_t";
            case DataType::Short:
              return "int16_t";
            case DataType::Int32:
              return "int";
            case DataType::Int:
              return "int64_t";
            case DataType::Byte:
              return "uint8_t";
            case DataType::UInt16:
              return "uint16_t";
            case DataType::UInt32:
            case DataType::SMemAddress:
            case DataType::TMemAddress:
              return "uint32_t";
            case DataType::UInt64:
              return "uint64_t";
            case DataType::ComplexFloat:
              return "std::complex<float>";
            case DataType::ComplexDouble:
              return "std::complex<double>";
            default:
              NVF_THROW("No string found for data type.");
          }
        } else if constexpr (std::is_same_v<T, PointerType>) {
          return data_type2string(*dtype.type) + "*";
        } else if constexpr (std::is_same_v<T, ArrayType>) {
          std::stringstream ss;
          ss << "Array<" << data_type2string(*dtype.type) << ", " << dtype.size
             << ", 1>";
          return ss.str();
        } else if constexpr (std::is_same_v<T, StructType>) {
          if (dtype.name != "") {
            return dtype.name;
          }
          std::stringstream ss;
          ss << "struct { ";
          for (const auto& field : dtype.fields) {
            ss << data_type2string(*field.type) << " " << field.name << "; ";
          }
          ss << "}";
          return ss.str();
        } else if constexpr (std::is_same_v<T, OpaqueType>) {
          if (dtype.name != "") {
            return dtype.name;
          } else {
            return dtype.type_info.get().name();
          }
        } else {
          NVF_THROW("No string found for data type.");
        }
        NVF_THROW("No string found for data type.");
      },
      t.type);
}

static const char* val_type2string(ValType t) {
  switch (t) {
    case ValType::TensorView:
      return "TensorView";
    case ValType::TensorDomain:
      return "TensorDomain";
    case ValType::IterDomain:
      return "IterDomain";
    case ValType::Others:
      return "Scalar";
    case ValType::NamedScalar:
      return "NamedScalar";
    case ValType::Predicate:
      return "Predicate";
    case ValType::TensorIndex:
      return "TensorIndex";
    case ValType::Stream:
      return "Stream";
  }
  std::unreachable();
}

const char* block_sf_layout2string(BlockScalingFactorLayout t) {
  switch (t) {
    case BlockScalingFactorLayout::Block128x4:
      return "block_128_4";
    default:
      NVF_THROW("No string found for layout.");
  }
}

const char* predicate_type2string(PredicateType t) {
  switch (t) {
    case PredicateType::Manual:
      return "Manual";
    case PredicateType::Inline:
      return "Inline";
    case PredicateType::Unswitch:
      return "Unswitch";
    case PredicateType::Vectorize:
      return "Vectorize";
    case PredicateType::Misaligned:
      return "Misaligned";
    case PredicateType::ReductionWrite:
      return "ReductionWrite";
    case PredicateType::LoopRotation:
      return "LoopRotation";
    case PredicateType::ElectSync:
      return "ElectSync";
    case PredicateType::OneDimTmaLoadExpectArrive:
      return "OneDimTmaLoadExpectArrive";
    case PredicateType::OneDimTmaWaitParity:
      return "OneDimTmaWaitParity";
  }
  std::unreachable();
}

bool needFloatSuffix(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Abs:
    case UnaryOpType::Cast:
    case UnaryOpType::Frac:
    case UnaryOpType::Gelu:
    case UnaryOpType::Imag:
    case UnaryOpType::Silu:
    case UnaryOpType::BitCast:
    case UnaryOpType::Dereference:
    case UnaryOpType::Neg:
    case UnaryOpType::BitwiseNot:
    case UnaryOpType::LogicalNot:
    case UnaryOpType::Real:
    case UnaryOpType::Relu:
    case UnaryOpType::Reciprocal:
    case UnaryOpType::Sigmoid:
    case UnaryOpType::IsFinite:
    case UnaryOpType::IsInf:
    case UnaryOpType::IsNan:
    case UnaryOpType::IsNegInf:
    case UnaryOpType::IsPosInf:
    case UnaryOpType::IsReal:
    case UnaryOpType::Print:
    case UnaryOpType::ToUnsignedSmemAddr:
    case UnaryOpType::AdjustPartialLdMatrixAddrInTuring8:
    case UnaryOpType::AdjustPartialLdMatrixAddrInTuring16:
      return false;
    default:
      return true;
  }
}

bool needFloatSuffix(RNGOpType t) {
  return true;
}

static const char* unary_op_type2string(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Abs:
      return "abs";
    case UnaryOpType::Acos:
      return "acos";
    case UnaryOpType::Acosh:
      return "acosh";
    case UnaryOpType::Asin:
      return "asin";
    case UnaryOpType::Asinh:
      return "asinh";
    case UnaryOpType::Atan:
      return "atan";
    case UnaryOpType::Atanh:
      return "atanh";
    case UnaryOpType::Cast:
      return "cast";
    case UnaryOpType::Ceil:
      return "ceil";
    case UnaryOpType::Cos:
      return "cos";
    case UnaryOpType::Cosh:
      return "cosh";
    case UnaryOpType::Dereference:
      return "dereference";
    case UnaryOpType::Exp:
      return "exp";
    case UnaryOpType::Exp2:
      return "exp2";
    case UnaryOpType::Expm1:
      return "expm1";
    case UnaryOpType::Erf:
      return "erf";
    case UnaryOpType::Erfc:
      return "erfc";
    case UnaryOpType::Erfinv:
      return "erfinv";
    case UnaryOpType::Erfcinv:
      return "erfcinv";
    case UnaryOpType::Floor:
      return "floor";
    case UnaryOpType::Frac:
      return "frac";
    case UnaryOpType::Silu:
      return "silu";
    case UnaryOpType::Lgamma:
      return "lgamma";
    case UnaryOpType::Log:
      return "log";
    case UnaryOpType::Log10:
      return "log10";
    case UnaryOpType::Log1p:
      return "log1p";
    case UnaryOpType::Log2:
      return "log2";
    case UnaryOpType::BitCast:
      return "bit_cast";
    case UnaryOpType::Neg:
      return "neg";
    case UnaryOpType::BitCeil:
      return "bit_ceil";
    case UnaryOpType::LogicalNot:
      return "logical_not";
    case UnaryOpType::BitwiseNot:
      return "bitwise_not";
    case UnaryOpType::Print:
      return "print";
    case UnaryOpType::Reciprocal:
      return "reciprocal";
    case UnaryOpType::Relu:
      return "relu";
    case UnaryOpType::Rsqrt:
      return "rsqrt";
    case UnaryOpType::Round:
      return "nearbyint";
    case UnaryOpType::Sigmoid:
      return "sigmoid";
    case UnaryOpType::Signbit:
      return "signbit";
    case UnaryOpType::Sin:
      return "sin";
    case UnaryOpType::Sinh:
      return "sinh";
    case UnaryOpType::Sqrt:
      return "sqrt";
    case UnaryOpType::Tan:
      return "tan";
    case UnaryOpType::Tanh:
      return "tanh";
    case UnaryOpType::Trunc:
      return "trunc";
    case UnaryOpType::IsFinite:
      return "isfinite";
    case UnaryOpType::IsInf:
      return "isinf";
    case UnaryOpType::IsNan:
      return "isnan";
    case UnaryOpType::IsNegInf:
      return "isneginf";
    case UnaryOpType::IsPosInf:
      return "isposinf";
    case UnaryOpType::IsReal:
      return "isreal";
    case UnaryOpType::Real:
      return "std::real";
    case UnaryOpType::Imag:
      return "std::imag";
    case UnaryOpType::ToUnsignedSmemAddr:
      return "toSmem";
    case UnaryOpType::ElectSync:
      return "Hopper::electSync";
    case UnaryOpType::AdjustPartialLdMatrixAddrInTuring8:
      return "Turing::adjustPartialLdMatrixAddrInTuring<8>";
    case UnaryOpType::AdjustPartialLdMatrixAddrInTuring16:
      return "Turing::adjustPartialLdMatrixAddrInTuring<16>";
    default:
      NVF_THROW("No string found for unary op type.");
  }
}

static const char* unary_op_type_inline_op2string(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Dereference:
      return "*";
    case UnaryOpType::Neg:
      return "-";
    case UnaryOpType::LogicalNot:
      return "!";
    case UnaryOpType::BitwiseNot:
      return "~";
    case UnaryOpType::Address:
      return "&";
    default:
      break;
  }
  return nullptr;
}

bool needFloatSuffix(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Atan2:
    case BinaryOpType::Div:
    case BinaryOpType::Fmod:
      return true;
    default:
      return false;
  }
}

static const char* binary_op_type2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Add:
      return "add";
    case BinaryOpType::Atan2:
      return "atan2";
    case BinaryOpType::Div:
      return "div";
    case BinaryOpType::Fmod:
      return "fmod";
    case BinaryOpType::Max:
      return "fmax";
    case BinaryOpType::Min:
      return "fmin";
    case BinaryOpType::Mul:
      return "mul";
    case BinaryOpType::Nextafter:
      return "nextafter";
    case BinaryOpType::Pow:
      return "pow";
    case BinaryOpType::Remainder:
      return "remainder";
    case BinaryOpType::Sub:
      return "sub";
    case BinaryOpType::Complex:
      return "std::complex";

    // Integer Ops
    case BinaryOpType::Mod:
      return "mod";
    case BinaryOpType::CeilDiv:
      return "ceilDiv";
    case BinaryOpType::Lshift:
      return "lshift";
    case BinaryOpType::Rshift:
      return "rshift";
    case BinaryOpType::Gcd:
      return "gcd";

    // Bitwise Ops
    case BinaryOpType::BitwiseAnd:
      return "bitwise_and";
    case BinaryOpType::BitwiseOr:
      return "bitwise_or";
    case BinaryOpType::BitwiseXor:
      return "bitwise_xor";

    // Logical Ops
    case BinaryOpType::LogicalAnd:
      return "logical_and";
    case BinaryOpType::LogicalOr:
      return "logical_or";
    case BinaryOpType::Eq:
      return "equal";
    case BinaryOpType::GE:
      return "greaterThanOrEqual";
    case BinaryOpType::GT:
      return "greaterThan";
    case BinaryOpType::LE:
      return "lessThanOrEqual";
    case BinaryOpType::LT:
      return "lessThan";
    case BinaryOpType::NE:
      return "notEqual";
  }
  std::unreachable();
}

static const char* binary_op_integer_op2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Max:
      return "max";
    case BinaryOpType::Min:
      return "min";
    case BinaryOpType::Fmod:
      return "fmod";
    default:
      break;
  }
  return nullptr;
}

static const char* binary_op_bool_op2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Max:
      return "max";
    case BinaryOpType::Min:
      return "min";
    default:
      break;
  }
  return nullptr;
}

static const char* binary_op_type_inline_op2string(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Add:
      return "+";
    case BinaryOpType::Div:
      return "/";
    case BinaryOpType::Mul:
      return "*";
    case BinaryOpType::Sub:
      return "-";

    // Integer ops
    case BinaryOpType::Mod:
      return "%";
    case BinaryOpType::Lshift:
      return "<<";
    case BinaryOpType::Rshift:
      return ">>";
    // Logical Ops
    case BinaryOpType::Eq:
      return "==";
    case BinaryOpType::GE:
      return ">=";
    case BinaryOpType::GT:
      return ">";
    case BinaryOpType::LE:
      return "<=";
    case BinaryOpType::LT:
      return "<";
    case BinaryOpType::NE:
      return "!=";
    case BinaryOpType::LogicalAnd:
      return "&&";
    case BinaryOpType::LogicalOr:
      return "||";
    case BinaryOpType::BitwiseAnd:
      return "&";
    case BinaryOpType::BitwiseOr:
      return "|";
    case BinaryOpType::BitwiseXor:
      return "^";
    default:
      break;
  }
  return nullptr;
}

static const char* rng_op_type_inline_op2string(RNGOpType t) {
  switch (t) {
    case RNGOpType::Uniform:
      return "rng_uniform";
    case RNGOpType::UniformRange:
      return "rng_uniform_range";
    case RNGOpType::NormalStandard:
      return "rng_normal_standard";
    case RNGOpType::NormalGeneral:
      return "rng_normal_general";
    default:
      break;
  }
  return nullptr;
}

static const char* ternary_op_type2string(TernaryOpType t) {
  switch (t) {
    case TernaryOpType::Clamp:
      return "clamp";
    case TernaryOpType::Lerp:
      return "lerp";
    case TernaryOpType::Threshold:
      return "threshold";
    case TernaryOpType::Where:
      return "where";
    case TernaryOpType::Philox:
      return "philox";
  }
  std::unreachable();
}

static const char* rng_op_type2string(RNGOpType t) {
  switch (t) {
    case RNGOpType::Uniform:
      return "rng_uniform";
    case RNGOpType::UniformRange:
      return "rng_uniform_range";
    case RNGOpType::NormalStandard:
      return "rng_normal_standard";
    case RNGOpType::NormalGeneral:
      return "rng_normal_general";
    default:
      NVF_THROW("Unexpected RNGOpType");
  }
}

static const char* parallel_type2string(ParallelType t) {
  switch (t) {
    case ParallelType::DIDx:
      return "deviceIdx.x";
    case ParallelType::DIDy:
      return "deviceIdx.y";
    case ParallelType::DIDz:
      return "deviceIdx.z";
    case ParallelType::BIDz:
      return "blockIdx.z";
    case ParallelType::BIDy:
      return "blockIdx.y";
    case ParallelType::BIDx:
      return "blockIdx.x";
    case ParallelType::TIDz:
      return "threadIdx.z";
    case ParallelType::TIDy:
      return "threadIdx.y";
    case ParallelType::TIDx:
      return "threadIdx.x";
    case ParallelType::Stream:
      return "streamIdx";
    case ParallelType::Vectorize:
      return "V";
    case ParallelType::Unroll:
      return "UR";
    case ParallelType::Unswitch:
      return "US";
    case ParallelType::Mma:
      return "MMA";
    case ParallelType::Group:
      return "G";
    case ParallelType::Serial:
      return "S";
    case ParallelType::Bulk:
      return "B";
    default:
      NVF_THROW("Unexpected ParallelType");
  }
}

std::unordered_set<ParallelType> allParallelTypes() {
  static auto all_parallel_types = []() {
    std::unordered_set<ParallelType> s;
    for (auto i : arange(static_cast<int>(ParallelType::Count))) {
      s.insert(static_cast<ParallelType>(i));
    }
    return s;
  }();

  return all_parallel_types;
}

std::unordered_set<ParallelType> allParallelTypesExcept(
    const std::unordered_set<ParallelType>& except) {
  std::unordered_set<ParallelType> s = allParallelTypes();
  for (const auto t : except) {
    s.erase(t);
  }
  return s;
}

static const char* memory_type2string(MemoryType t) {
  switch (t) {
    case MemoryType::Local:
      return "register";
    case MemoryType::Shared:
      return "shared";
    case MemoryType::Global:
      return "global";
    case MemoryType::Tensor:
      return "tensor";
  }
  std::unreachable();
}

static const char* id_map_mode_type2string(IdMappingMode t) {
  switch (t) {
    case IdMappingMode::EXACT:
      return "exact";
    case IdMappingMode::ALMOSTEXACT:
      return "almost_exact";
    case IdMappingMode::BROADCAST:
      return "broadcast";
    case IdMappingMode::PERMISSIVE:
      return "permissive";
    case IdMappingMode::LOOP:
      return "loop";
    case IdMappingMode::INNERMOST:
      return "innermost";
    case IdMappingMode::PERMISSIVE_RESIZE:
      return "permissive_resize";
  }
  std::unreachable();
}

static const char* iter_type2string(IterType t) {
  switch (t) {
    case IterType::Iteration:
      return "i";
    case IterType::Reduction:
      return "r";
    case IterType::Broadcast:
      return "b";
    case IterType::Stride:
      return "s";
    case IterType::GatherScatter:
      return "n";
    case IterType::VectorComponent:
      return "v";
    case IterType::Symbolic:
      return "?";
  }
  std::unreachable();
}

static const char* thread_size2string(ParallelType t) {
  switch (t) {
    case ParallelType::BIDz:
      return "gridDim.z";
    case ParallelType::BIDy:
      return "gridDim.y";
    case ParallelType::BIDx:
      return "gridDim.x";
    case ParallelType::TIDz:
      return "blockDim.z";
    case ParallelType::TIDy:
      return "blockDim.y";
    case ParallelType::TIDx:
      return "blockDim.x";
    default:
      NVF_THROW("Unexpected parallel type");
  }
}

const char* load_store_type2string(LoadStoreOpType t) {
  switch (t) {
    case LoadStoreOpType::SegmenterSet:
      return "SegmenterSet";
    case LoadStoreOpType::Set:
      return "Set";
    case LoadStoreOpType::LdMatrix:
      return "LdMatrix";
    case LoadStoreOpType::StMatrix:
      return "StMatrix";
    case LoadStoreOpType::CpAsync:
      return "CpAsync";
    case LoadStoreOpType::CpAsyncBulk:
      return "CpAsyncBulk";
    case LoadStoreOpType::CpAsyncBulkTensorTile:
      return "CpAsyncBulkTensorTile";
    case LoadStoreOpType::LdTMem:
      return "LdTMem";
    case LoadStoreOpType::StTMem:
      return "StTMem";
  }
  std::unreachable();
}

const unsigned int _WORD_SHIFT = 16;
constexpr unsigned int supported_switch_pair(PrimDataType t1, PrimDataType t2) {
  return ((unsigned int)t1 << _WORD_SHIFT) + (unsigned int)t2;
}

static const char* supported_casts2string(std::pair<DataType, DataType> t) {
  if (t.first == DataType::SMemAddress) {
    t.first = DataType::UInt32;
  }
  switch (supported_switch_pair(
      std::get<PrimDataType>(t.first.type),
      std::get<PrimDataType>(t.second.type))) {
    case supported_switch_pair(DataType::Index, DataType::Float):
    case supported_switch_pair(DataType::Char, DataType::Float):
    case supported_switch_pair(DataType::Short, DataType::Float):
    case supported_switch_pair(DataType::Int32, DataType::Float):
    case supported_switch_pair(DataType::Int, DataType::Float):
    case supported_switch_pair(DataType::Byte, DataType::Float):
    case supported_switch_pair(DataType::UInt16, DataType::Float):
    case supported_switch_pair(DataType::UInt32, DataType::Float):
    case supported_switch_pair(DataType::UInt64, DataType::Float):
    case supported_switch_pair(DataType::Double, DataType::Float):
    case supported_switch_pair(DataType::Bool, DataType::Float):
      return "__to_float";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Float):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Float):
      return "__real_then_to_float";
    case supported_switch_pair(DataType::Index, DataType::Char):
    case supported_switch_pair(DataType::Short, DataType::Char):
    case supported_switch_pair(DataType::Int32, DataType::Char):
    case supported_switch_pair(DataType::Int, DataType::Char):
    case supported_switch_pair(DataType::Byte, DataType::Char):
    case supported_switch_pair(DataType::UInt16, DataType::Char):
    case supported_switch_pair(DataType::UInt32, DataType::Char):
    case supported_switch_pair(DataType::UInt64, DataType::Char):
    case supported_switch_pair(DataType::Float, DataType::Char):
    case supported_switch_pair(DataType::Double, DataType::Char):
    case supported_switch_pair(DataType::Bool, DataType::Char):
      return "__to_int8";
    case supported_switch_pair(DataType::Index, DataType::Short):
    case supported_switch_pair(DataType::Char, DataType::Short):
    case supported_switch_pair(DataType::Int32, DataType::Short):
    case supported_switch_pair(DataType::Int, DataType::Short):
    case supported_switch_pair(DataType::Byte, DataType::Short):
    case supported_switch_pair(DataType::UInt16, DataType::Short):
    case supported_switch_pair(DataType::UInt32, DataType::Short):
    case supported_switch_pair(DataType::UInt64, DataType::Short):
    case supported_switch_pair(DataType::Float, DataType::Short):
    case supported_switch_pair(DataType::Double, DataType::Short):
    case supported_switch_pair(DataType::Bool, DataType::Short):
      return "__to_int16";
    case supported_switch_pair(DataType::Index, DataType::Int32):
    case supported_switch_pair(DataType::Char, DataType::Int32):
    case supported_switch_pair(DataType::Short, DataType::Int32):
    case supported_switch_pair(DataType::Int, DataType::Int32):
    case supported_switch_pair(DataType::Byte, DataType::Int32):
    case supported_switch_pair(DataType::UInt16, DataType::Int32):
    case supported_switch_pair(DataType::UInt32, DataType::Int32):
    case supported_switch_pair(DataType::UInt64, DataType::Int32):
    case supported_switch_pair(DataType::Float, DataType::Int32):
    case supported_switch_pair(DataType::Double, DataType::Int32):
    case supported_switch_pair(DataType::Bool, DataType::Int32):
      return "__to_int32";
    case supported_switch_pair(DataType::Index, DataType::Int):
    case supported_switch_pair(DataType::Char, DataType::Int):
    case supported_switch_pair(DataType::Short, DataType::Int):
    case supported_switch_pair(DataType::Int32, DataType::Int):
    case supported_switch_pair(DataType::Byte, DataType::Int):
    case supported_switch_pair(DataType::UInt16, DataType::Int):
    case supported_switch_pair(DataType::UInt32, DataType::Int):
    case supported_switch_pair(DataType::UInt64, DataType::Int):
    case supported_switch_pair(DataType::Float, DataType::Int):
    case supported_switch_pair(DataType::Double, DataType::Int):
    case supported_switch_pair(DataType::Bool, DataType::Int):
      return "__to_int64";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Char):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Char):
      return "__real_then_to_int8";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Short):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Short):
      return "__real_then_to_int16";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Int32):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Int32):
      return "__real_then_to_int32";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Int):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Int):
      return "__real_then_to_int64";
    case supported_switch_pair(DataType::Index, DataType::Byte):
    case supported_switch_pair(DataType::Char, DataType::Byte):
    case supported_switch_pair(DataType::Short, DataType::Byte):
    case supported_switch_pair(DataType::Int32, DataType::Byte):
    case supported_switch_pair(DataType::Int, DataType::Byte):
    case supported_switch_pair(DataType::UInt16, DataType::Byte):
    case supported_switch_pair(DataType::UInt32, DataType::Byte):
    case supported_switch_pair(DataType::UInt64, DataType::Byte):
    case supported_switch_pair(DataType::Float, DataType::Byte):
    case supported_switch_pair(DataType::Double, DataType::Byte):
    case supported_switch_pair(DataType::Bool, DataType::Byte):
      return "__to_uint8";
    case supported_switch_pair(DataType::Index, DataType::UInt16):
    case supported_switch_pair(DataType::Char, DataType::UInt16):
    case supported_switch_pair(DataType::Short, DataType::UInt16):
    case supported_switch_pair(DataType::Int32, DataType::UInt16):
    case supported_switch_pair(DataType::Int, DataType::UInt16):
    case supported_switch_pair(DataType::Byte, DataType::UInt16):
    case supported_switch_pair(DataType::UInt32, DataType::UInt16):
    case supported_switch_pair(DataType::UInt64, DataType::UInt16):
    case supported_switch_pair(DataType::Float, DataType::UInt16):
    case supported_switch_pair(DataType::Double, DataType::UInt16):
    case supported_switch_pair(DataType::Bool, DataType::UInt16):
      return "__to_uint16";
    case supported_switch_pair(DataType::Index, DataType::UInt32):
    case supported_switch_pair(DataType::Char, DataType::UInt32):
    case supported_switch_pair(DataType::Short, DataType::UInt32):
    case supported_switch_pair(DataType::Int32, DataType::UInt32):
    case supported_switch_pair(DataType::Int, DataType::UInt32):
    case supported_switch_pair(DataType::Byte, DataType::UInt32):
    case supported_switch_pair(DataType::UInt16, DataType::UInt32):
    case supported_switch_pair(DataType::UInt64, DataType::UInt32):
    case supported_switch_pair(DataType::Float, DataType::UInt32):
    case supported_switch_pair(DataType::Double, DataType::UInt32):
    case supported_switch_pair(DataType::Bool, DataType::UInt32):
      return "__to_uint32";
    case supported_switch_pair(DataType::Index, DataType::UInt64):
    case supported_switch_pair(DataType::Char, DataType::UInt64):
    case supported_switch_pair(DataType::Short, DataType::UInt64):
    case supported_switch_pair(DataType::Int32, DataType::UInt64):
    case supported_switch_pair(DataType::Int, DataType::UInt64):
    case supported_switch_pair(DataType::Byte, DataType::UInt64):
    case supported_switch_pair(DataType::UInt16, DataType::UInt64):
    case supported_switch_pair(DataType::UInt32, DataType::UInt64):
    case supported_switch_pair(DataType::Float, DataType::UInt64):
    case supported_switch_pair(DataType::Double, DataType::UInt64):
    case supported_switch_pair(DataType::Bool, DataType::UInt64):
      return "__to_uint64";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Byte):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Byte):
      return "__real_then_to_uint8";
    case supported_switch_pair(DataType::ComplexFloat, DataType::UInt16):
    case supported_switch_pair(DataType::ComplexDouble, DataType::UInt16):
      return "__real_then_to_uint16";
    case supported_switch_pair(DataType::ComplexFloat, DataType::UInt32):
    case supported_switch_pair(DataType::ComplexDouble, DataType::UInt32):
      return "__real_then_to_uint32";
    case supported_switch_pair(DataType::ComplexFloat, DataType::UInt64):
    case supported_switch_pair(DataType::ComplexDouble, DataType::UInt64):
      return "__real_then_to_uint64";
    case supported_switch_pair(DataType::Char, DataType::Index):
    case supported_switch_pair(DataType::Short, DataType::Index):
    case supported_switch_pair(DataType::Int32, DataType::Index):
    case supported_switch_pair(DataType::Int, DataType::Index):
    case supported_switch_pair(DataType::Byte, DataType::Index):
    case supported_switch_pair(DataType::UInt16, DataType::Index):
    case supported_switch_pair(DataType::UInt32, DataType::Index):
    case supported_switch_pair(DataType::UInt64, DataType::Index):
    case supported_switch_pair(DataType::Float, DataType::Index):
    case supported_switch_pair(DataType::Double, DataType::Index):
    case supported_switch_pair(DataType::Bool, DataType::Index):
      return "__to_index";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Index):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Index):
      return "__real_then_to_index";
    case supported_switch_pair(DataType::Index, DataType::Double):
    case supported_switch_pair(DataType::Char, DataType::Double):
    case supported_switch_pair(DataType::Short, DataType::Double):
    case supported_switch_pair(DataType::Int32, DataType::Double):
    case supported_switch_pair(DataType::Int, DataType::Double):
    case supported_switch_pair(DataType::Byte, DataType::Double):
    case supported_switch_pair(DataType::UInt16, DataType::Double):
    case supported_switch_pair(DataType::UInt32, DataType::Double):
    case supported_switch_pair(DataType::UInt64, DataType::Double):
    case supported_switch_pair(DataType::Float, DataType::Double):
    case supported_switch_pair(DataType::Bool, DataType::Double):
      return "__to_double";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Double):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Double):
      return "__real_then_to_double";
    case supported_switch_pair(DataType::Float, DataType::Bool):
    case supported_switch_pair(DataType::Double, DataType::Bool):
    case supported_switch_pair(DataType::Index, DataType::Bool):
    case supported_switch_pair(DataType::Char, DataType::Bool):
    case supported_switch_pair(DataType::Short, DataType::Bool):
    case supported_switch_pair(DataType::Int32, DataType::Bool):
    case supported_switch_pair(DataType::Int, DataType::Bool):
    case supported_switch_pair(DataType::Byte, DataType::Bool):
    case supported_switch_pair(DataType::UInt16, DataType::Bool):
    case supported_switch_pair(DataType::UInt32, DataType::Bool):
    case supported_switch_pair(DataType::UInt64, DataType::Bool):
      return "__to_bool";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Bool):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Bool):
      return "__real_then_to_bool";
    case supported_switch_pair(DataType::Index, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Char, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Short, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Int32, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Int, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Byte, DataType::ComplexDouble):
    case supported_switch_pair(DataType::UInt16, DataType::ComplexDouble):
    case supported_switch_pair(DataType::UInt32, DataType::ComplexDouble):
    case supported_switch_pair(DataType::UInt64, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Double, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Float, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Bool, DataType::ComplexDouble):
    case supported_switch_pair(DataType::ComplexFloat, DataType::ComplexDouble):
      return "__to_complex_double";
    case supported_switch_pair(DataType::Index, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Char, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Short, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Int32, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Int, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Byte, DataType::ComplexFloat):
    case supported_switch_pair(DataType::UInt16, DataType::ComplexFloat):
    case supported_switch_pair(DataType::UInt32, DataType::ComplexFloat):
    case supported_switch_pair(DataType::UInt64, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Double, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Float, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Bool, DataType::ComplexFloat):
    case supported_switch_pair(DataType::ComplexDouble, DataType::ComplexFloat):
      return "__to_complex_float";

    case supported_switch_pair(DataType::Float, DataType::Half):
      return "__float2half";
    case supported_switch_pair(DataType::Double, DataType::Half):
      return "__double2half";
    case supported_switch_pair(DataType::Char, DataType::Half):
    case supported_switch_pair(DataType::Short, DataType::Half):
    case supported_switch_pair(DataType::Int32, DataType::Half):
    case supported_switch_pair(DataType::Int, DataType::Half):
    case supported_switch_pair(DataType::Byte, DataType::Half):
    case supported_switch_pair(DataType::UInt16, DataType::Half):
    case supported_switch_pair(DataType::UInt32, DataType::Half):
    case supported_switch_pair(DataType::UInt64, DataType::Half):
    case supported_switch_pair(DataType::Index, DataType::Half):
      return "__int2half";
    case supported_switch_pair(DataType::Bool, DataType::Half):
      return "__bool2half";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Half):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Half):
      return "__real_then_2half";

    case supported_switch_pair(DataType::Half, DataType::Float):
      return "__half2float";
    case supported_switch_pair(DataType::Half, DataType::Double):
      return "__half2double";
    case supported_switch_pair(DataType::Half, DataType::Char):
    case supported_switch_pair(DataType::Half, DataType::Short):
    case supported_switch_pair(DataType::Half, DataType::Int32):
      return "__half2int32";
    case supported_switch_pair(DataType::Half, DataType::Int):
      return "__half2int";
    case supported_switch_pair(DataType::Half, DataType::Byte):
    case supported_switch_pair(DataType::Half, DataType::UInt16):
    case supported_switch_pair(DataType::Half, DataType::UInt32):
      return "__half2uint32";
    case supported_switch_pair(DataType::Half, DataType::UInt64):
      return "__half2uint";
    case supported_switch_pair(DataType::Half, DataType::Index):
      return "__half2index";
    case supported_switch_pair(DataType::Half, DataType::Bool):
      return "__half2bool";
    case supported_switch_pair(DataType::Half, DataType::ComplexFloat):
      return "__half2complex_float";
    case supported_switch_pair(DataType::Half, DataType::ComplexDouble):
      return "__half2complex_double";

    case supported_switch_pair(DataType::Float, DataType::BFloat16):
      return "__float2bfloat";
    case supported_switch_pair(DataType::Double, DataType::BFloat16):
      return "__double2bfloat";
    case supported_switch_pair(DataType::Half, DataType::BFloat16):
      return "__half2bfloat";
    case supported_switch_pair(DataType::Char, DataType::BFloat16):
    case supported_switch_pair(DataType::Short, DataType::BFloat16):
    case supported_switch_pair(DataType::Int32, DataType::BFloat16):
    case supported_switch_pair(DataType::Int, DataType::BFloat16):
    case supported_switch_pair(DataType::Byte, DataType::BFloat16):
    case supported_switch_pair(DataType::UInt16, DataType::BFloat16):
    case supported_switch_pair(DataType::UInt32, DataType::BFloat16):
    case supported_switch_pair(DataType::UInt64, DataType::BFloat16):
    case supported_switch_pair(DataType::Index, DataType::BFloat16):
      return "__int2bfloat";
    case supported_switch_pair(DataType::Bool, DataType::BFloat16):
      return "__bool2bfloat";
    case supported_switch_pair(DataType::ComplexFloat, DataType::BFloat16):
    case supported_switch_pair(DataType::ComplexDouble, DataType::BFloat16):
      return "__real_then_2bfloat";

    case supported_switch_pair(DataType::BFloat16, DataType::Float):
      return "__bfloat2float";
    case supported_switch_pair(DataType::BFloat16, DataType::Double):
      return "__bfloat2double";
    case supported_switch_pair(DataType::BFloat16, DataType::Half):
      return "__bfloat2half";
    case supported_switch_pair(DataType::BFloat16, DataType::Char):
    case supported_switch_pair(DataType::BFloat16, DataType::Short):
    case supported_switch_pair(DataType::BFloat16, DataType::Int32):
      return "__bfloat2int32";
    case supported_switch_pair(DataType::BFloat16, DataType::Int):
      return "__bfloat2int";
    case supported_switch_pair(DataType::BFloat16, DataType::Byte):
    case supported_switch_pair(DataType::BFloat16, DataType::UInt16):
    case supported_switch_pair(DataType::BFloat16, DataType::UInt32):
      return "__bfloat2uint32";
    case supported_switch_pair(DataType::BFloat16, DataType::UInt64):
      return "__bfloat2uint";
    case supported_switch_pair(DataType::BFloat16, DataType::Index):
      return "__bfloat2index";
    case supported_switch_pair(DataType::BFloat16, DataType::Bool):
      return "__bfloat2bool";
    case supported_switch_pair(DataType::BFloat16, DataType::ComplexFloat):
      return "__bfloat2complex_float";
    case supported_switch_pair(DataType::BFloat16, DataType::ComplexDouble):
      return "__bfloat2complex_double";

    case supported_switch_pair(DataType::Float8_e5m2, DataType::Float):
      return "__e5m22float";
    case supported_switch_pair(DataType::Float8_e5m2, DataType::Double):
      return "__e5m22double";
    case supported_switch_pair(DataType::Float8_e5m2, DataType::Half):
      return "__e5m22half";
    case supported_switch_pair(DataType::Float8_e5m2, DataType::BFloat16):
      return "__e5m22bfloat";
    case supported_switch_pair(DataType::Float, DataType::Float8_e5m2):
      return "__float2e5m2";
    case supported_switch_pair(DataType::Double, DataType::Float8_e5m2):
      return "__double2e5m2";
    case supported_switch_pair(DataType::Half, DataType::Float8_e5m2):
      return "__half2e5m2";
    case supported_switch_pair(DataType::BFloat16, DataType::Float8_e5m2):
      return "__bfloat2e5m2";

    case supported_switch_pair(DataType::Float8_e4m3fn, DataType::Float):
      return "__e4m32float";
    case supported_switch_pair(DataType::Float8_e4m3fn, DataType::Double):
      return "__e4m32double";
    case supported_switch_pair(DataType::Float8_e4m3fn, DataType::Half):
      return "__e4m32half";
    case supported_switch_pair(DataType::Float8_e4m3fn, DataType::BFloat16):
      return "__e4m32bfloat";
    case supported_switch_pair(DataType::Float, DataType::Float8_e4m3fn):
      return "__float2e4m3";
    case supported_switch_pair(DataType::Double, DataType::Float8_e4m3fn):
      return "__double2e4m3";
    case supported_switch_pair(DataType::Half, DataType::Float8_e4m3fn):
      return "__half2e4m3";
    case supported_switch_pair(DataType::BFloat16, DataType::Float8_e4m3fn):
      return "__bfloat2e4m3";

    case supported_switch_pair(DataType::Float8_e8m0fnu, DataType::Float):
      return "__e8m02float";
    case supported_switch_pair(DataType::Float8_e8m0fnu, DataType::Double):
      return "__e8m02double";
    case supported_switch_pair(DataType::Float8_e8m0fnu, DataType::Half):
      return "__e8m02half";
    case supported_switch_pair(DataType::Float8_e8m0fnu, DataType::BFloat16):
      return "__e8m02bfloat";
    case supported_switch_pair(DataType::Float, DataType::Float8_e8m0fnu):
      return "__float2e8m0";
    case supported_switch_pair(DataType::Double, DataType::Float8_e8m0fnu):
      return "__double2e8m0";
    case supported_switch_pair(DataType::Half, DataType::Float8_e8m0fnu):
      return "__half2e8m0";
    case supported_switch_pair(DataType::BFloat16, DataType::Float8_e8m0fnu):
      return "__bfloat2e8m0";

    case supported_switch_pair(DataType::Float4_e2m1fn, DataType::Float):
      return "__e2m12float";
    case supported_switch_pair(DataType::Float4_e2m1fn, DataType::Double):
      return "__e2m12double";
    case supported_switch_pair(DataType::Float4_e2m1fn, DataType::Half):
      return "__e2m12half";
    case supported_switch_pair(DataType::Float4_e2m1fn, DataType::BFloat16):
      return "__e2m12bfloat";
    case supported_switch_pair(DataType::Float, DataType::Float4_e2m1fn):
      return "__float2e2m1";
    case supported_switch_pair(DataType::Double, DataType::Float4_e2m1fn):
      return "__double2e2m1";
    case supported_switch_pair(DataType::Half, DataType::Float4_e2m1fn):
      return "__half2e2m1";
    case supported_switch_pair(DataType::BFloat16, DataType::Float4_e2m1fn):
      return "__bfloat2e2m1";

    default:
      return nullptr;
  }
}

DataType aten_to_data_type(const at::ScalarType& scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Bool:
      return DataType::Bool;
    case at::ScalarType::Double:
      return DataType::Double;
    case at::ScalarType::Float:
      return DataType::Float;
    case at::ScalarType::Half:
      return DataType::Half;
    case at::ScalarType::BFloat16:
      return DataType::BFloat16;
    case at::ScalarType::Float8_e4m3fn:
      return DataType::Float8_e4m3fn;
    case at::ScalarType::Float8_e5m2:
      return DataType::Float8_e5m2;
    case at::ScalarType::Float8_e8m0fnu:
      return DataType::Float8_e8m0fnu;
#if NVF_TORCH_VERSION_NO_LESS(2, 8, 0)
    case at::ScalarType::Float4_e2m1fn_x2:
      return DataType::Float4_e2m1fn;
#endif
    case at::ScalarType::Char:
      return DataType::Char;
    case at::ScalarType::Short:
      return DataType::Short;
    case at::ScalarType::Int:
      return DataType::Int32;
    case at::ScalarType::Long:
      return DataType::Int;
    case at::ScalarType::Byte:
      return DataType::Byte;
    case at::ScalarType::UInt16:
      return DataType::UInt16;
    case at::ScalarType::UInt32:
      return DataType::UInt32;
    case at::ScalarType::UInt64:
      return DataType::UInt64;
    case at::ScalarType::ComplexFloat:
      return DataType::ComplexFloat;
    case at::ScalarType::ComplexDouble:
      return DataType::ComplexDouble;
    default:
      return DataType::Null;
  }
}

at::ScalarType data_type_to_aten(const DataType& data_type) {
  if (std::holds_alternative<PrimDataType>(data_type.type)) {
    switch (std::get<PrimDataType>(data_type.type)) {
      case DataType::Bool:
        return at::ScalarType::Bool;
      case DataType::Double:
        return at::ScalarType::Double;
      case DataType::Float:
        return at::ScalarType::Float;
      case DataType::Half:
        return at::ScalarType::Half;
      case DataType::BFloat16:
        return at::ScalarType::BFloat16;
      case DataType::Float8_e4m3fn:
        return at::ScalarType::Float8_e4m3fn;
      case DataType::Float8_e5m2:
        return at::ScalarType::Float8_e5m2;
      case DataType::Float8_e8m0fnu:
        return at::ScalarType::Float8_e8m0fnu;
#if NVF_TORCH_VERSION_NO_LESS(2, 8, 0)
      case DataType::Float4_e2m1fn_x2:
        return at::ScalarType::Float4_e2m1fn_x2;
      case DataType::Float4_e2m1fn:
        return at::ScalarType::Float4_e2m1fn_x2;
#endif
      case DataType::Index:
        NVF_THROW(
            "Index is determined at compile time,",
            " to convert from an aten type you need to have the compiled "
            "information. ",
            "This information is passed to GpuLower at compile time, and then "
            "copied to kerned.",
            "There's also this information in FusionExecutorCache and the "
            "Registry system.");
      case DataType::Char:
        return at::ScalarType::Char;
      case DataType::Short:
        return at::ScalarType::Short;
      case DataType::Int32:
        return at::ScalarType::Int;
      case DataType::Int:
        return at::ScalarType::Long;
      case DataType::Byte:
        return at::ScalarType::Byte;
      case DataType::UInt16:
        return at::ScalarType::UInt16;
      case DataType::UInt32:
        return at::ScalarType::UInt32;
      case DataType::UInt64:
        return at::ScalarType::UInt64;
      case DataType::ComplexFloat:
        return at::ScalarType::ComplexFloat;
      case DataType::ComplexDouble:
        return at::ScalarType::ComplexDouble;
      default:
        break;
    }
  }
  // NVFuser's DataType is much wider than PyTorch's ScalarType. If
  // there is no direct mapping, we use some data type as a proxy.
  // If there is a data type with the same size, we use that
  const int64_t size_bit = dataTypeSizeBit(data_type);
  if (size_bit == 8) {
    return at::ScalarType::Byte;
  } else if (size_bit == 16) {
    return at::ScalarType::UInt16;
  } else if (size_bit == 32) {
    return at::ScalarType::UInt32;
  } else if (size_bit == 64) {
    return at::ScalarType::UInt64;
  } else if (size_bit == 128) {
    return at::ScalarType::ComplexDouble;
  } else {
    // If there is no data type with the same size, we use byte.
    // For this case, we adjust the size of the last dimension.
    // For example, if we have a TensorView with shape [10, 4],
    // and dtype is 3 bytes, then the corresponding ScalarType is Byte,
    // and the shape of the corresponding at::Tensor is [10, 12].
    return at::ScalarType::Byte;
  }
}

at::ScalarType data_type_to_aten(
    const DataType& data_type,
    const DataType& index_type) {
  if (data_type == DataType::Index) {
    return data_type_to_aten(index_type);
  }
  return data_type_to_aten(data_type);
}

AdjustLastDim getLastDimAdjustment(const DataType& dtype) {
  if (dtype == DataType::Index) {
    return AdjustLastDim{1, 1};
  }
  const int64_t scalar_type_bit =
      (int64_t)c10::elementSize(data_type_to_aten(dtype)) * 8;
  const int64_t dtype_bit = dataTypeSizeBit(dtype);
  // Example: dtype_bit = 6, scalar_type_bit = 8
  // Then we need to adjust the last dimension by 4/3, that is,
  // at_size * 4 / 3 is the size of the last dimension of the corresponding
  // TensorView.
  const int64_t gcd = std::gcd(scalar_type_bit, dtype_bit);
  return AdjustLastDim{scalar_type_bit / gcd, dtype_bit / gcd};
}

std::ostream& operator<<(std::ostream& out, const ValType vtype) {
  return out << val_type2string(vtype);
}

std::ostream& operator<<(
    std::ostream& out,
    const BlockScalingFactorLayout layout) {
  return out << block_sf_layout2string(layout);
}

std::ostream& operator<<(std::ostream& out, const PredicateType ptype) {
  return out << predicate_type2string(ptype);
}

std::ostream& operator<<(std::ostream& out, const DataType dtype) {
  return out << data_type2string(dtype);
}

std::ostream& operator<<(std::ostream& out, const UnaryOpType uotype) {
  return out << unary_op_type2string(uotype);
}

std::ostream& operator<<(std::ostream& out, const BinaryOpType botype) {
  return out << binary_op_type2string(botype);
}

std::ostream& operator<<(std::ostream& out, const TernaryOpType totype) {
  return out << ternary_op_type2string(totype);
}

std::ostream& operator<<(std::ostream& out, const RNGOpType rngtype) {
  return out << rng_op_type2string(rngtype);
}

std::ostream& operator<<(std::ostream& out, const ParallelType ptype) {
  return out << stringifyThread(ptype);
}

std::ostream& operator<<(std::ostream& out, const MemoryType mtype) {
  return out << memory_type2string(mtype);
}

std::ostream& operator<<(std::ostream& out, const IdMappingMode immtype) {
  return out << id_map_mode_type2string(immtype);
}

std::ostream& operator<<(
    std::ostream& out,
    const LoadStoreOpType load_store_type) {
  return out << load_store_type2string(load_store_type);
}

std::ostream& operator<<(std::ostream& out, const IterType bt) {
  return out << iter_type2string(bt);
}

std::ostream& operator<<(std::ostream& os, const SwizzleType& swizzle) {
  switch (swizzle) {
    case SwizzleType::NoSwizzle:
      os << "NoSwizzle";
      break;
    case SwizzleType::XOR:
      os << "Xor";
      break;
    case SwizzleType::CyclicShift:
      os << "CyclicShift";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Swizzle2DType& swizzle) {
  switch (swizzle) {
    case Swizzle2DType::NoSwizzle:
      os << "NoSwizzle";
      break;
    case Swizzle2DType::ZShape:
      os << "ZShape";
      break;
    case Swizzle2DType::XOR:
      os << "Xor";
      break;
    case Swizzle2DType::CyclicShift:
      os << "CyclicShift";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const SwizzleMode& swizzle) {
  switch (swizzle) {
    case SwizzleMode::NoSwizzle:
      os << "NoSwizzle";
      break;
    case SwizzleMode::Loop:
      os << "Loop";
      break;
    case SwizzleMode::Data:
      os << "Data";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const KernelIndexMode& index_mode) {
  switch (index_mode) {
    case KernelIndexMode::INT32:
      os << "INT32";
      break;
    case KernelIndexMode::INT64:
      os << "INT64";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const CacheOp& cache_op) {
  switch (cache_op) {
    case CacheOp::Unspecified:
      os << "Unspecified";
      break;
    case CacheOp::AllLevels:
      os << "AllLevels";
      break;
    case CacheOp::Streaming:
      os << "Streaming";
      break;
    case CacheOp::Global:
      os << "Global";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::optional<bool>& b) {
  return os << (b.has_value() ? (*b ? "t" : "f") : "n");
}

std::optional<std::string> inline_op_str(const UnaryOpType uotype) {
  const char* str = unary_op_type_inline_op2string(uotype);
  return str != nullptr ? std::optional<std::string>(std::string(str))
                        : std::nullopt;
}

std::optional<std::string> inline_op_str(const BinaryOpType botype) {
  const char* str = binary_op_type_inline_op2string(botype);
  return str != nullptr ? std::optional<std::string>(std::string(str))
                        : std::nullopt;
}

std::optional<std::string> inline_op_str(const RNGOpType rngtype) {
  const char* str = rng_op_type_inline_op2string(rngtype);
  return str != nullptr ? std::optional<std::string>(std::string(str))
                        : std::nullopt;
}

std::optional<std::string> integer_op_str(const BinaryOpType botype) {
  const char* str = binary_op_integer_op2string(botype);
  return str != nullptr ? std::optional<std::string>(std::string(str))
                        : std::nullopt;
}

std::optional<std::string> bool_op_str(const BinaryOpType botype) {
  const char* str = binary_op_bool_op2string(botype);
  return str != nullptr ? std::optional<std::string>(std::string(str))
                        : std::nullopt;
}

std::string stringifyThreadSize(const ParallelType ptype) {
  return thread_size2string(ptype);
}

std::string stringifyThread(const ParallelType ptype) {
  return parallel_type2string(ptype);
}

std::string typePrefix(const DataType data_type) {
  if (std::holds_alternative<PointerType>(data_type.type)) {
    return "ptr";
  }
  if (std::holds_alternative<ArrayType>(data_type.type)) {
    return "a";
  }
  if (std::holds_alternative<StructType>(data_type.type)) {
    return "s";
  }
  if (std::holds_alternative<OpaqueType>(data_type.type)) {
    return "var";
  }
  switch (std::get<PrimDataType>(data_type.type)) {
    case DataType::Bool:
      return "b";
    case DataType::Double:
      return "d";
    case DataType::Float:
    case DataType::Half:
    case DataType::BFloat16:
    case DataType::Float8_e4m3fn:
    case DataType::Float8_e5m2:
    case DataType::Float8_e8m0fnu:
    case DataType::Float4_e2m1fn:
      return "f";
    case DataType::Float4_e2m1fn_x2:
      return "f4x2_";
    case DataType::Index:
    case DataType::Int:
    case DataType::Int32:
    case DataType::Short:
    case DataType::UInt64:
    case DataType::UInt32:
    case DataType::UInt16:
    case DataType::SMemAddress:
      return "i";
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
      return "c";
    default:
      NVF_THROW("No data type found for scalar type.");
  }
}

bool isParallelTypeThreadDim(ParallelType ptype) {
  return ptype == ParallelType::TIDx || ptype == ParallelType::TIDy ||
      ptype == ParallelType::TIDz;
}

bool isParallelTypeBlockDim(ParallelType ptype) {
  return ptype == ParallelType::BIDx || ptype == ParallelType::BIDy ||
      ptype == ParallelType::BIDz;
}

bool isParallelTypeDeviceDim(ParallelType ptype) {
  return ptype == ParallelType::DIDx || ptype == ParallelType::DIDy ||
      ptype == ParallelType::DIDz;
}

bool isParallelTypeThread(ParallelType ptype) {
  return isParallelTypeBlockDim(ptype) || isParallelTypeThreadDim(ptype);
}

bool isParallelTypeVectorize(ParallelType ptype) {
  return ptype == ParallelType::Vectorize;
}

std::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>& cast) {
  const char* str = supported_casts2string(cast);
  return str != nullptr ? std::optional<std::string>(std::string(str))
                        : std::nullopt;
}

int64_t dataTypeSizeBit(DataType type) {
  return std::visit(
      [](auto&& dtype) -> int64_t {
        using T = std::decay_t<decltype(dtype)>;
        if constexpr (std::is_same_v<T, PrimDataType>) {
          return primDataTypeSizeBit(dtype);
        } else if constexpr (std::is_same_v<T, PointerType>) {
          return sizeof(void*) * 8;
        } else if constexpr (std::is_same_v<T, ArrayType>) {
          return dataTypeSizeBit(*dtype.type) * dtype.size;
        } else if constexpr (std::is_same_v<T, StructType>) {
          int64_t size = 0;
          for (const auto& field : dtype.fields) {
            if (!field.used_in_kernel) {
              continue;
            }
            size += dataTypeSizeBit(*field.type);
          }
          return size;
        } else if constexpr (std::is_same_v<T, OpaqueType>) {
          return dtype.size * 8;
        }
        NVF_THROW("Size undefined for data type.");
      },
      type.type);
}

int64_t dataTypeSizeByte(DataType type) {
  int64_t bits = dataTypeSizeBit(type);
  // FIXME: this isn't right
  if (bits < 8) {
    return 1;
  }
  NVF_CHECK(bits % 8 == 0, "Size is not a multiple of 8 bits.");
  return bits / 8;
}

int64_t dataTypeSizeBit(DataType type, DataType index_type) {
  if (type == DataType::Index) {
    NVF_ERROR(
        index_type == DataType::Int32 || index_type == DataType::Int,
        "Invalid index type of ",
        index_type);
    return dataTypeSizeBit(index_type);
  }
  return dataTypeSizeBit(type);
}

int64_t dataTypeSizeByte(DataType type, DataType index_type) {
  int64_t bits = dataTypeSizeBit(type, index_type);
  // FIXME: this isn't right
  if (bits < 8) {
    return 1;
  }
  NVF_CHECK(bits % 8 == 0, "Size is not a multiple of 8 bits.");
  return bits / 8;
}

std::ostream& operator<<(
    std::ostream& os,
    const CircularBufferLoopStage loop_stage) {
  switch (loop_stage) {
    case CircularBufferLoopStage::NotApplicable:
      break;
    case CircularBufferLoopStage::Prolog:
      os << "{CircularBufferProlog}";
      break;
    case CircularBufferLoopStage::Main:
      os << "{CircularBufferMainLoop}";
      break;
    case CircularBufferLoopStage::Epilog:
      os << "{CircularBufferEpilog}";
      break;
    case CircularBufferLoopStage::AsyncWarp:
      os << "{AsyncWarp}";
      break;
    case CircularBufferLoopStage::ComputeWarp:
      os << "{ComputeWarp}";
      break;
    default:
      NVF_THROW("unknown circular buffer stage");
  }
  return os;
}

int max_digits10(DataType dtype) {
  // [max_digits10 calculation]
  // As of C++17 there is no max_digits10 for __half or bfloat16, so we use the
  // general formula (see [1] p31 Section 5.2.4.2.2 part 11):
  //   ceil(1 + p log10(2))
  // where p is the precision of the type (aka significand):
  //    Type      Precision   max_digits10
  //   fp8_e5m2       3           2
  //   fp8_e4m3       4           3
  //   fp8_e8m0       1           2
  //   fp4_e2m1       2           2
  //   bfloat16       8           4
  //   float16       11           5
  //   float32       24           9
  //   float64       53          17
  // [1] http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
  if (dtype == DataType::Float || dtype == DataType::ComplexFloat) {
    return std::numeric_limits<float>::max_digits10;
  } else if (dtype == DataType::Double || dtype == DataType::ComplexDouble) {
    return std::numeric_limits<double>::max_digits10;
  } else if (dtype == DataType::Half) {
    return 5;
  } else if (dtype == DataType::BFloat16) {
    return 4;
  } else if (dtype == DataType::Float8_e4m3fn) {
    return 3;
  } else if (
      dtype == DataType::Float8_e5m2 || dtype == DataType::Float8_e8m0fnu) {
    return 2;
  } else if (dtype == DataType::Float4_e2m1fn) {
    return 2;
  } else {
    NVF_CHECK(
        !isFloatingPointType(dtype),
        "Unhandled floating point type in max_digits10 ",
        dtype);
    return 0;
  }
}

std::ostream& operator<<(std::ostream& os, TMemRegisterDataPath dp) {
  switch (dp) {
    case TMemRegisterDataPath::Path32x32b:
      return os << "32x32b";
    case TMemRegisterDataPath::Path16x64b:
      return os << "16x64b";
    case TMemRegisterDataPath::Path16x128b:
      return os << "16x128b";
    case TMemRegisterDataPath::Path16x256b:
      return os << "16x256b";
    case TMemRegisterDataPath::Path16x32bx2:
      return os << "16x32bx2";
  }
  std::unreachable();
}

std::ostream& operator<<(
    std::ostream& os,
    const cudaDriverEntryPointQueryResult result) {
  switch (result) {
    case cudaDriverEntryPointSuccess:
      os << "Success";
      break;
    case cudaDriverEntryPointSymbolNotFound:
      os << "SymbolNotFound";
      break;
    case cudaDriverEntryPointVersionNotSufficent:
      os << "VersionNotSufficient";
      break;
    default:
      NVF_THROW(
          "Unknown cudaDriverEntryPointQueryResult: ",
          static_cast<int>(result));
  }
  return os;
}

} // namespace nvfuser

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
#include <stdexcept>
#include <unordered_map>

#include <ir/all_nodes.h>

namespace nvfuser {

DataType globalTensorMetaData(
    const DataType& dtype,
    size_t dim,
    size_t alloc_dim) {
  std::stringstream ss;
  ss << "Tensor<" << dtype << ", " << dim << ", " << alloc_dim << ">";

  StructOf tv_metadata;
  tv_metadata.name = ss.str();
  tv_metadata.field_names = {"data", "logical_size", "alloc_stride"};
  tv_metadata.types["data"] =
      NVFUSER_MAYBE_MAKE_SHARED(PointerOf{std::make_shared<DataType>(dtype)});
  tv_metadata.types["logical_size"] = NVFUSER_MAYBE_MAKE_SHARED2(
      ArrayOf{std::make_shared<DataType>(DataType::Index), dim});
  tv_metadata.types["logical_stride"] = NVFUSER_MAYBE_MAKE_SHARED2(
      ArrayOf{std::make_shared<DataType>(DataType::Index), dim});
  tv_metadata.types["alloc_size"] = NVFUSER_MAYBE_MAKE_SHARED2(
      ArrayOf{std::make_shared<DataType>(DataType::Index), alloc_dim});
  tv_metadata.types["alloc_stride"] = NVFUSER_MAYBE_MAKE_SHARED2(
      ArrayOf{std::make_shared<DataType>(DataType::Index), alloc_dim});
  return tv_metadata;
}

DataType metaDataTypeOf(const Val* v) {
  auto tv = dynamic_cast<const TensorView*>(v);
  TORCH_INTERNAL_ASSERT(
      tv != nullptr, "Currently, only supports getting metadata of TensorView");
  if (tv->getMemoryType() == MemoryType::Shared) {
    // Smem tensor is defined locally as a pointer
    return PointerOf{std::make_shared<DataType>(tv->dtype())};
  }

  size_t dim = TensorDomain::noReductions(tv->getMaybeRFactorDomain()).size();
  size_t alloc_dim =
      TensorDomain::noReductions(tv->getMaybeAllocationDomain()).size();
  return globalTensorMetaData(tv->dtype(), dim, alloc_dim);
}

PrimDataType indexModeToDtype(KernelIndexMode index_mode) {
  switch (index_mode) {
    case KernelIndexMode::INT32:
      return DataType::Int32;
    case KernelIndexMode::INT64:
      return DataType::Int;
    default:
      TORCH_CHECK(false, "Invalid kernel index mode type.");
  }
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
       base_type == DataType::Half || base_type == DataType::BFloat16)) {
    return true;
  }
  if ((wider_type == DataType::Float || wider_type == DataType::ComplexFloat) &&
      (base_type == DataType::Float || base_type == DataType::Half ||
       base_type == DataType::BFloat16)) {
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
      TORCH_INTERNAL_ASSERT(
          false,
          "Only support ComplexFloat and ComplexDouble, current type:",
          dtype);
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
      TORCH_INTERNAL_ASSERT(
          false, "Only support Float and Double, current type:", dtype);
  }
}

bool isSupportedTypeByDevice(DataType dtype) {
  auto prop = at::cuda::getCurrentDeviceProperties();
  auto major_ver = prop->major;
  if (dtype == DataType::BFloat16) {
    return major_ver >= 8;
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
  TORCH_CHECK(false, "Expected promotable ValTypes but got: ", t1, " and ", t2);
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
            case DataType::Int:
              return "int64_t";
            case DataType::Index:
              return "nvfuser_index_t";
            case DataType::Int32:
              return "int";
            case DataType::SMemAddress:
              return "unsigned";
            case DataType::ComplexFloat:
              return "std::complex<float>";
            case DataType::ComplexDouble:
              return "std::complex<double>";
            case DataType::Opaque:
              return "std::any";
            default:
              TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
          }
        } else if constexpr (std::is_same_v<T, PointerOf>) {
          return data_type2string(*dtype.type) + "*";
        } else if constexpr (std::is_same_v<T, ArrayOf>) {
          std::stringstream ss;
          ss << "Array<" << data_type2string(*dtype.type) << ", " << dtype.size
             << ", 1>";
          return ss.str();
        } else if constexpr (std::is_same_v<T, StructOf>) {
          if (dtype.name != "") {
            return dtype.name;
          }
          std::stringstream ss;
          ss << "struct { ";
          for (auto& name : dtype.field_names) {
            ss << data_type2string(NVFUSER_MAYBE_STAR dtype.types.at(name))
               << " " << name << "; ";
          }
          ss << "}";
          return ss.str();
        } else {
          TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
        }
        TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
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
    case ValType::PipelineVal:
      return "PipelineVal";
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for val type.");
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
    case PredicateType::Shift:
      return "Shift";
    case PredicateType::Padding:
      return "Padding";
    case PredicateType::ReductionWrite:
      return "ReductionWrite";
    case PredicateType::LoopRotation:
      return "LoopRotation";
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for predicate type.");
  }
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
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for unary op type.");
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
      return "(int64_t) &";
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
    default:
      TORCH_INTERNAL_ASSERT(false, "No string found for binary op type.");
  }
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
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected TernaryOpType");
  }
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
      TORCH_INTERNAL_ASSERT(false, "Unexpected RNGOpType");
  }
}

static const char* parallel_type2string(ParallelType t) {
  switch (t) {
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
    case ParallelType::Vectorize:
      return "V";
    case ParallelType::MisalignedVectorize:
      return "MV";
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
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected ParallelType");
  }
}

std::unordered_set<ParallelType> allParallelTypesExcept(
    const std::unordered_set<ParallelType>& except) {
  std::unordered_set<ParallelType> result = {
      ParallelType::BIDz,
      ParallelType::BIDy,
      ParallelType::BIDx,
      ParallelType::TIDz,
      ParallelType::TIDy,
      ParallelType::TIDx,
      ParallelType::Vectorize,
      ParallelType::MisalignedVectorize,
      ParallelType::Unroll,
      ParallelType::Unswitch,
      ParallelType::Mma,
      ParallelType::Group,
      ParallelType::Serial};
  for (auto t : except) {
    result.erase(t);
  }
  return result;
}

static const char* memory_type2string(MemoryType t) {
  switch (t) {
    case MemoryType::Local:
      return "register";
    case MemoryType::Shared:
      return "shared";
    case MemoryType::Global:
      return "global";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected MemoryType");
  }
}

static const char* id_map_mode_type2string(IdMappingMode t) {
  switch (t) {
    case IdMappingMode::EXACT:
      return "exact";
    case IdMappingMode::ALMOSTEXACT:
      return "almost_exact";
    case IdMappingMode::PERMISSIVE:
      return "permissive";
    case IdMappingMode::LOOP:
      return "loop";
    default:
      // Don't try to print t as it would recursively call this function
      TORCH_INTERNAL_ASSERT(false, "Unexpected IdMappingMode Type.");
  }
}

static const char* iter_type2string(IterType t) {
  switch (t) {
    case IterType::Iteration:
      return "i";
    case IterType::Reduction:
      return "r";
    case IterType::Broadcast:
      return "b";
    case IterType::Gather:
      return "g";
    case IterType::Stride:
      return "s";
    case IterType::GatherScatter:
      return "n";
    case IterType::VectorComponent:
      return "v";
    case IterType::Symbolic:
      return "?";
    default:
      // Don't try to print t as it would recursively call this function
      TORCH_INTERNAL_ASSERT(false, "Unexpected IterType");
  }
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
      TORCH_INTERNAL_ASSERT(false, "Unexpected parallel type");
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
    case LoadStoreOpType::LdMatrixTranspose:
      return "LdMatrixTranspose";
    case LoadStoreOpType::CpAsyncCa:
      return "CpAsyncCa";
    case LoadStoreOpType::CpAsyncCg:
      return "CpAsyncCg";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected parallel type");
  }
}

const unsigned int _WORD_SHIFT = 16;
constexpr unsigned int supported_switch_pair(PrimDataType t1, PrimDataType t2) {
  return ((unsigned int)t1 << _WORD_SHIFT) + (unsigned int)t2;
}

static const char* supported_casts2string(
    const std::pair<DataType, DataType>& t) {
  switch (supported_switch_pair(
      std::get<PrimDataType>(t.first.type),
      std::get<PrimDataType>(t.second.type))) {
    case supported_switch_pair(DataType::Index, DataType::Float):
    case supported_switch_pair(DataType::Int, DataType::Float):
    case supported_switch_pair(DataType::Int32, DataType::Float):
    case supported_switch_pair(DataType::Double, DataType::Float):
    case supported_switch_pair(DataType::Bool, DataType::Float):
      return "(float)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Float):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Float):
      return "(float)std::real";
    case supported_switch_pair(DataType::Index, DataType::Int):
    case supported_switch_pair(DataType::Int32, DataType::Int):
    case supported_switch_pair(DataType::Float, DataType::Int):
    case supported_switch_pair(DataType::Double, DataType::Int):
    case supported_switch_pair(DataType::Bool, DataType::Int):
      return "(int64_t)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Int):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Int):
      return "(int64_t)std::real";
    case supported_switch_pair(DataType::Index, DataType::Int32):
    case supported_switch_pair(DataType::Int, DataType::Int32):
    case supported_switch_pair(DataType::Float, DataType::Int32):
    case supported_switch_pair(DataType::Double, DataType::Int32):
    case supported_switch_pair(DataType::Bool, DataType::Int32):
      return "(int32_t)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Int32):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Int32):
      return "(int32_t)std::real";
    case supported_switch_pair(DataType::Int, DataType::Index):
    case supported_switch_pair(DataType::Int32, DataType::Index):
    case supported_switch_pair(DataType::Float, DataType::Index):
    case supported_switch_pair(DataType::Double, DataType::Index):
      return "(nvfuser_index_t)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Index):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Index):
      return "(nvfuser_index_t)std::real";
    case supported_switch_pair(DataType::Index, DataType::Double):
    case supported_switch_pair(DataType::Int, DataType::Double):
    case supported_switch_pair(DataType::Int32, DataType::Double):
    case supported_switch_pair(DataType::Float, DataType::Double):
    case supported_switch_pair(DataType::Bool, DataType::Double):
      return "(double)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Double):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Double):
      return "(double)std::real";
    case supported_switch_pair(DataType::Float, DataType::Bool):
    case supported_switch_pair(DataType::Double, DataType::Bool):
    case supported_switch_pair(DataType::Int32, DataType::Bool):
    case supported_switch_pair(DataType::Int, DataType::Bool):
      return "(bool)";
    case supported_switch_pair(DataType::ComplexFloat, DataType::Bool):
    case supported_switch_pair(DataType::ComplexDouble, DataType::Bool):
      return "(bool)std::real";
    case supported_switch_pair(DataType::Index, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Int, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Int32, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Double, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Float, DataType::ComplexDouble):
    case supported_switch_pair(DataType::Bool, DataType::ComplexDouble):
    case supported_switch_pair(DataType::ComplexFloat, DataType::ComplexDouble):
      return "(std::complex<double>)";
    case supported_switch_pair(DataType::Index, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Int, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Int32, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Double, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Float, DataType::ComplexFloat):
    case supported_switch_pair(DataType::Bool, DataType::ComplexFloat):
    case supported_switch_pair(DataType::ComplexDouble, DataType::ComplexFloat):
      return "(std::complex<float>)";

    case supported_switch_pair(DataType::Float, DataType::Half):
      return "__float2half";
    case supported_switch_pair(DataType::Double, DataType::Half):
      return "__double2half";
    case supported_switch_pair(DataType::Int32, DataType::Half):
      return "__int322half";
    case supported_switch_pair(DataType::Int, DataType::Half):
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
    case supported_switch_pair(DataType::Half, DataType::Int32):
      return "__half2int32";
    case supported_switch_pair(DataType::Half, DataType::Int):
      return "__half2int";
    case supported_switch_pair(DataType::Half, DataType::Bool):
      return "__half2bool";
    case supported_switch_pair(DataType::Half, DataType::ComplexFloat):
      return "(std::complex<float>)__half2float";
    case supported_switch_pair(DataType::Half, DataType::ComplexDouble):
      return "(std::complex<double>)__half2double";

    case supported_switch_pair(DataType::Float, DataType::BFloat16):
      return "__float2bfloat";
    case supported_switch_pair(DataType::Double, DataType::BFloat16):
      return "__double2bfloat";
    case supported_switch_pair(DataType::Half, DataType::BFloat16):
      return "__half2bfloat";
    case supported_switch_pair(DataType::Int32, DataType::BFloat16):
      return "__int322bfloat";
    case supported_switch_pair(DataType::Int, DataType::BFloat16):
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
    case supported_switch_pair(DataType::BFloat16, DataType::Int32):
      return "__bfloat2int32";
    case supported_switch_pair(DataType::BFloat16, DataType::Int):
      return "__bfloat2int";
    case supported_switch_pair(DataType::BFloat16, DataType::Bool):
      return "__bfloat2bool";
    case supported_switch_pair(DataType::BFloat16, DataType::ComplexFloat):
      return "(std::complex<float>)__bfloat2float";
    case supported_switch_pair(DataType::BFloat16, DataType::ComplexDouble):
      return "(std::complex<double>)__bfloat2double";

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
    case at::ScalarType::Long:
      return DataType::Int;
    case at::ScalarType::Int:
      return DataType::Int32;
    case at::ScalarType::ComplexFloat:
      return DataType::ComplexFloat;
    case at::ScalarType::ComplexDouble:
      return DataType::ComplexDouble;
    default:
      return DataType::Null;
  }
}

at::ScalarType data_type_to_aten(const DataType& data_type) {
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
    case DataType::Int:
      return at::ScalarType::Long;
    case DataType::Index:
      TORCH_INTERNAL_ASSERT(
          false,
          "Index is determined at compile time,",
          " to convert from an aten type you need to have the compiled information. ",
          "This information is passed to GpuLower at compile time, and then copied to kerned.",
          "There's also this information in FusionExecutorCache and the Registry system.");
    case DataType::Int32:
      return at::ScalarType::Int;
    case DataType::ComplexFloat:
      return at::ScalarType::ComplexFloat;
    case DataType::ComplexDouble:
      return at::ScalarType::ComplexDouble;
    default:
      TORCH_INTERNAL_ASSERT(false, "No data type found for scalar type.");
  }
}

std::ostream& operator<<(std::ostream& out, const ValType vtype) {
  return out << val_type2string(vtype);
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

std::ostream& operator<<(std::ostream& out, const ScatterOpType sotype) {
  if (sotype == ScatterOpType::Set) {
    return out << "scatter";
  }
  TORCH_INTERNAL_ASSERT(false, "No scatterOp type found for scatterOp.");
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
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined 2D swizzle");
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
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined 2D swizzle");
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
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined index mode");
      break;
  }
  return os;
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
  if (std::holds_alternative<PointerOf>(data_type.type)) {
    return "ptr";
  }
  if (std::holds_alternative<ArrayOf>(data_type.type)) {
    return "a";
  }
  if (std::holds_alternative<StructOf>(data_type.type)) {
    return "s";
  }
  switch (std::get<PrimDataType>(data_type.type)) {
    case DataType::Bool:
      return "b";
    case DataType::Double:
      return "d";
    case DataType::Float:
    case DataType::Half:
    case DataType::BFloat16:
      return "f";
    case DataType::Index:
    case DataType::Int:
    case DataType::Int32:
    case DataType::SMemAddress:
      return "i";
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
      return "c";
    case DataType::Opaque:
      return "opaque";
    default:
      TORCH_INTERNAL_ASSERT(false, "No data type found for scalar type.");
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

bool isParallelTypeThread(ParallelType ptype) {
  return isParallelTypeBlockDim(ptype) || isParallelTypeThreadDim(ptype);
}

bool isParallelTypeVectorize(ParallelType ptype) {
  return ptype == ParallelType::Vectorize ||
      ptype == ParallelType::MisalignedVectorize;
}

std::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>& cast) {
  const char* str = supported_casts2string(cast);
  return str != nullptr ? std::optional<std::string>(std::string(str))
                        : std::nullopt;
}

int64_t dataTypeSize(DataType type) {
  return std::visit(
      [](auto&& dtype) -> int64_t {
        using T = std::decay_t<decltype(dtype)>;
        if constexpr (std::is_same_v<T, PrimDataType>) {
          return primDataTypeSize(dtype);
        } else if constexpr (std::is_same_v<T, PointerOf>) {
          return sizeof(void*);
        } else if constexpr (std::is_same_v<T, ArrayOf>) {
          return dataTypeSize(*dtype.type) * dtype.size;
        }
        TORCH_INTERNAL_ASSERT(false, "Size undefined for data type.");
      },
      type.type);
}

int64_t dataTypeSize(DataType type, DataType index_type) {
  if (type == DataType::Index) {
    TORCH_INTERNAL_ASSERT(
        index_type == DataType::Int32 || index_type == DataType::Int,
        "Invalid index type of ",
        index_type);
    return dataTypeSize(index_type);
  }
  return dataTypeSize(type);
}

std::ostream& operator<<(
    std::ostream& os,
    const DoubleBufferLoopStage loop_stage) {
  switch (loop_stage) {
    case DoubleBufferLoopStage::NotApplicable:
      break;
    case DoubleBufferLoopStage::Prolog:
      os << "{DoubleBufferProlog}";
      break;
    case DoubleBufferLoopStage::Main:
      os << "{DoubleBufferMainLoop}";
      break;
    case DoubleBufferLoopStage::Epilog:
      os << "{DoubleBufferEpilog}";
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown double buffer stage");
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
  } else {
    TORCH_CHECK(
        !isFloatingPointType(dtype),
        "Unhandled floating point type in max_digits10 ",
        dtype);
    return 0;
  }
}

} // namespace nvfuser

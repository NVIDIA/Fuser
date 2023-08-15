#include <polymorphic_value.h>
#include <serde/utils.h>

namespace nvfuser::serde {

serde::UnaryOpType mapToSerdeUnaryOp(nvfuser::UnaryOpType t) {
  switch (t) {
    case nvfuser::UnaryOpType::Cast:
      return serde::UnaryOpType_Cast;
    case nvfuser::UnaryOpType::Neg:
      return serde::UnaryOpType_Neg;
    default:
      return serde::UnaryOpType_None;
  }
}

serde::BinaryOpType mapToSerdeBinaryOp(nvfuser::BinaryOpType t) {
  switch (t) {
    case nvfuser::BinaryOpType::Add:
      return serde::BinaryOpType_Add;
    case nvfuser::BinaryOpType::CeilDiv:
      return serde::BinaryOpType_CeilDiv;
    case nvfuser::BinaryOpType::Div:
      return serde::BinaryOpType_Div;
    case nvfuser::BinaryOpType::Mod:
      return serde::BinaryOpType_Mod;
    case nvfuser::BinaryOpType::Mul:
      return serde::BinaryOpType_Mul;
    case nvfuser::BinaryOpType::Sub:
      return serde::BinaryOpType_Sub;
    default:
      return serde::BinaryOpType_None;
  }
}

serde::DataType mapToSerdeDtype(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Bool:
      return serde::DataType_Bool;
    case at::ScalarType::Double:
      return serde::DataType_Double;
    case at::ScalarType::Float:
      return serde::DataType_Float;
    case at::ScalarType::Half:
      return serde::DataType_Half;
    case at::ScalarType::BFloat16:
      return serde::DataType_BFloat16;
    case at::ScalarType::Long:
      return serde::DataType_Int;
    case at::ScalarType::Int:
      return serde::DataType_Int32;
    case at::ScalarType::ComplexFloat:
      return serde::DataType_ComplexFloat;
    case at::ScalarType::ComplexDouble:
      return serde::DataType_ComplexDouble;
    default:
      return serde::DataType_None;
  }
}

at::ScalarType mapToAtenDtype(serde::DataType t) {
  switch (t) {
    case serde::DataType_Bool:
      return at::ScalarType::Bool;
    case serde::DataType_Double:
      return at::ScalarType::Double;
    case serde::DataType_Float:
      return at::ScalarType::Float;
    case serde::DataType_Half:
      return at::ScalarType::Half;
    case serde::DataType_BFloat16:
      return at::ScalarType::BFloat16;
    case serde::DataType_Int:
      return at::ScalarType::Long;
    case serde::DataType_Int32:
      return at::ScalarType::Int;
    case serde::DataType_ComplexFloat:
      return at::ScalarType::ComplexFloat;
    case serde::DataType_ComplexDouble:
      return at::ScalarType::ComplexDouble;
    case serde::DataType_None:
      return at::ScalarType::Undefined;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No nvfuser dtype found for serde data type.");
  return at::ScalarType::Undefined;
}

serde::DataType mapToSerdeDtype(nvfuser::DataType t) {
  return mapToSerdeDtype(std::get<PrimDataType>(t.type));
}

nvfuser::DataType mapToDtypeStruct(serde::DataType t) {
  return nvfuser::DataType(mapToNvfuserDtype(t));
}

serde::DataType mapToSerdeDtype(PrimDataType t) {
  switch (t) {
    case PrimDataType::Index:
      return serde::DataType_Index;
    case PrimDataType::Bool:
      return serde::DataType_Bool;
    case PrimDataType::Double:
      return serde::DataType_Double;
    case PrimDataType::Float:
      return serde::DataType_Float;
    case PrimDataType::Half:
      return serde::DataType_Half;
    case PrimDataType::BFloat16:
      return serde::DataType_BFloat16;
    case PrimDataType::Int:
      return serde::DataType_Int;
    case PrimDataType::Int32:
      return serde::DataType_Int32;
    case PrimDataType::ComplexFloat:
      return serde::DataType_ComplexFloat;
    case PrimDataType::ComplexDouble:
      return serde::DataType_ComplexDouble;
    case PrimDataType::Null:
      return serde::DataType_None;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No serde dtype found for nvfuser data type.");
  return serde::DataType_MAX;
}

PrimDataType mapToNvfuserDtype(serde::DataType t) {
  switch (t) {
    case serde::DataType_Index:
      return PrimDataType::Index;
    case serde::DataType_Bool:
      return PrimDataType::Bool;
    case serde::DataType_Double:
      return PrimDataType::Double;
    case serde::DataType_Float:
      return PrimDataType::Float;
    case serde::DataType_Half:
      return PrimDataType::Half;
    case serde::DataType_BFloat16:
      return PrimDataType::BFloat16;
    case serde::DataType_Int:
      return PrimDataType::Int;
    case serde::DataType_Int32:
      return PrimDataType::Int32;
    case serde::DataType_ComplexFloat:
      return PrimDataType::ComplexFloat;
    case serde::DataType_ComplexDouble:
      return PrimDataType::ComplexDouble;
    case serde::DataType_None:
      return PrimDataType::Null;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No nvfuser dtype found for serde data type.");
  return PrimDataType::Null;
}

std::vector<bool> parseBoolVector(
    const flatbuffers::Vector<uint8_t>* fb_vector) {
  std::vector<bool> result(fb_vector->begin(), fb_vector->end());
  return result;
}

} // namespace nvfuser::serde

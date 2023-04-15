#include <serde/utils.h>

namespace nvfuser::serde {

serde::DataType mapToSerdeDtype(nvfuser::DataType t) {
  return mapToSerdeDtype(std::get<PrimDataType>(t.type));
}

nvfuser::DataType mapToDtypeStruct(serde::DataType t) {
  return nvfuser::DataType(mapToNvfuserDtype(t));
}

serde::DataType mapToSerdeDtype(PrimDataType t) {
  switch (t) {
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

} // namespace nvfuser::serde

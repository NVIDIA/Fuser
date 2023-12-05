#include <polymorphic_value.h>
#include <serde/utils.h>

namespace nvfuser::serde {

at::ScalarType mapToAtenDtype(long data_type) {
  return static_cast<at::ScalarType>(data_type);
}

nvfuser::DataType mapToDtypeStruct(long data_type) {
  return nvfuser::DataType(mapToNvfuserDtype(data_type));
}

PrimDataType mapToNvfuserDtype(long data_type) {
  return static_cast<PrimDataType>(data_type);
}

std::vector<bool> parseBoolVector(
    const flatbuffers::Vector<uint8_t>* fb_vector) {
  std::vector<bool> result(fb_vector->begin(), fb_vector->end());
  return result;
}

} // namespace nvfuser::serde

set(FlatBuffers_Include ${PROJECT_SOURCE_DIR}/../third_party/flatbuffers/include)
file(GLOB FlatBuffers_Library_SRCS
  ${FlatBuffers_Include}/flatbuffers/*.h
)
add_library(flatbuffers INTERFACE)
target_sources(
  flatbuffers
  INTERFACE ${FlatBuffers_Library_SRCS}
)
target_include_directories(flatbuffers INTERFACE ${FlatBuffers_Include})
include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/flatbuffers/include)

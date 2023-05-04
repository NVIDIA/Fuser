
set(FlatBuffers_Src_Dir ${PROJECT_SOURCE_DIR}/third_party/flatbuffers)

option(FLATBUFFERS_BUILD_TESTS "Enable the build of tests and samples." OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" ON)
option(FLATBUFFERS_STATIC_FLATC "Build flatbuffers compiler with -static flag" OFF)
option(FLATBUFFERS_BUILD_FLATHASH "Enable the build of flathash" OFF)

# Add FlatBuffers directly to our build. This defines the `flatbuffers` target.
add_subdirectory(${FlatBuffers_Src_Dir})
include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/flatbuffers/include)

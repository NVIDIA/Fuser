cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nvfuser_matmul_heuristic_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(nvfuser_include_dir /opt/pytorch/nvfuser/csrc CACHE FILEPATH REQUIRED)
set(nvfuser_lib_dir /opt/pytorch/nvfuser/build CACHE FILEPATH REQUIRED)

add_library(
    matmul_heuristic_plugin
    SHARED
    matmul_heuristic_plugin.cpp)
target_include_directories(
    matmul_heuristic_plugin
    PUBLIC
    ${nvfuser_include_dir})

list(APPEND CMAKE_INSTALL_RPATH
    ${nvfuser_lib_dir})

target_link_directories(matmul_heuristic_plugin
    PUBLIC
    ${nvfuser_lib_dir})
target_link_libraries(
    matmul_heuristic_plugin
    PUBLIC
    nvfuser_codegen)

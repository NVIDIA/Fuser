# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sinh_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(Nvfuser REQUIRED)

add_executable(sinh_example main.cpp)
target_link_libraries(sinh_example ${TORCH_LIBRARIES})
target_link_libraries(sinh_example nvfuser_codegen)

# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nvfuser_matmul_heuristic_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(
    matmul_heuristic_plugin
    SHARED
    matmul_heuristic_plugin.cpp)

target_include_directories(
    matmul_heuristic_plugin
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR})

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <exceptions.h>
#include <serde/fusion_cache_generated.h>
#include <type.h>

namespace nvfuser::serde {

//! A function to map the serde dtype to its corresponding nvfuser prim dtype
PrimDataType mapToNvfuserDtype(long data_type);

//! A function to map the serde dtype to its corresponding nvfuser datatype
nvfuser::DataType mapToDtypeStruct(long data_type);

//! A function to map the serde dtype to its corresponding aten dtype
at::ScalarType mapToAtenDtype(long data_type);

template <typename T>
std::vector<T> parseVector(const flatbuffers::Vector<T>* fb_vector) {
  std::vector<T> result(fb_vector->begin(), fb_vector->end());
  return result;
}

// Flatbuffer stores bool values as uint8_t.
std::vector<bool> parseBoolVector(
    const flatbuffers::Vector<uint8_t>* fb_vector);

} // namespace nvfuser::serde

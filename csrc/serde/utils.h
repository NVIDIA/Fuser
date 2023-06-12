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

//! A function to map the nvfuser prim datatype to the corresponding serde dtype
serde::DataType mapToSerdeDtype(PrimDataType t);

//! A function to map the nvfuser datatype to the corresponding serde dtype
serde::DataType mapToSerdeDtype(nvfuser::DataType t);

//! A function to map the aten dtype to the corresponding serde dtype
serde::DataType mapToSerdeDtype(at::ScalarType t);

//! A function to map the serde dtype to its corresponding nvfuser prim dtype
PrimDataType mapToNvfuserDtype(serde::DataType t);

//! A function to map the serde dtype to its corresponding nvfuser datatype
nvfuser::DataType mapToDtypeStruct(serde::DataType t);

//! A function to map the serde dtype to its corresponding aten dtype
at::ScalarType mapToAtenDtype(serde::DataType t);

template <typename T>
std::vector<T> parseVector(const flatbuffers::Vector<T>* fb_vector) {
  std::vector<T> result(fb_vector->begin(), fb_vector->end());
  return result;
}

// Flatbuffer stores bool values as uint8_t.
std::vector<bool> parseBoolVector(
    const flatbuffers::Vector<uint8_t>* fb_vector);

} // namespace nvfuser::serde

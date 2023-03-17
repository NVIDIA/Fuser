// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser::python_frontend {

// Flatbuffer enum are represented as an unscoped enumeration, so we can map
// them to an Integer type. This Factory class contains a vector that maps from
// an enum integer to its corresponding parser function.
//
// All parser functions have the same signature. We use lambdas to support
// functions that require extra arguments.

template <typename SerdeBuffer, typename BaseType>
class Factory {
 public:
  // A function pointer that creates a BaseType object given a Buffer
  typedef std::function<BaseType*(const SerdeBuffer*)> SerdeParser;

  Factory(size_t num_parsers) : parsers_(num_parsers, nullptr){};

  void registerParser(int serde_type, SerdeParser parser) {
    TORCH_CHECK(
        serde_type >= 0 && serde_type < (int)parsers_.size(),
        "RegisterParser: Invalid serde type: ",
        serde_type);
    parsers_.at(serde_type) = parser;
  }

  BaseType* parse(int serde_type, const SerdeBuffer* buffer) {
    TORCH_CHECK(
        serde_type >= 0 && serde_type < (int)parsers_.size(),
        "Deserialize: Invalid serde type: ",
        serde_type);
    return parsers_.at(serde_type)(buffer);
  }

 private:
  std::vector<SerdeParser> parsers_;
};

} // namespace nvfuser::python_frontend

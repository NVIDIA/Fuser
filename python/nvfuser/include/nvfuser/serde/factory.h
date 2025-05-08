// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <type.h>
#include <functional>

namespace nvfuser::serde {

// Flatbuffer enum are represented as an unscoped enumeration, so we can map
// them to an Integer type. This Factory class contains a vector that maps from
// an enum integer to its corresponding parser function.
//
// All parser functions have the same signature. We use lambdas to support
// functions that require extra arguments.

template <typename SerdeBuffer, typename BaseTypePtr>
class Factory {
 public:
  // A function pointer that creates a BaseType object given a Buffer
  typedef std::function<BaseTypePtr(const SerdeBuffer*)> SerdeParser;

  Factory(size_t num_parsers) : parsers_(num_parsers, nullptr) {};

  template <typename SerdeEnum>
  void registerParser(SerdeEnum serde_type, SerdeParser parser) {
    auto serde_integer = nvfuser::toUnderlying(serde_type);
    NVF_ERROR(
        serde_integer >= 0 && serde_integer < (int)parsers_.size(),
        "RegisterParser: Invalid serde type: ",
        serde_integer);
    parsers_.at(serde_integer) = parser;
  }

  template <typename SerdeEnum>
  BaseTypePtr parse(SerdeEnum serde_type, const SerdeBuffer* buffer) {
    auto serde_integer = nvfuser::toUnderlying(serde_type);
    NVF_ERROR(
        serde_integer >= 0 && serde_integer < (int)parsers_.size(),
        "Deserialize: Invalid serde type: ",
        serde_integer);
    return parsers_.at(serde_integer)(buffer);
  }

 private:
  std::vector<SerdeParser> parsers_;
};

} // namespace nvfuser::serde

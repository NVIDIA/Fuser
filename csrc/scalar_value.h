// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <dynamic_type.h>
#include <complex>
#include <cstddef>
#include <unordered_map>

#include <type.h>

namespace nvfuser {

using ScalarValue =
    DynamicType<NoContainers, std::complex<double>, double, int64_t, bool>;

namespace ScalarValue_functions {

inline ScalarValue ceildiv(const ScalarValue& a, const ScalarValue& b) {
  if (a.is<int64_t>() && b.is<int64_t>()) {
    auto aa = a.as<int64_t>();
    auto bb = b.as<int64_t>();
    if (bb > 0) {
      return ScalarValue((aa + bb - 1) / bb);
    } else {
      return ScalarValue((aa + bb + 1) / bb);
    }
  }
  return ScalarValue(std::ceil((a / b).as<double>()));
}

inline ScalarValue max(const ScalarValue& a, const ScalarValue& b) {
  return ScalarValue(a > b ? a : b);
}

inline ScalarValue min(const ScalarValue& a, const ScalarValue& b) {
  return ScalarValue(a < b ? a : b);
}

inline ScalarValue gcd(const ScalarValue& a, const ScalarValue& b) {
  return ScalarValue(std::gcd(a.as<int64_t>(), b.as<int64_t>()));
}

inline ScalarValue notExpr(const ScalarValue& a) {
  if (a.is<int64_t>()) {
    return ScalarValue(~a.as<int64_t>());
  }
  if (a.is<bool>()) {
    return ScalarValue(!a.as<bool>());
  }
  TORCH_INTERNAL_ASSERT(false);
}

inline ScalarValue abs(const ScalarValue& a) {
  if (a.is<int64_t>()) {
    return ScalarValue(std::abs(a.as<int64_t>()));
  }
  if (a.is<double>()) {
    return ScalarValue(std::abs(a.as<double>()));
  }
  if (a.is<bool>()) {
    return a;
  }
  TORCH_INTERNAL_ASSERT(false);
}

} // namespace ScalarValue_functions

} // namespace nvfuser

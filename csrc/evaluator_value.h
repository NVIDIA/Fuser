// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dynamic_type.h>

#include <complex>
#include <cstdint>
#include <cmath>
#include <numeric>

namespace nvfuser {

using EvaluatorValue = DynamicType<std::complex<double>, double, int64_t, bool>;

namespace EvaluatorValue_functions {

inline EvaluatorValue ceildiv(
    const EvaluatorValue& a,
    const EvaluatorValue& b) {
  if (a.is<int64_t>() && b.is<int64_t>()) {
    auto aa = a.as<int64_t>();
    auto bb = b.as<int64_t>();
    if (bb > 0) {
      return EvaluatorValue((aa + bb - 1) / bb);
    } else {
      return EvaluatorValue((aa + bb + 1) / bb);
    }
  }
  return EvaluatorValue(std::ceil((a / b).as<double>()));
}

inline EvaluatorValue max(const EvaluatorValue& a, const EvaluatorValue& b) {
  return EvaluatorValue((a > b).as<bool>() ? a : b);
}

inline EvaluatorValue min(const EvaluatorValue& a, const EvaluatorValue& b) {
  return EvaluatorValue((a < b).as<bool>() ? a : b);
}

inline EvaluatorValue gcd(const EvaluatorValue& a, const EvaluatorValue& b) {
  return EvaluatorValue(std::gcd(a.as<int64_t>(), b.as<int64_t>()));
}

inline EvaluatorValue notExpr(const EvaluatorValue& a) {
  if (a.is<int64_t>()) {
    return EvaluatorValue(~a.as<int64_t>());
  }
  if (a.is<bool>()) {
    return EvaluatorValue(!a.as<bool>());
  }
  TORCH_INTERNAL_ASSERT(false);
}

inline EvaluatorValue abs(const EvaluatorValue& a) {
  if (a.is<int64_t>()) {
    return EvaluatorValue(std::abs(a.as<int64_t>()));
  }
  if (a.is<double>()) {
    return EvaluatorValue(std::abs(a.as<double>()));
  }
  if (a.is<bool>()) {
    return a;
  }
  TORCH_INTERNAL_ASSERT(false);
}

} // namespace EvaluatorValue_functions

} // namespace nvfuser

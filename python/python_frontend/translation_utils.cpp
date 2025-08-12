// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
#include <ops/all_ops.h>
#include <python_frontend/translation_utils.h>

namespace nvfuser::python_frontend {

#define GET_FUNCTION_TERNARY_SPECIALIZATION_DEFINITION(                      \
    ResultType, InType1, InType2, InType3)                                   \
  template <>                                                                \
  std::function<ResultType(InType1, InType2, InType3)>                       \
  getFunction<ResultType, InType1, InType2, InType3>(const TernaryOp* top) { \
    auto wrap_function = [](ResultType (*fn)(InType1, InType2, InType3)) {   \
      return fn;                                                             \
    };                                                                       \
                                                                             \
    switch (top->getTernaryOpType()) {                                       \
      case TernaryOpType::Clamp:                                             \
        return wrap_function(clamp);                                         \
        break;                                                               \
      case TernaryOpType::Lerp:                                              \
        return wrap_function(lerp);                                          \
        break;                                                               \
      case TernaryOpType::Threshold:                                         \
        return wrap_function(threshold);                                     \
        break;                                                               \
      case TernaryOpType::Where:                                             \
        return wrap_function(where);                                         \
        break;                                                               \
      default:                                                               \
        NVF_CHECK(                                                           \
            false,                                                           \
            "Unexpected operator type: ",                                    \
            top->getTernaryOpType(),                                         \
            " in ",                                                          \
            top->toString());                                                \
    }                                                                        \
  }

// Fully specialized template functions to create std::function for TernaryOp.
GET_FUNCTION_TERNARY_SPECIALIZATION_DEFINITION(
    TensorView*,
    TensorView*,
    Val*,
    Val*)
GET_FUNCTION_TERNARY_SPECIALIZATION_DEFINITION(Val*, Val*, Val*, Val*)

serde::RecordType getSerdeType(const ReductionOp* rop) {
  switch (rop->getReductionOpType()) {
    case BinaryOpType::Add:
      return serde::RecordType::ReductionSum;
      break;
    case BinaryOpType::Mul:
      return serde::RecordType::ReductionProd;
      break;
    case BinaryOpType::Max:
      return serde::RecordType::ReductionMax;
      break;
    case BinaryOpType::Min:
      return serde::RecordType::ReductionMin;
      break;
    default:
      NVF_CHECK(
          false,
          "Unexpected reduction operator type: ",
          rop->getReductionOpType(),
          " in ",
          rop->toString());
  }
}

} // namespace nvfuser::python_frontend

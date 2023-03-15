// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <ir_all_nodes.h>
#include <iter_visitor.h>
#include <transform_view.h>

#include <functional>
#include <memory>
#include <vector>

namespace nvfuser {

class Fusion;
class ExpressionEvaluator;
class DynamicTransformInfoBuilder;

class TORCH_CUDA_CU_API DynamicTransformInfo {
 public:
  bool operator==(const DynamicTransformInfo& other) const;

  bool operator!=(const DynamicTransformInfo& other) const {
    return !(*this == other);
  }

  std::string toString() const;

  static DynamicTransformInfo get(
      Fusion* fusion,
      ExpressionEvaluator* expr_eval);

  // TODO: Make it private
 public:
  Fusion* fusion_ = nullptr;
  std::vector<std::pair<TensorView*, AnalyzeViewResult>> reshape_transforms_;

  // TODO: resize transforms

  friend class DynamicTransformInfoBuilder;
};

//! Concretize dynamic transforms in
void concretizeFusion(
    Fusion* fusion,
    const DynamicTransformInfo& transform_info);

} // namespace nvfuser

// TODO: hash
namespace std {
template <>
struct hash<nvfuser::DynamicTransformInfo> {
  std::size_t operator()(const nvfuser::DynamicTransformInfo& info) const {
    return 0;
  }
};
} // namespace std

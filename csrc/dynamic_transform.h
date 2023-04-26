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

//! A set of transformations for a symbolic fusion with concrete sizes
//! of the fusion inputs
class TORCH_CUDA_CU_API DynamicTransformConcretizationInfo {
 public:
  const std::vector<std::pair<TensorView*, AnalyzeViewResult>>
  getReshapeTransforms() const {
    return reshape_transforms_;
  }

  bool operator==(const DynamicTransformConcretizationInfo& other) const;

  bool operator!=(const DynamicTransformConcretizationInfo& other) const {
    return !(*this == other);
  }

  Fusion* fusion() const {
    return fusion_;
  }

  std::string toString() const;

  size_t hash() const;

  DynamicTransformConcretizationInfo clone(IrCloner& ir_cloner) const;

 private:
  DynamicTransformConcretizationInfo(Fusion* fusion) : fusion_(fusion) {}

 private:
  Fusion* fusion_ = nullptr;
  std::vector<std::pair<TensorView*, AnalyzeViewResult>> reshape_transforms_;

  friend class DynamicTransformInfoBuilder;
};

class TORCH_CUDA_CU_API DynamicTransform {
 public:
  //! Get concrete transformations for a symbolic fusion with concrete
  //! input sizes given through an expression evaluator.
  static DynamicTransformConcretizationInfo getConcretizationInfo(
      Fusion* fusion,
      ExpressionEvaluator* expr_eval);

  //! Get concrete transformations for a symbolic fusion with concrete
  //! input sizes given through kernel arguments
  static DynamicTransformConcretizationInfo getConcretizationInfo(
      Fusion* fusion,
      const KernelArgumentHolder* args);

  //! Concretizes a given fusion. Note that the concretization is
  //! in-place and the given fusion is modified.
  static void concretizeFusion(
      Fusion*,
      const DynamicTransformConcretizationInfo& info);
};

} // namespace nvfuser

namespace std {
template <>
struct hash<nvfuser::DynamicTransformConcretizationInfo> {
  size_t operator()(
      const nvfuser::DynamicTransformConcretizationInfo& info) const {
    return info.hash();
  }
};
} // namespace std

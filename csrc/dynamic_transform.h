// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <iter_visitor.h>
#include <transform_view.h>

#include <functional>
#include <memory>
#include <vector>

namespace nvfuser {

class Fusion;
class DynamicTransformInitialInfoBuilder;

//! Initial information derived only from the symbolic Fusion without input
//! sizes
class TORCH_CUDA_CU_API DynamicTransformInitialInfo {
 public:
  bool operator==(const DynamicTransformConcretizationInfo& other) const;

  bool operator!=(const DynamicTransformConcretizationInfo& other) const {
    return !(*this == other);
  }

  Fusion* fusion() const {
    return fusion_;
  }

  //! Return whether any dynamic transforms exist in the Fusion
  bool hasDynamicTransforms() const {
    return !dynamic_reshapes_.empty() || !dynamic_resizes_.empty();
  }

  //! Return a set of scalars that are inputs or extents of input TensorViews
  //! and that appear in inputs to dynamic expressions. Any Vals not in this
  //! list do not affect concretization.
  const std::unordered_set<Val*> getRootDynamicVals() const {
    return root_dynamic_vals_;
  }

  //! Return a vector of ViewOp expressions that have dynamic output shapes
  const std::vector<ViewOp*>& getDynamicReshapes() const {
    return dynamic_reshapes_;
  }

  //! Return a vector of Resize expressions that have symbolic output IterTypes
  const std::vector<Resize*>& getDynamicResizes() const {
    return dynamic_resizes_;
  }

  const ExpressionEvaluator& getExpressionEvaluator() const {
    return expr_eval_;
  }

  std::string toString() const;

  DynamicTransformInitialInfo clone(IrCloner& ir_cloner) const;

  //! Return a set containing positions in inputs() holding any scalar input
  //! that would affect the structure of the concretized Fusion.
  const std::unordered_set<size_t>& scalarInputsAffectingConcretization()
      const {
    return scalar_inputs_affecting_concretization_;
  }

 protected:
  //! Holds the set of scalar fusion inputs that affect concretization.
  std::unordered_set<size_t> scalar_inputs_affecting_concretization_;

 private:
  DynamicTransformInitialInfo(Fusion* fusion) : fusion_(fusion) {}

 private:
  Fusion* fusion_ = nullptr;

  std::vector<ViewOp*> dynamic_reshapes_;

  std::vector<Resize*> dynamic_resizes_;

  // Root Vals that determine concretization
  std::unordered_set<Val*> root_dynamic_vals_;

  // ExpressionEvaluator that we use to pre-compute as much as possible
  ExpressionEvaluator expr_eval_;

  friend class DynamicTransformInitialInfoBuilder;
};

//! A set of transformations for a symbolic fusion with concrete sizes
//! of the fusion inputs
class TORCH_CUDA_CU_API DynamicTransformConcretizationInfo {
 public:
  DynamicTransformConcretizationInfo(
      Fusion* fusion,
      const DynamicTransformInitialInfo* info,
      ExpressionEvaluator* expr_eval)
      : fusion_(fusion) {
    TORCH_INTERNAL_ASSERT(
        !fusion->isA<kir::Kernel>(),
        "Invalid container. Kernel container not allowed.\n");

    // Make sure all exactly mapped IDs have the same value in the
    // evaluator when any one of the IDs has a known value
    expr_eval->propagateBoundValuesThroughExactMaps(fusion);

    analyzeReshapes(info, expr_eval);

    analyzeResizes(info, expr_eval);
  }

  const std::vector<std::pair<TensorView*, AnalyzeViewResult>>&
  getReshapeTransforms() const {
    return reshape_transforms_;
  }

  const std::vector<std::pair<IterDomain*, IterType>>& getResizeTransforms()
      const {
    return resize_transforms_;
  }

  bool operator==(const DynamicTransformConcretizationInfo& other) const;

  bool operator!=(const DynamicTransformConcretizationInfo& other) const {
    return !(*this == other);
  }

  void analyzeReshapes(
      const DynamicTransformInitialInfo* info,
      ExpressionEvaluator* expr_eval);

  void analyzeResizes(
      const DynamicTransformInitialInfo* info,
      ExpressionEvaluator* expr_eval);

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

  // Holds, for each dynamic reshape, the output TensorView, and the result of
  // analyzeView
  std::vector<std::pair<TensorView*, AnalyzeViewResult>> reshape_transforms_;

  // Holds the resized IterDomain (output of the Resize op) along with the
  // TensorView where it appears, and its concretized IterType
  std::vector<std::pair<IterDomain*, IterType>> resize_transforms_;

  friend class DynamicTransformInfoBuilder;
};

class TORCH_CUDA_CU_API DynamicTransform {
 public:
  //! Get initial information before we have inputs. This analyzes the Fusion to
  //! determine whether it has dynamic operations, and caches their position for
  //! faster concretization once inputs are available.
  static DynamicTransformInitialInfo getInitialInfo(Fusion* fusion);

  //! Get concrete transformations for a symbolic fusion with concrete
  //! input sizes given through an expression evaluator.
  static DynamicTransformConcretizationInfo getConcretizationInfo(
      Fusion* fusion,
      const DynamicTransformInitialInfo* info,
      ExpressionEvaluator* expr_eval);

  //! Get concrete transformations for a symbolic fusion with concrete
  //! input sizes given through kernel arguments
  static DynamicTransformConcretizationInfo getConcretizationInfo(
      Fusion* fusion,
      const DynamicTransformInitialInfo* info,
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

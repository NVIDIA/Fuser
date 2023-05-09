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
#include <ir_all_nodes.h>
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
    return dynamic_reshapes_.size() > 0;
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

  std::string toString() const;

  DynamicTransformInitialInfo clone(IrCloner& ir_cloner) const;

 private:
  DynamicTransformInitialInfo(Fusion* fusion) : fusion_(fusion) {}

 private:
  Fusion* fusion_ = nullptr;

  std::vector<ViewOp*> dynamic_reshapes_;

  // Root Vals that determine concretization
  std::unordered_set<Val*> root_dynamic_vals_;

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
  }

  const std::vector<std::pair<TensorView*, AnalyzeViewResult>>
  getReshapeTransforms() const {
    return reshape_transforms_;
  }

  bool operator==(const DynamicTransformConcretizationInfo& other) const;

  bool operator!=(const DynamicTransformConcretizationInfo& other) const {
    return !(*this == other);
  }

  void analyzeReshapes(
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
  std::vector<std::pair<TensorView*, AnalyzeViewResult>> reshape_transforms_;
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

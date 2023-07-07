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
#include <utils.h>

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

  //! Return whether any dynamic transforms exist in the Fusion, or whether
  //! there are any tensors which could potentially be empty (size-0 extent)
  //! given some user input. In either of these cases, concretization may change
  //! the structure of the Fusion.
  bool isDynamic() const {
    return hasPossibleEmptyTensor() || !dynamic_reshaped_tvs_.empty() ||
        !dynamic_resized_ids_.empty();
  }

  //! Return whether there are any tensors with unknown extent in some
  //! dimension, so that they might be empty
  bool hasPossibleEmptyTensor() const {
    return !maybe_zero_extents_.empty();
  }

  //! Return a set of scalars that are inputs or extents of input TensorViews
  //! and that appear in inputs to dynamic expressions. Any Vals not in this
  //! list do not affect concretization.
  const std::unordered_set<Val*>& getRootDynamicVals() const {
    return root_dynamic_vals_;
  }

  //! Return a set of scalars that appear as extents in TensorViews in the
  //! Fusion. If any of these evaluate to zero, there is at least one empty
  //! TensorView present.
  const std::vector<Val*>& getMaybeZeroExtents() const {
    return maybe_zero_extents_;
  }

  //! Return a vector of outputs of ViewOp expressions that have dynamic output
  //! shapes
  const std::vector<TensorView*>& getDynamicReshapedTensorViews() const {
    return dynamic_reshaped_tvs_;
  }

  //! Return a vector of outputs of Resize expressions that have symbolic output
  //! IterTypes
  const std::vector<IterDomain*>& getDynamicResizedIterDomains() const {
    return dynamic_resized_ids_;
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

  // We hold vectors of the _outputs_ of dynamic ops. The reason we don't hold
  // the ops themselves is that during concretization, the ops will actually be
  // removed by ir_utils::replaceValInExpr. The outputs will not: their
  // definitions will merely be altered. When the ops are replaced, if we had
  // referred to them directly here, we would run into segfaults. Referring only
  // to the outputs avoids this issue.
  std::vector<TensorView*> dynamic_reshaped_tvs_;

  std::vector<IterDomain*> dynamic_resized_ids_;

  // This is a minimal set of scalars to check for empty tensors. If any are
  // zero, we should traverse to find empty tensors.
  std::unordered_set<Val*> maybe_zero_extents_set_;
  // The set above is populated then used to create this unique vector
  std::vector<Val*> maybe_zero_extents_;

  // Root Vals that determine concretization
  std::unordered_set<Val*> root_dynamic_vals_;

  friend class DynamicTransformInitialInfoBuilder;
};

//! A set of transformations for a symbolic fusion with concrete sizes
//! of the fusion inputs
class TORCH_CUDA_CU_API DynamicTransformConcretizationInfo {
 public:
  DynamicTransformConcretizationInfo(
      const DynamicTransformInitialInfo* initial_info,
      ExpressionEvaluator* expr_eval);

  const std::vector<size_t>& getEmptyExtents() const {
    return empty_extents_;
  }

  //! Return a vector of pairs holding the index of each reshaped TensorView in
  //! the vector returned by initialInfo()->getDynamicReshapedTensorViews(),
  //! along with an AnalyzeViewResult describing how that reshape operation
  //! should be decomposed into split, merge, squeeze, and broadcast transforms.
  const std::vector<std::pair<size_t, AnalyzeViewResult>>& getReshapeTransforms()
      const {
    return reshape_transforms_;
  }

  //! Return a vector of pairs holding the index of each resized IterDomain in
  //! the vector returned by initialInfo()->getDynamicResizedIterDomains(),
  //! along with the IterType it should be concretized to.
  const std::vector<std::pair<size_t, IterType>>& getResizeIterTypes() const {
    return resize_itertypes_;
  }

  //! Comparison operator for the purposes of determining cache hits. This does
  //! not guarantee equality of all members. Instead, it returns equal if the
  //! resulting concretizations would be structurally equivalent. Note that
  //! pointers to Statements may differ between equivalent concretizations due
  //! to cloning before concretization.
  bool operator==(const DynamicTransformConcretizationInfo& other) const;

  bool operator!=(const DynamicTransformConcretizationInfo& other) const {
    return !(*this == other);
  }

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine the decomposition of each dynamic reshape operation to use
  //! during concretization.
  void analyzeReshapes(ExpressionEvaluator* expr_eval);

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine the concrete IterType of each resized IterDomain.
  void analyzeResizes(ExpressionEvaluator* expr_eval);

  const DynamicTransformInitialInfo* initialInfo() const {
    return initial_info_;
  }

  void setInitialInfo(const DynamicTransformInitialInfo* initial_info) {
    initial_info_ = initial_info;
  }

  Fusion* fusion() const {
    return initial_info_->fusion();
  }

  std::string toString() const;

  size_t hash() const;

 private:
  DynamicTransformConcretizationInfo(
      const DynamicTransformInitialInfo* initial_info)
      : initial_info_(initial_info) {}

 private:
  const DynamicTransformInitialInfo* initial_info_ = nullptr;

  //! Holds the index of the output TensorView in the vector returned by
  //! initial_info_->getDynamicReshapedTensorViews(), and the corresponding
  //! result of analyzeView
  std::vector<std::pair<size_t, AnalyzeViewResult>> reshape_transforms_;

  //! Holds a vector of indices into initial_info_.getMaybeZeroExtents() which
  //! evaluate to 0
  std::vector<size_t> empty_extents_;

  //! Holds the index of the resized IterDomain (output of the Resize op) in the
  //! vector returned by initial_info_->getDynamicResizedIterDomains() along
  //! with its concretized IterType
  std::vector<std::pair<size_t, IterType>> resize_itertypes_;

  friend class DynamicTransformInfoBuilder;
};

class TORCH_CUDA_CU_API DynamicTransform {
 public:
  //! Get initial information before we have inputs. This analyzes the Fusion to
  //! determine whether it has dynamic operations, and caches their position for
  //! faster concretization once inputs are available.
  static DynamicTransformInitialInfo getInitialInfo(Fusion* fusion);

  //! Concretizes a given fusion. Note that the concretization is
  //! in-place and the given fusion is modified.
  static void concretizeFusion(
      Fusion* fusion,
      const DynamicTransformConcretizationInfo* info);
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

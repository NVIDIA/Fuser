// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>
#include <transform_view.h>
#include <utils.h>

#include <memory>
#include <vector>

namespace nvfuser {

class Fusion;
class DynamicTransformInitialInfoBuilder;

//! Initial information derived only from the symbolic Fusion without input
//! sizes
class DynamicTransformInitialInfo {
 public:
  Fusion* fusion() const {
    return fusion_;
  }

  //! Return whether any dynamic transforms exist in the Fusion, or whether
  //! there are any tensors which could potentially be empty (size-0 extent)
  //! given some user input. In either of these cases, concretization may change
  //! the structure of the Fusion.
  bool isDynamic() const {
    return hasPossibleEmptyTensor() || !dynamic_reshaped_tvs_.empty() ||
        !dynamic_resized_ids_.empty() || !dynamic_topk_tvs_.empty();
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

  //! Return a vector of outputs of ReshapeOp expressions that have dynamic
  //! output shapes
  const std::vector<TensorView*>& getDynamicReshapedTensorViews() const {
    return dynamic_reshaped_tvs_;
  }

  //! Return a vector of outputs of Resize expressions that have symbolic output
  //! IterTypes
  const std::vector<IterDomain*>& getDynamicResizedIterDomains() const {
    return dynamic_resized_ids_;
  }

  //! Return a vector of outputs of ExpandOp expressions that have Symbolic
  //! output IterTypes
  const std::vector<TensorView*>& getDynamicExpandedTensorViews() const {
    return dynamic_expanded_tvs_;
  }

  //! Return a vector of outputs of factory expressions like full, iota,
  //! normal, and uniform that have Symbolic output IterTypes
  const std::vector<TensorView*>& getDynamicFactoryOutputs() const {
    return dynamic_factory_tvs_;
  }

  //! Return a vector of outputs of TopKOp expressions that have Symbolic
  //! output IterTypes
  const std::vector<TensorView*>& getDynamicTopKTensorViews() const {
    return dynamic_topk_tvs_;
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
  // removed by ir_utils::replaceValInExprInputs. The outputs will not: their
  // definitions will merely be altered. When the ops are replaced, if we had
  // referred to them directly here, we would run into segfaults. Referring only
  // to the outputs avoids this issue.
  std::vector<TensorView*> dynamic_reshaped_tvs_;

  std::vector<IterDomain*> dynamic_resized_ids_;

  std::vector<TensorView*> dynamic_expanded_tvs_;

  std::vector<TensorView*> dynamic_factory_tvs_;

  std::vector<TensorView*> dynamic_topk_tvs_;

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
class DynamicTransformConcretizationInfo {
 public:
  NVF_API DynamicTransformConcretizationInfo(
      const DynamicTransformInitialInfo* initial_info,
      ExpressionEvaluator* expr_eval,
      ExactLogicalDomainMap* exact_map = nullptr);

  //! Return a vector of integers each corresponding to the position in
  //! initialInfo()->getMaybeZeroExtents() of an extent Val which is guaranteed
  //! to be zero.
  const std::vector<int64_t>& getEmptyExtents() const {
    return empty_extents_;
  }

  //! Return a vector of pairs holding the index of each reshaped TensorView in
  //! the vector returned by initialInfo()->getDynamicReshapedTensorViews(),
  //! along with an AnalyzeViewResult describing how that reshape operation
  //! should be decomposed into split, merge, squeeze, and broadcast transforms.
  //!
  //! In case there are any zeros in the size of the input and output we will
  //! not perform a reshape but rather replace the output with full(). Then
  //! instead of an AnalyzeViewResult we will hold a vector of symbolic sizes
  //! indicating how to concretize the output IterDomains.
  //!
  //! The symbolic sizes are the actual sizes 0 or 1, or -1 if the size of a
  //! given reshaped dimension is greater than 1.
  using ViewConcretizationInfo =
      std::variant<AnalyzeViewResult, std::vector<int64_t>>;
  const std::vector<std::pair<int64_t, ViewConcretizationInfo>>&
  getReshapeTransforms() const {
    return reshape_transforms_;
  }

  //! Return a vector of pairs holding the index of each resized IterDomain in
  //! the vector returned by initialInfo()->getDynamicResizedIterDomains(),
  //! along with the IterType it should be concretized to.
  const std::vector<std::pair<int64_t, IterType>>& getResizeIterTypes() const {
    return resize_itertypes_;
  }

  //! Return a vector of pairs holding the index of each expanded TensorView in
  //! the vector returned by initialInfo()->getDynamicExpandedTensorViews(),
  //! along with a vector of bools describing whether each axis in the output
  //! root domain is expanded.
  const std::vector<std::pair<int64_t, std::vector<bool>>>& getExpandAxes()
      const {
    return expand_axes_;
  }

  //! Return a vector of vectors of pairs. Each vector of pairs corresponds to a
  //! TensorView returned by by initialInfo()->getDynamicFactoryOutputs(). The
  //! pairs contain an integer position of a Symbolic axis and the IterType that
  //! axis will be converted to.
  const std::vector<std::vector<std::pair<int64_t, IterType>>>&
  getFactoryOutputIterTypes() const {
    return factory_output_itertypes_;
  }

  //! Return a vector of pairs holding the index of each TopK TensorView in
  //! the vector returned by initialInfo()->getDynamicTopKTensorViews(),
  //! along with the IterType the TopK dimension should be concretized to.
  const std::vector<std::pair<int64_t, IterType>>& getTopKIterTypes() const {
    return topk_itertypes_;
  }

  //! Comparison operator for the purposes of determining cache hits. This does
  //! not guarantee equality of all members. Instead, it returns equal if the
  //! resulting concretizations would be structurally equivalent. Note that
  //! pointers to Statements may differ between equivalent concretizations due
  //! to cloning before concretization.
  NVF_API bool operator==(
      const DynamicTransformConcretizationInfo& other) const;

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

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine which axes of dynamic expand operations are expanded.
  void analyzeExpands(ExpressionEvaluator* expr_eval);

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine the IterTypes of factory function outputs.
  void analyzeFactoryOutputs(ExpressionEvaluator* expr_eval);

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine the concrete IterType of each TopK operation.
  void analyzeTopK(ExpressionEvaluator* expr_eval);

  const DynamicTransformInitialInfo* initialInfo() const {
    return initial_info_;
  }

  void setInitialInfo(const DynamicTransformInitialInfo* initial_info) {
    initial_info_ = initial_info;
  }

  Fusion* fusion() const {
    return initial_info_->fusion();
  }

  NVF_API std::string toString() const;

  NVF_API size_t hash() const;

 private:
  DynamicTransformConcretizationInfo(
      const DynamicTransformInitialInfo* initial_info)
      : initial_info_(initial_info) {}

 private:
  const DynamicTransformInitialInfo* initial_info_ = nullptr;

  //! Holds the index of the output TensorView in the vector returned by
  //! initial_info_->getDynamicReshapedTensorViews(), and the corresponding
  //! result of analyzeView (or list of IterTypes for output of full() in the
  //! case of empty reshapes).
  std::vector<std::pair<int64_t, ViewConcretizationInfo>> reshape_transforms_;

  //! Holds a vector of indices into initial_info_.getMaybeZeroExtents() which
  //! evaluate to 0
  std::vector<int64_t> empty_extents_;

  //! Holds the index of the resized IterDomain (output of the Resize op) in the
  //! vector returned by initial_info_->getDynamicResizedIterDomains() along
  //! with its concretized IterType
  std::vector<std::pair<int64_t, IterType>> resize_itertypes_;

  //! Holds the index of the expanded TensorView in the vector returned by
  //! initial_info_->getDynamicExpandedTensorViews(), and a corresponding vector
  //! of bools indicating whether each axis is in fact expanded.
  std::vector<std::pair<int64_t, std::vector<bool>>> expand_axes_;

  //! Holds the axis and IterType corresponding to each TensorView returned by
  //! initial_info_->getDynamicFactoryOutputs().
  std::vector<std::vector<std::pair<int64_t, IterType>>>
      factory_output_itertypes_;

  //! Holds the index of the TopK TensorView (values output) in the vector
  //! returned by initial_info_->getDynamicTopKTensorViews() along with its
  //! concretized IterType for the TopK dimension
  std::vector<std::pair<int64_t, IterType>> topk_itertypes_;

  friend class DynamicTransformInfoBuilder;
};

class DynamicTransform {
 public:
  //! Get initial information before we have inputs. This analyzes the Fusion to
  //! determine whether it has dynamic operations, and caches their position for
  //! faster concretization once inputs are available.
  NVF_API static DynamicTransformInitialInfo getInitialInfo(Fusion* fusion);

  //! Concretizes a given fusion. Note that the concretization is
  //! in-place and the given fusion is modified. Return a map from old, symbolic
  //! values to new, concrete values.
  NVF_API static std::unordered_map<Val*, Val*> concretizeFusion(
      Fusion* fusion,
      const DynamicTransformConcretizationInfo* info);

  //! Calls the above after computing concretization info from
  //! KernelArgumentHolder
  static std::unordered_map<Val*, Val*> concretizeFusion(
      Fusion* fusion,
      const KernelArgumentHolder& args);
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

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
    return !dynamic_expr_outputs_.empty();
  }

  //! Return a set of scalars that are inputs or extents of input TensorViews
  //! and that appear in inputs to dynamic expressions. Any Vals not in this
  //! list do not affect concretization.
  const std::unordered_set<Val*> getRootDynamicVals() const {
    return root_dynamic_vals_;
  }

  const std::vector<Val*>& getDynamicExprOutputs() const {
    return dynamic_expr_outputs_;
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
  // std::vector<TensorView*> dynamic_reshaped_tvs_;

  // std::vector<IterDomain*> dynamic_resized_ids_;

  // Slice operations can have complicated output extents. The inputs to slice
  // are a start, stop, and step for each sliced dimension. Each of these is an
  // integer, and any combination of three finite integers with step != 0 is
  // acceptable and should run without error. Normalization of the start and
  // stop values must be done, followed by computation of the output extent:
  //
  //   normed_start = min(max(where(start < 0, extent + start, start), 0),
  //   extent); normed_stop = max(min(max(where(stop < 0, extent + stop, stop),
  //   0), extent), normed_start); extent = max((normed_stop - normed_start + 1)
  //   / step, 0);
  //
  // These expressions are unwieldy and cannot be significantly simplified
  // unless we know certain relations about the start, stop, and step scalars.
  // Here we keep track of non-static slices or slices with non-static input
  // extents. That way we can restrict to a single branch in each of these
  // expressions during concretization.
  // std::vector<TensorView*> dynamic_sliced_tvs_;

  // This is a topologically sorted list of outputs of dynamic operations.
  std::vector<Val*> dynamic_expr_outputs_;

  // Root Vals that determine concretization
  std::unordered_set<Val*> root_dynamic_vals_;

  friend class DynamicTransformInitialInfoBuilder;
};

//! This enum describes cases that can occur for the start or stop arguments to
//! slice(). Each of these leads to a different branch in the normalized form's
//! general expression.
enum class SliceIndexBranch {
  Negative, // -extent < a < 0
  Zero, // a == 0  OR  a <= -extent
  Positive, // 0 < a < extent
  Extent // extent <= a
};

//! This enum describes the "step" argument to slice, which can be a positive or
//! negative integer (but not zero). We handle the special case of step == 1
//! separately from step > 1 since this simplifies some expressions.
enum class SliceStepBranch { Negative, One, GreaterThanOne };

//! Describes a 1D slice in terms of the start, stop, and extent values
struct Concrete1DSliceDescriptor {
  //! These enums determine the form of the simplified expressions
  SliceIndexBranch start_branch = SliceIndexBranch::Zero;
  SliceIndexBranch stop_branch = SliceIndexBranch::Extent;
  SliceStepBranch step_branch = SliceStepBranch::One;

  //! True if normalized values satisfy (stop - start) * step <= 0 in which case
  //! we would return an empty tensor.
  bool is_empty = false;

  //! This can be either Iteration or Broadcast (if sliced extent is 1)
  IterType iter_type = IterType::Iteration;

  bool operator==(const Concrete1DSliceDescriptor& other) const {
    return start_branch == other.start_branch &&
        stop_branch == other.stop_branch && step_branch == other.step_branch &&
        is_empty == other.is_empty && iter_type == other.iter_type;
  }
  bool operator!=(const Concrete1DSliceDescriptor& other) const {
    return !operator==(other);
  }

  size_t hash() const {
    size_t h = (size_t)start_branch;
    hashCombine(h, (size_t)stop_branch);
    hashCombine(h, (size_t)step_branch);
    hashCombine(h, (size_t)is_empty);
    hashCombine(h, (size_t)iter_type);
    return h;
  }
};

//! A set of transformations for a symbolic fusion with concrete sizes
//! of the fusion inputs
class TORCH_CUDA_CU_API DynamicTransformConcretizationInfo {
 public:
  DynamicTransformConcretizationInfo(
      const DynamicTransformInitialInfo* initial_info,
      ExpressionEvaluator* expr_eval)
      : initial_info_(initial_info) {
    TORCH_INTERNAL_ASSERT(
        !fusion()->isA<kir::Kernel>(),
        "Invalid container. Kernel container not allowed.\n");

    // Make sure all exactly mapped IDs have the same value in the
    // evaluator when any one of the IDs has a known value
    expr_eval->propagateBoundValuesThroughExactMaps(initial_info->fusion());

    analyze(expr_eval);
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

  //! Return a vector of pairs holding the index of each sliced TensorView in
  //! the vector returned by initialInfo()->getDynamicSlicedTensorViews(),
  //! along with a vector of descriptors indicating how each axis should be
  //! concretized.
  const std::vector<std::pair<size_t, std::vector<Concrete1DSliceDescriptor>>>&
  getSliceDescriptors() const {
    return slice_descriptors_;
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
  //! analyze all dynamic ops in topological order.
  void analyze(ExpressionEvaluator* expr_eval);

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine the decomposition of a dynamic reshape operation to use
  //! during concretization.
  void analyzeReshape(ExpressionEvaluator* expr_eval, size_t val_index);

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine the branches of expressions in a dynamic slice op.
  void analyzeSlice(ExpressionEvaluator* expr_eval, size_t val_index);

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine the concrete IterType of a resized IterDomain.
  void analyzeResize(ExpressionEvaluator* expr_eval, size_t val_index);

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

  //! Holds the index of the resized IterDomain (output of the Resize op) in the
  //! vector returned by initial_info_->getDynamicResizedIterDomains() along
  //! with its concretized IterType
  std::vector<std::pair<size_t, IterType>> resize_itertypes_;

  //! Holds the index of the sliced TensorView (output of the SliceOp) in the
  //! vector returned by initial_info_->getDynamicSlicedTensorViews() along
  //! with a descriptor of how it should be concretized.
  std::vector<std::pair<size_t, std::vector<Concrete1DSliceDescriptor>>>
      slice_descriptors_;
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
      Fusion*,
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

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <iter_visitor.h>
#include <transform_view.h>
#include <utils.h>

#include <functional>
#include <memory>
#include <variant>
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
    return hasPossibleEmptyTensor() || !dynamic_exprs_.empty();
  }

  //! Return whether there are any tensors with unknown extent in some
  //! dimension, so that they might be empty
  bool hasPossibleEmptyTensor() const {
    return !maybe_zero_extents_.empty();
  }

  const std::vector<Expr*>& getDynamicExprs() const {
    return dynamic_exprs_;
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

  std::string toString() const;

  DynamicTransformInitialInfo clone(IrCloner& ir_cloner) const;

  //! Return a set containing positions in inputs() holding any scalar input
  //! that would affect the structure of the concretized Fusion.
  const std::unordered_set<size_t>& scalarInputsAffectingConcretization()
      const {
    return scalar_inputs_affecting_concretization_;
  }

 protected:
  //! Holds the set of scalar fusion input positions that affect concretization.
  std::unordered_set<size_t> scalar_inputs_affecting_concretization_;

 private:
  DynamicTransformInitialInfo(Fusion* fusion) : fusion_(fusion) {}

 private:
  Fusion* fusion_ = nullptr;

  // This is a vector of dynamic Exprs, in topological order.
  std::vector<Expr*> dynamic_exprs_;

  // This is a minimal set of scalars to check for empty tensors. If any are
  // zero, we should traverse to find empty tensors.
  std::unordered_set<Val*> maybe_zero_extents_set_;
  // The set above is populated then used to create this unique vector which is
  // ordered arbitrarily.
  std::vector<Val*> maybe_zero_extents_;

  // Minimal set of Vals that determine all aspects of concretization
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

  //! Return a vector of descriptors describing how to concretize the dynamic
  //! expressions in the initial info.
  const auto& getExprConcretizationDescriptors() const {
    return concretization_descriptors_;
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
  //! determine the decomposition of a dynamic reshape operation to use during
  //! concretization.
  void analyze(ViewOp* vop, ExpressionEvaluator* expr_eval);

  //! Given an ExpressionEvaluator which already has input scalars bound to it,
  //! determine the concrete IterType of each resized IterDomain.
  void analyze(Resize* rop, ExpressionEvaluator* expr_eval);

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

  //! Holds data required to concretize an operation. Entries in this vector
  //! correspond to the Exprs in initial_info_->getDynamicExprs().
  //!
  //! Each type of Expr requires a different type of information at
  //! concretization, so this holds a variant that should enumerate all data
  //! types encountered. Each should be copyable and should not include pointers
  //! to any Statements.
  std::vector<std::variant<
      AnalyzeViewResult, // For ViewOp
      IterType // For Resize, the IterType of the output
      >>
      concretization_descriptors_;

  //! Holds a vector of indices into initial_info_.getMaybeZeroExtents() which
  //! evaluate to 0
  std::vector<size_t> empty_extents_;

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

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

  DynamicTransformInfo(Fusion* fusion) : fusion_(fusion) {}

  Fusion* fusion() const {
    return fusion_;
  }

  const std::vector<std::pair<TensorView*, AnalyzeViewResult>>
  getReshapeTransforms() const {
    return reshape_transforms_;
  }

  // TODO: Make it private
 public:
  Fusion* fusion_ = nullptr;
  std::vector<std::pair<TensorView*, AnalyzeViewResult>> reshape_transforms_;

  // TODO: resize transforms

  friend class DynamicTransformInfoBuilder;
};

//! Concretize dynamic transforms in
void concretizeDynamicTransform(
    Fusion* fusion,
    const DynamicTransformInfo& info);

class TORCH_CUDA_CU_API DynamicTransformConcretizer : public OptOutMutator {
 public:
  static void concretizeFusion(
      Fusion* fusion,
      const DynamicTransformInfo& info);

 private:
  DynamicTransformConcretizer(Fusion* fusion, const DynamicTransformInfo& info)
      : info_(info) {
    TORCH_INTERNAL_ASSERT(
        fusion == info.fusion(),
        "Invalid DynamicTransformInfo. The associated Fusion is different from the given Fusion");
  }

  void concretize();
  void concretizeReshape();

 private:
  using OptOutMutator::mutate;

  // void mutate(IterDomain* id) final {
  // std::cerr << "ID: " << id->toString() << std::endl;
  // OptOutMutator::mutate(id);
  // }

  void mutate(TensorView* tv) final;
  void mutate(TensorDomain* td) final;

  // void mutate(Expr* expr) final;

  void handleTensorViewExpr(Expr* expr);
  void handleIterDomainExpr(Expr* expr);
  bool propagateFromProducerToConsumer(TensorView* consumer);

 private:
  const DynamicTransformInfo& info_;
  std::unordered_map<IterDomain*, IterDomain*> update_map_;
};

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

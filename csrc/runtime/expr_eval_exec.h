// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <runtime/executor_abstract.h>

namespace nvfuser {

class ExprEvalExecutor : public ExecutorAbstract {
 public:
  ExprEvalExecutor(
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0)
      : ExecutorAbstract(fusion_id, concrete_id, runtime_id, group_id) {}

  // Returns true if all fusion outputs are expression evaluated.
  static bool supported(Fusion* fusion);

  void compile(Fusion* fusion);

  bool isCompiled() const override;

  NVF_API std::vector<at::Tensor> run(
      KernelArgumentHolder& args,
      std::vector<at::Tensor> outputs = {});

  const std::unique_ptr<Fusion>& fusion() {
    return fusion_;
  }

 private:
  std::unique_ptr<Fusion> fusion_;

  // Expressions to evaluate
  std::vector<Expr*> exprs_;

  struct ViewInfo {
    // Sizes of the output of view ops, only one value can be unknown at it gets
    // processed in aten as a -1 size, every other dim is a constant positive
    // integer value.
    std::vector<Val*> output_view_sizes;
    // PyTorch's API defines all output shapes as a constant known size except
    // upto 1 which can be easily inferred based on the input numel and the rest
    // of the ouput sizes. nvFuser can have dynamic reshape operations where the
    // output sizes are inferred through split and merge operations on IDs. If
    // use_neg_1 is true then all values except up to one are constant values.
    bool use_neg_1 = false;
    // at::view can be used on contiguous tensors and is faster than
    // at::reshape. Since we know at compile time if the tensor is contiguous
    // then we can route evaluation to view.
    bool use_at_view = false;
  };

  std::unordered_map<ViewOp*, ViewInfo> view_infos;

  // Permute map, stores permutation axes if a LoadStoreOp requires them.
  std::unordered_map<LoadStoreOp*, std::vector<int64_t>> permutation_orders;

  void compile(ViewOp* view_op);
  at::Tensor run(ViewOp* view_op, ExpressionEvaluator& expr_eval);

  void compile(LoadStoreOp* ld_st_op);
  at::Tensor run(LoadStoreOp* ld_st_op, at::Tensor input);
};
} // namespace nvfuser

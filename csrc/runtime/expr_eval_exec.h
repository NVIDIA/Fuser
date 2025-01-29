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
  // TODO: Set properly
  std::unique_ptr<Fusion> fusion_;
};
} // namespace nvfuser

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

TEST_F(NVFuserTest, FusionHash_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeContigTensor(1);
  fusion_ptr->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion_ptr->addOutput(tv1);

  std::cout << "Fusion hash: " << fusion_ptr->hash() << std::endl;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100}, options);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  auto ref = t0 + 1;
  testValidate(fusion_ptr.get(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

} // namespace nvfuser

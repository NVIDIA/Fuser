// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on

// This file contains integration tests that run fusions through
// FusionExecutorCache with host IR lowering turned on.
#include <algorithm>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include "fusion.h"
#include "host_ir/ir.h"
#include "ir/all_nodes.h"
#include "ops/all_ops.h"
#include "options.h"
#include "runtime/fusion_kernel_runtime.h"
#include "tests/cpp/utils.h"
#include "tests/cpp/validator.h"

namespace nvfuser {

class HostIrPassesTest : public NVFuserTest {
 protected:
  HostIrPassesTest() {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
};

TEST_F(HostIrPassesTest, TwoMatmulsInlinable) {
  constexpr int64_t c = 3;

  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::InferContiguity);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* inp = makeContigTensor(2);
  TensorView* w1 = makeContigTensor(2);
  TensorView* w2 = makeContigTensor(2);
  TensorView* intermediate = matmul(inp, w1);
  TensorView* out = matmul(intermediate, w2);

  fusion->addInput(inp);
  fusion->addInput(w1);
  fusion->addInput(w2);
  fusion->addOutput(out);

  inp->outer_split(0, c);
  inp->axis(0)->parallelize(ParallelType::Stream);

  FusionExecutorCache executor(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp_tensor = at::randn({c * 2, 3}, options);
  at::Tensor w1_tensor = at::randn({3, 5}, options);
  at::Tensor w2_tensor = at::randn({5, 3}, options);

  auto out_tensors =
      executor.runFusionWithInputs({inp_tensor, w1_tensor, w2_tensor});

  // Both matmuls are inlined in the same loop; `intermediate` is
  // allocated and deallocated within the loop body.
  FusionKernelRuntime* runtime = executor.getMostRecentKernelRuntime();
  const auto& exprs = runtime->getHostIrContainer().topLevelExprs();
  auto it = std::find_if(exprs.begin(), exprs.end(), [](Expr* e) {
    return e->isA<hir::ForLoop>();
  });
  ASSERT_NE(it, exprs.end());
  const auto& body = (*it)->as<hir::ForLoop>()->body();
  int deallocate_count =
      std::count_if(body.exprs().begin(), body.exprs().end(), [](Expr* e) {
        return e->isA<hir::Deallocate>();
      });
  EXPECT_EQ(deallocate_count, 1)
      << "Expected for-loop body to have exactly one Deallocate, got "
      << deallocate_count;

  testValidate(
      executor.fusion(),
      out_tensors,
      {inp_tensor, w1_tensor, w2_tensor},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(HostIrPassesTest, TwoMatmulsNotInlinable) {
  constexpr int64_t c = 3;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* inp = makeContigTensor(2);
  TensorView* w1 = makeContigTensor(2);
  TensorView* w2 = makeContigTensor(2);
  TensorView* out1 = matmul(inp, w1);
  TensorView* out = matmul(out1, w2);

  fusion->addInput(inp);
  fusion->addInput(w1);
  fusion->addInput(w2);
  fusion->addOutput(out);

  w1->split(1, c, /*inner_split=*/false);
  w1->axis(1)->parallelize(ParallelType::Stream);
  out->split(0, c, /*inner_split=*/false);
  out->axis(0)->parallelize(ParallelType::Stream);

  FusionExecutorCache executor(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp_tensor = at::randn({c * 2, 3}, options);
  at::Tensor w1_tensor = at::randn({3, c * 5}, options);
  at::Tensor w2_tensor = at::randn({c * 5, 3}, options);

  auto out_tensors =
      executor.runFusionWithInputs({inp_tensor, w1_tensor, w2_tensor});

  // The intermediate (out1) is fully allocated; its deallocate is at top level.
  FusionKernelRuntime* runtime = executor.getMostRecentKernelRuntime();
  const auto& exprs = runtime->getHostIrContainer().topLevelExprs();
  int deallocate_count = std::count_if(exprs.begin(), exprs.end(), [](Expr* e) {
    return e->isA<hir::Deallocate>();
  });
  EXPECT_EQ(deallocate_count, 1)
      << "Expected exactly one Deallocate at top level, got "
      << deallocate_count;

  testValidate(
      executor.fusion(),
      out_tensors,
      {inp_tensor, w1_tensor, w2_tensor},
      __LINE__,
      __FILE__,
      "");
}

} // namespace nvfuser

// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on

#include <ranges>
#include <unordered_set>

#include <gtest/gtest.h>

#include "fusion.h"
#include "host_ir/ir.h"
#include "ir/all_nodes.h"
#include "ops/composite.h"
#include "options.h"
#include "runtime/fusion_kernel_runtime.h"
#include "tests/cpp/utils.h"
#include "tests/cpp/validator.h"

namespace nvfuser {

namespace {

// Traverse the IR and collect all allocated Tensorviews and remove them when
// a Deallocate is encountered.
void collectPersistentTensorViews(
    const Scope& scope,
    std::unordered_set<TensorView*>& allocated) {
  for (Expr* e : scope.exprs()) {
    if (auto* dealloc = dynamic_cast<hir::Deallocate*>(e)) {
      allocated.erase(dealloc->buffer());
      continue;
    }
    if (auto* alloc = dynamic_cast<kir::Allocate*>(e)) {
      allocated.insert(alloc->buffer()->as<TensorView>());
      continue;
    }
    for (auto* tv : ir_utils::filterByType<TensorView>(e->inputs())) {
      allocated.insert(tv);
    }
    for (auto* tv : ir_utils::filterByType<TensorView>(e->outputs())) {
      allocated.insert(tv);
    }
    if (auto* loop = dynamic_cast<hir::ForLoop*>(e)) {
      collectPersistentTensorViews(loop->body(), allocated);
    }
  }
}

void checkMemoryLeak(const hir::HostIrContainer& hic) {
  std::unordered_set<TensorView*> allocated;
  collectPersistentTensorViews(hic.topLevel(), allocated);
  EXPECT_TRUE(std::all_of(
      allocated.begin(),
      allocated.end(),
      [](TensorView* tv) {
        return tv->isFusionInput() || tv->isFusionOutput();
      }))
      << "Some TensorViews allocated in IR are not deallocated and not fusion "
         "inputs/outputs.";
}

} // namespace

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
  TensorView* out = matmul(inp, w1);
  out = matmul(out, w2);

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

  FusionKernelRuntime* runtime = executor.getMostRecentKernelRuntime();
  checkMemoryLeak(runtime->getHostIrContainer());

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
  TensorView* out = matmul(inp, w1);
  out = matmul(out, w2);

  fusion->addInput(inp);
  fusion->addInput(w1);
  fusion->addInput(w2);
  fusion->addOutput(out);

  w1->outer_split(1, c);
  w1->axis(1)->parallelize(ParallelType::Stream);
  out->outer_split(0, c);
  out->axis(0)->parallelize(ParallelType::Stream);

  FusionExecutorCache executor(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp_tensor = at::randn({c * 2, 3}, options);
  at::Tensor w1_tensor = at::randn({3, c * 5}, options);
  at::Tensor w2_tensor = at::randn({c * 5, 3}, options);

  auto out_tensors =
      executor.runFusionWithInputs({inp_tensor, w1_tensor, w2_tensor});

  FusionKernelRuntime* runtime = executor.getMostRecentKernelRuntime();
  checkMemoryLeak(runtime->getHostIrContainer());

  testValidate(
      executor.fusion(),
      out_tensors,
      {inp_tensor, w1_tensor, w2_tensor},
      __LINE__,
      __FILE__,
      "");
}

} // namespace nvfuser

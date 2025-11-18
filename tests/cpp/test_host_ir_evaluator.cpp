// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on

// As test_host_irs.cpp exceeded 1000 lines, this file now contains unit tests
// specifically for host IRs utilized solely within FusionExecutorCache.
#include <gtest/gtest.h>

#include <ATen/ops/matmul.h>
#include <ATen/ops/randn.h>

#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/evaluator.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/composite.h>
#include <tests/cpp/utils.h>

namespace nvfuser::hir {

using HostIrEvaluatorTest = NVFuserTest;

// This test manually creates a HostIrContainer with LaunchKernels and runs it
// using HostIrEvaluator.
TEST_F(HostIrEvaluatorTest, LaunchKernel) {
  Fusion fusion;
  {
    FusionGuard fg(&fusion);
    TensorView* in = makeSymbolicTensor(2);
    TensorView* out = set(in);
    fusion.addInput(in);
    fusion.addOutput(out);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({32, 32}, options);

  auto hic = std::make_unique<HostIrContainer>();
  {
    FusionGuard fg(hic.get());

    auto ke = std::make_unique<KernelExecutor>();
    ke->setGroupId(0);
    ke->compile(&fusion, {in_tensor});
    hic->addKernelExecutor(std::move(ke));

    IrCloner ir_cloner(hic.get());
    Val* in = ir_cloner.clone(fusion.inputs().at(0));
    Val* out = ir_cloner.clone(fusion.outputs().at(0));

    auto allocate = IrBuilder::create<kir::Allocate>(out, MemoryType::Global);
    auto* cache_id =
        IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
    auto launch_kernel = IrBuilder::create<LaunchKernel>(
        0,
        LaunchParams(),
        CompileParams(),
        std::vector<Val*>{in},
        std::vector<Val*>{out},
        cache_id);

    hic->addInput(in);
    hic->addOutput(out);

    hic->pushBackTopLevelExprs(allocate);
    hic->pushBackTopLevelExprs(launch_kernel);
  }

  HostIrEvaluator hie(
      std::move(hic),
      /*communicator=*/nullptr,
      HostIrEvaluatorParams{
          .mode = HostIrEvaluatorParams::Mode::kFusionExecutorCache});
  KernelArgumentHolder ins(in_tensor);
  ins.setCacheId(0);
  KernelArgumentHolder outs = hie.runWithInputs(ins);

  auto out_tensor = outs[0].as<at::Tensor>();
  EXPECT_TRUE(out_tensor.equal(in_tensor));
}

namespace {
// Ideally, recomputation should be done automatically in TensorView's cloner.
// But I'm hitting #4849 when trying that.
void recomputeTv(const TensorView* tv, IrCloner& ir_cloner) {
  for (Expr* e : StmtSort::getExprsTo(
           {tv->getLoopDomain().begin(), tv->getLoopDomain().end()})) {
    ir_cloner.clone(e);
  }
  for (IterDomain* id : tv->getLoopDomain()) {
    for (Expr* e : StmtSort::getExprsTo({id->extent()})) {
      ir_cloner.clone(e);
    }
  }
}
} // namespace

TEST_F(HostIrEvaluatorTest, MatmulInLoop) {
  constexpr int64_t c = 3;

  Fusion fusion;
  {
    FusionGuard fg(&fusion);
    TensorView* in = makeSymbolicTensor(2);
    TensorView* w = makeSymbolicTensor(2);
    TensorView* out = matmul(in, w);
    fusion.addInput(in);
    fusion.addInput(w);
    fusion.addOutput(out);

    w->outer_split(1, c);
    w->axis(1)->parallelize(ParallelType::Stream);
    out->outer_split(1, c);
    out->axis(1)->parallelize(ParallelType::Stream);
  }

  // We don't have host IR lowering in place for the stream parallel type yet.
  // Below is a hand-written host IR implementation for the above fusion with
  // some tweaks for test coverage.
  auto hic = std::make_unique<HostIrContainer>();
  {
    auto* original_in = fusion.inputs().at(0)->as<TensorView>();
    auto* original_w = fusion.inputs().at(1)->as<TensorView>();
    auto* original_out = fusion.outputs().at(0)->as<TensorView>();

    FusionGuard fg(hic.get());
    IrCloner ir_cloner(hic.get());
    auto* in = ir_cloner.clone(original_in);
    auto* w = ir_cloner.clone(original_w);
    recomputeTv(original_w, ir_cloner);
    auto* out = ir_cloner.clone(original_out);
    recomputeTv(original_out, ir_cloner);
    hic->addInput(in);
    hic->addInput(w);
    hic->addOutput(out);

    auto* allocate_out = IrBuilder::create<kir::Allocate>(
        out, MemoryType::Global, std::vector<Val*>({}), /*zero_init=*/true);

    auto* stream_index = IrBuilder::create<Val>(DataType::Index);
    // `start` is set to one intentially because I wanted to harden the test for
    // the for loop. Imagine a buggy nvFuser that completely ignores allocation
    // domain being stream parallelized. It would make each loop iteration
    // overwrite the entire tensor instead of a slice. This bug wouldn't be
    // captured by the test if the for loop started from 0.
    auto* for_loop = IrBuilder::create<ForLoop>(
        stream_index,
        /*start=*/hic->oneVal(DataType::Index),
        /*stop=*/IrBuilder::create<Val>(c, DataType::Index));

    TensorView* loop_w = shardByStream(w, stream_index);
    for_loop->body().push_back(loop_w->definition());

    TensorView* loop_out = shardByStream(out, stream_index);
    for_loop->body().push_back(loop_out->definition());

    // By default, MatmulOp is computed by ExpressionEvaluator so it appears in
    // host IR.
    auto* mm = IrBuilder::create<MatmulOp>(loop_out, in, loop_w);
    for_loop->body().push_back(mm);

    hic->pushBackTopLevelExprs(allocate_out);
    hic->pushBackTopLevelExprs(for_loop);
  }

  at::Tensor in_tensor =
      at::randn({5, 7}, at::dtype(at::kFloat).device(at::kCUDA));
  at::Tensor w_tensor =
      at::randn({7, c * 2}, at::dtype(at::kFloat).device(at::kCUDA));
  at::Tensor expected_out_tensor = at::matmul(in_tensor, w_tensor);
  expected_out_tensor.chunk(c, 1)[0].zero_();

  HostIrEvaluator hie(
      std::move(hic),
      /*communicator=*/nullptr,
      HostIrEvaluatorParams{
          .mode = HostIrEvaluatorParams::Mode::kFusionExecutorCache});
  KernelArgumentHolder ins({in_tensor, w_tensor});
  ins.setCacheId(0);
  KernelArgumentHolder outs = hie.runWithInputs(ins);
  auto out_tensor = outs[0].as<at::Tensor>();

  EXPECT_TRUE(at::allclose(out_tensor, expected_out_tensor))
      << out_tensor << " vs " << expected_out_tensor;
}

TEST_F(HostIrEvaluatorTest, InplaceUpdateInLoop) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in_tensor = at::zeros({10}, options);

  auto ke = std::make_unique<KernelExecutor>();
  {
    Fusion fusion;
    {
      FusionGuard fg(&fusion);
      TensorView* x = makeSymbolicTensor(1);
      auto* y = IrBuilder::create<Val>(DataType::Int);
      TensorView* z = add(x, y);
      fusion.addInput(x);
      fusion.addInput(y);
      fusion.addOutput(z);
    }

    ke->setGroupId(0);
    // The shapes are used for compilation; the content doesn't matter.
    ke->compile(&fusion, {in_tensor, 1});
  }

  // x = torch.zeros(10)
  // for loop_index in range(3):
  //   y = loop_index * 2
  //   y = y + 1
  //   x.add_(y)  # y = 1, then 3, then 5
  // torch.testing.assert_close(x, torch.full((10,), 9.0))
  auto hic = std::make_unique<HostIrContainer>();
  {
    hic->addKernelExecutor(std::move(ke));

    FusionGuard fg(hic.get());
    TensorView* x = makeSymbolicTensor(1);
    auto* cache_id =
        IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
    auto* loop_index = IrBuilder::create<Val>(DataType::Int);
    auto* for_loop = IrBuilder::create<ForLoop>(
        loop_index,
        /*start=*/hic->zeroVal(DataType::Int),
        /*stop=*/IrBuilder::create<Val>(3, DataType::Int));
    {
      auto* y = mul(loop_index, IrBuilder::create<Val>(2, DataType::Int));
      for_loop->body().push_back(y->definition());
      y = add(y, IrBuilder::create<Val>(1, DataType::Int));
      for_loop->body().push_back(y->definition());
      auto* launch_kernel = IrBuilder::create<LaunchKernel>(
          0,
          LaunchParams(),
          CompileParams(),
          std::vector<Val*>{x, y},
          std::vector<Val*>{x},
          cache_id);
      for_loop->body().push_back(launch_kernel);
    }

    hic->addInput(x);
    hic->addOutput(x);
    hic->pushBackTopLevelExprs(for_loop);
  }

  HostIrEvaluator hie(
      std::move(hic),
      /*communicator=*/nullptr,
      HostIrEvaluatorParams{
          .mode = HostIrEvaluatorParams::Mode::kFusionExecutorCache});
  KernelArgumentHolder ins(in_tensor);
  ins.setCacheId(0);
  KernelArgumentHolder outs = hie.runWithInputs(ins);
  auto out_tensor = outs[0].as<at::Tensor>();

  at::Tensor expected_out_tensor = at::full_like(in_tensor, 9.0);
  EXPECT_TRUE(at::allclose(out_tensor, expected_out_tensor))
      << out_tensor << " vs " << expected_out_tensor;
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());
}

TEST_F(HostIrEvaluatorTest, AddInLoop) {
  constexpr int64_t c = 3;
  Fusion fusion;
  {
    FusionGuard fg(&fusion);
    TensorView* in = makeContigTensor(2);
    TensorView* out = add(in, in);
    fusion.addInput(in);
    fusion.addOutput(out);
    in->outer_split(1, c);
    in->axis(1)->parallelize(ParallelType::Stream);
    out->outer_split(1, c);
    out->axis(1)->parallelize(ParallelType::Stream);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({5, c * 2}, options);

  auto hic = std::make_unique<HostIrContainer>();
  {
    FusionGuard fg(hic.get());
    IrCloner ir_cloner(hic.get());
    auto* in = ir_cloner.clone(fusion.inputs().at(0))->as<TensorView>();
    auto* out = ir_cloner.clone(fusion.outputs().at(0))->as<TensorView>();
    hic->addInput(in);
    hic->addOutput(out);

    auto* allocate_out = IrBuilder::create<kir::Allocate>(
        out, MemoryType::Global, std::vector<Val*>({}), /*zero_init=*/true);

    auto* stream_index = IrBuilder::create<Val>(DataType::Index);
    auto* for_loop = IrBuilder::create<ForLoop>(
        stream_index,
        /*start=*/hic->oneVal(DataType::Index),
        /*stop=*/IrBuilder::create<Val>(c, DataType::Index));

    auto ke = std::make_unique<KernelExecutor>();
    ke->compile(&fusion, {in_tensor});
    hic->addKernelExecutor(std::move(ke));

    auto* cache_id =
        IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
    auto* launch_kernel = IrBuilder::create<LaunchKernel>(
        0,
        LaunchParams(),
        CompileParams(),
        std::vector<Val*>{in, stream_index},
        std::vector<Val*>{out},
        cache_id);
    for_loop->body().push_back(launch_kernel);

    hic->pushBackTopLevelExprs(allocate_out);
    hic->pushBackTopLevelExprs(for_loop);
  }

  HostIrEvaluator hie(
      std::move(hic),
      /*communicator=*/nullptr,
      HostIrEvaluatorParams{
          .mode = HostIrEvaluatorParams::Mode::kFusionExecutorCache});
  KernelArgumentHolder ins(in_tensor);
  ins.setCacheId(0);
  KernelArgumentHolder outs = hie.runWithInputs(ins);
  auto out_tensor = outs[0].as<at::Tensor>();

  at::Tensor expected_out_tensor = in_tensor + in_tensor;
  expected_out_tensor.chunk(c, 1)[0].zero_();
  EXPECT_TRUE(at::allclose(out_tensor, expected_out_tensor))
      << out_tensor << " vs " << expected_out_tensor;
}

} // namespace nvfuser::hir

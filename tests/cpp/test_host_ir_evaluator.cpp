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

#include <ATen/ATen.h>

#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/evaluator.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ops/alias.h>
#include <ops/arith.h>
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

  HostIrEvaluator hie(std::move(hic));
  KernelArgumentHolder ins(in_tensor);
  ins.setCacheId(0);
  KernelArgumentHolder outs = hie.runWithInputs(ins);

  auto out_tensor = outs[0].as<at::Tensor>();
  EXPECT_TRUE(out_tensor.equal(in_tensor));
}

TEST_F(HostIrEvaluatorTest, AddInLoop) {
  constexpr int64_t c = 3;

  Fusion fusion;
  {
    FusionGuard fg(&fusion);
    TensorView* in = makeSymbolicTensor(2);
    TensorView* out = add(in, in);
    fusion.addInput(in);
    fusion.addOutput(out);

    in->outer_split(1, c);
    in->axis(1)->parallelize(ParallelType::Stream);
    out->outer_split(1, c);
    out->axis(1)->parallelize(ParallelType::Stream);
  }

  // We don't have host IR lowering in place for the stream parallel type yet.
  // Below is a hand-written host IR implementation for the above fusion with
  // some tweaks for test coverage.
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
    // `start` is set to one intentially because I wanted to harden the test for
    // the for loop. Imagine a buggy nvFuser that completely ignores allocation
    // domain being stream parallelized. It would make each loop iteration
    // overwrite the entire tensor instead of a slice. This bug wouldn't be
    // captured by the test if the for loop started from 0.
    auto* for_loop = IrBuilder::create<ForLoop>(
        out->axis(1),
        stream_index,
        /*start=*/hic->oneVal(DataType::Index),
        /*stop=*/IrBuilder::create<Val>(c, DataType::Index),
        /*step=*/hic->oneVal());

    TensorView* loop_in = set(in);
    loop_in->outer_split(1, c);
    loop_in->axis(1)->parallelize(ParallelType::Stream);
    loop_in->setAllocationDomain(loop_in->getLoopDomain(), {false, true, true});
    auto* shard_in =
        IrBuilder::create<ShardByStream>(loop_in, in, stream_index);
    for_loop->body().push_back(shard_in);

    TensorView* loop_out = set(out);
    loop_out->outer_split(1, c);
    loop_out->axis(1)->parallelize(ParallelType::Stream);
    loop_out->setAllocationDomain(
        loop_out->getLoopDomain(), {false, true, true});
    auto* shard_out =
        IrBuilder::create<ShardByStream>(loop_out, out, stream_index);
    for_loop->body().push_back(shard_out);

    // In reality, this should be a LaunchKernel and the two `ShardByStream`s
    // should disappear. But currently we can't pass streamIdx to a kernel yet.
    auto* add = IrBuilder::create<BinaryOp>(
        BinaryOpType::Add, loop_out, loop_in, loop_in);
    for_loop->body().push_back(add);

    hic->pushBackTopLevelExprs(allocate_out);
    hic->pushBackTopLevelExprs(for_loop);
  }

  at::Tensor in_tensor =
      at::randn({5, c * 2}, at::dtype(at::kFloat).device(at::kCUDA));
  at::Tensor expected_out_tensor = in_tensor + in_tensor;
  expected_out_tensor.chunk(c, 1)[0].zero_();

  HostIrEvaluator hie(std::move(hic));
  KernelArgumentHolder ins(in_tensor);
  ins.setCacheId(0);
  KernelArgumentHolder outs = hie.runWithInputs(ins);
  auto out_tensor = outs[0].as<at::Tensor>();

  EXPECT_TRUE(torch::allclose(out_tensor, expected_out_tensor))
      << "out_tensor: " << std::endl
      << out_tensor << std::endl
      << "expected_out_tensor: " << std::endl
      << expected_out_tensor << std::endl;
}

} // namespace nvfuser::hir

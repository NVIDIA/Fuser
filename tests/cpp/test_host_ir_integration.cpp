// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <global_allocator.h>
#include <host_ir/container.h>
#include <host_ir/evaluator.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace hir {

using testing::Contains;

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
    // FIXME: should in and out be stream parallelized?
    FusionGuard fg(&fusion);
    TensorView* in = makeSymbolicTensor(2);
    TensorView* out = add(in, in);
    fusion.addInput(in);
    fusion.addOutput(out);
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
    // the for loop.
    auto* for_loop = IrBuilder::create<ForLoop>(
        in->axis(0),
        stream_index,
        /*start=*/hic->oneVal(DataType::Index),
        /*stop=*/IrBuilder::create<Val>(c, DataType::Index),
        /*step=*/hic->oneVal());

    TensorView* loop_in = set(in);
    loop_in->outer_split(0, c);
    loop_in->axis(0)->parallelize(ParallelType::Stream);
    loop_in->setAllocationDomain(loop_in->getLoopDomain(), false);
    auto* shard_in =
        IrBuilder::create<ShardByStream>(loop_in, in, stream_index);
    for_loop->body().push_back(shard_in);

    TensorView* loop_out = set(out);
    loop_out->outer_split(0, c);
    loop_out->axis(0)->parallelize(ParallelType::Stream);
    loop_out->setAllocationDomain(loop_out->getLoopDomain(), false);
    auto* shard_out =
        IrBuilder::create<ShardByStream>(loop_out, out, stream_index);
    for_loop->body().push_back(shard_out);

    // In reality, this should be a LaunchKernel. But currently we can't pass
    // streamIdx to a kernel yet.
    auto* add = IrBuilder::create<BinaryOp>(
        BinaryOpType::Add, loop_out, loop_in, loop_in);
    for_loop->body().push_back(add);

    hic->pushBackTopLevelExprs(allocate_out);
    hic->pushBackTopLevelExprs(for_loop);
  }

  at::Tensor in_tensor =
      at::randn({c * 2, 5}, at::dtype(at::kFloat).device(at::kCUDA));
  at::Tensor expected_out_tensor = in_tensor + in_tensor;
  expected_out_tensor.chunk(c, 0)[0].zero_();

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

class HostIrIntegrationTest : public NVFuserTest {
 protected:
  HostIrIntegrationTest() {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
};

TEST_F(HostIrIntegrationTest, Set_Kernel) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = set(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(HostIrIntegrationTest, Sum_Kernel) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = sum(in, {0});
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(HostIrIntegrationTest, ExprEvalAndKernel) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  TensorView* out = permute(in, {1, 0});
  out = segment_set(out);
  out = sum(out, {0});
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

// Same as AliasTest.ViewPermute but via host IR. This test verifies that Exprs
// in SchedulerType::ExprEval segments are cloned to the top level and produce
// aliases.
TEST_F(HostIrIntegrationTest, ViewPermute_ExprEval) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  out = permute(out, {1, 0});
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();

  // Verify aliasing.
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());

  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(HostIrIntegrationTest, Deallocate) {
  const std::vector<int64_t> sizes = {8, 64};
  c10::DeviceIndex device_index = 0;

  resetPeakMemoryStats(device_index);
  at::cuda::clearCublasWorkspaces();
  nvfuser::releaseZeroedMemory();
  ASSERT_EQ(memoryAllocated(device_index), 0)
      << "Previous tests leaked memory.";

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  for (int i = 0; i < 10; i++) {
    TensorView* tv = makeConcreteTensor(sizes);
    tv->setMemoryType(MemoryType::Global);
    auto* allocate = IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
    auto* deallocate = IrBuilder::create<Deallocate>(tv);

    hic->pushBackTopLevelExprs(allocate);
    hic->pushBackTopLevelExprs(deallocate);
  }

  HostIrEvaluator hie(std::move(hic));

  hie.runWithInput({});

  EXPECT_EQ(memoryAllocated(device_index), 0);
}

TEST_F(HostIrIntegrationTest, InsertDeallocations) {
  c10::DeviceIndex device_index = 0;
  at::cuda::clearCublasWorkspaces();
  nvfuser::releaseZeroedMemory();
  resetPeakMemoryStats(device_index);
  ASSERT_EQ(memoryAllocated(device_index), 0)
      << "Previous tests leaked memory.";

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Use 8x8 double tensors so each tensor is >=512 bytes, the minimum cache
  // allocation size (avoid incorrect peak memory stat due to rounding)
  std::vector<int64_t> input_shape{8, 8};
  auto in = TensorViewBuilder()
                .ndims(input_shape.size())
                .dtype(DataType::Double)
                .build();

  fusion->addInput(in);
  TensorView* t0 = add(in, in);
  auto t1 = segment_set(t0);
  TensorView* t2 = add(t1, t1);
  auto t3 = segment_set(t2);
  TensorView* t4 = sin(t3);
  TensorView* out = add(t4, t4);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));

  at::Tensor in_tensor = at::randn(
      input_shape, at::dtype(at::kDouble).device(at::kCUDA, device_index));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  const int64_t max_memory_allocated = maxMemoryAllocated(device_index);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  const std::vector<Expr*>& hicExprs =
      runtime->getHostIrContainer().topLevelExprs();

  EXPECT_THAT(hicExprs, Contains(IsA<Deallocate>()).Times(2));

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");

  if (c10::utils::check_env("PYTORCH_NO_CUDA_MEMORY_CACHING")) {
    GTEST_SKIP() << "Skipped because PYTORCH_NO_CUDA_MEMORY_CACHING is on. "
                    "This usually happens when running with compute-sanitizer. "
                    "maxMemoryAllocated can only collect peak memory "
                    "from a caching allocator.";
  }

  // At any given time a max of three 8x8 tensors are allocated. Here is the
  // flow in the container:
  //  1) Input "in" -> 1 tensor allocated
  //  2) Allocate t1 -> 2 tensors
  //  3) LaunchKernel with input in and output t1 -> 2 tensors
  //  4) Deallocate "in" (which gets invalidated in the HostIrEvaluator, but not
  //  actually deallocated because of the test fixture's reference to it) -> 2
  //  tensors
  //  5) Allocate t3 -> 3 tensors
  //  6) LaunchKernel with input t1 and output t3 -> 3 tensors
  //  7) Deallocate t1 -> 2 tensors
  //  8) Allocate "out" -> 3 tensors
  //  9) LaunchKernel with inputs t3 and output "out" -> 3 tensors
  // 10) Deallocate t3 -> 2 tensors allocated, "in" and "out"
  constexpr int64_t kExpectedPeakMemory = sizeof(double) * (8 * 8) * 3;
  EXPECT_EQ(max_memory_allocated, kExpectedPeakMemory)
      << "Max memory allocated (" << max_memory_allocated
      << ") was higher than expected << (" << kExpectedPeakMemory << ")";
}

TEST_F(HostIrIntegrationTest, ExcludeOutputsFromDeallocations) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> input_shape{8, 8};
  auto in = TensorViewBuilder()
                .ndims(input_shape.size())
                .dtype(DataType::Double)
                .build();

  fusion->addInput(in);
  TensorView* t0 = add(in, in);
  TensorView* t1 = matmul(t0, t0);
  fusion->addOutput(t0);
  fusion->addOutput(t1);

  FusionExecutorCache executor_cache(std::move(fusion));

  at::Tensor in_tensor =
      at::randn(input_shape, at::dtype(at::kDouble).device(at::kCUDA));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  const std::vector<Expr*>& hicExprs =
      runtime->getHostIrContainer().topLevelExprs();

  EXPECT_THAT(hicExprs, Contains(IsA<Deallocate>()).Times(0));

  EXPECT_EQ(out_tensors.size(), 2);
  EXPECT_TRUE(std::all_of(
      out_tensors.begin(), out_tensors.end(), [](const PolymorphicValue& v) {
        return v.as<at::Tensor>().defined();
      }));
}

} // namespace hir

} // namespace nvfuser

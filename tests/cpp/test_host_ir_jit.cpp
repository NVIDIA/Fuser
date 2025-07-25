// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <gmock/gmock-more-matchers.h>

#include <fusion.h>
#include <global_allocator.h>
#include <host_ir/container.h>
#include <host_ir/jit.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace hir {

using HostIrJitTest = NVFuserTest;
// Build with: python setup.py install --build-with-host-ir-jit
TEST_F(HostIrJitTest, Set) {
  auto hic = std::make_unique<HostIrContainer>(1);
  FusionGuard::setCurFusion(hic.get());

  auto hic_in = makeSymbolicTensor(2);
  auto hic_out = set(hic_in);

  hic->addInput(hic_in);
  hic->addOutput(hic_out);

  hic->pushBackTopLevelExprs(hic_out->definition());

  HostIrJit jit(std::move(hic));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in = at::randn({32, 16}, options);

  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  in_args.push(in);
  KernelArgumentHolder outs = jit.runWithInputs(in_args);
  auto out = outs[0].as<at::Tensor>();
  EXPECT_TRUE(at::equal(out, in)) << "Tensors are not equal:\n"
                                  << "in = " << in << "\n"
                                  << "out = " << out;
  EXPECT_EQ(out.strides(), in.strides());
}

TEST_F(HostIrJitTest, HostIrContainer) {
  auto hic = std::make_unique<HostIrContainer>(1);
  FusionGuard::setCurFusion(hic.get());

  int num_inputs = std::rand() % 10 + 1;
  for (int i = 0; i < num_inputs; i++) {
    auto hic_in = makeSymbolicTensor(2);
    auto hic_out = set(hic_in);
    hic->addInput(hic_in);
    hic->addOutput(hic_out);
    hic->pushBackTopLevelExprs(hic_out->definition());
  }
  HostIrJit jit(std::move(hic));
  EXPECT_EQ(jit.container().inputs().size(), num_inputs);
  EXPECT_EQ(jit.container().outputs().size(), num_inputs);
  EXPECT_EQ(jit.container().topLevelExprs().size(), num_inputs);
  EXPECT_EQ(jit.inputs().size(), num_inputs);
  EXPECT_EQ(jit.outputs().size(), num_inputs);
}

TEST_F(HostIrJitTest, Deallocate) {
  const std::vector<int64_t> t0_sizes = {8, 64};
  const std::vector<int64_t> t1_sizes = {16, 32};
  const std::vector<int64_t> t2_sizes = {32, 64};
  c10::DeviceIndex device_index = 0;

  resetPeakMemoryStats(device_index);
  at::cuda::clearCublasWorkspaces();
  nvfuser::releaseZeroedMemory();
  ASSERT_EQ(memoryAllocated(device_index), 0)
      << "Previous tests leaked memory.";

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* t0 = makeConcreteTensor(t0_sizes);
  t0->setMemoryType(MemoryType::Global);
  TensorView* t1 = makeConcreteTensor(t1_sizes);
  t1->setMemoryType(MemoryType::Global);
  TensorView* t2 = makeConcreteTensor(t2_sizes);
  t2->setMemoryType(MemoryType::Global);

  auto* allocate_t0 = IrBuilder::create<kir::Allocate>(t0, MemoryType::Global);
  auto* deallocate_t0 = IrBuilder::create<Deallocate>(t0);
  auto* allocate_t1 = IrBuilder::create<kir::Allocate>(t1, MemoryType::Global);
  auto* deallocate_t1 = IrBuilder::create<Deallocate>(t1);
  auto* allocate_t2 = IrBuilder::create<kir::Allocate>(t2, MemoryType::Global);

  hic->pushBackTopLevelExprs(allocate_t0);
  hic->pushBackTopLevelExprs(allocate_t1);
  hic->pushBackTopLevelExprs(allocate_t2);
  hic->pushBackTopLevelExprs(deallocate_t0);
  hic->pushBackTopLevelExprs(deallocate_t1);

  hic->addOutput(t2);

  // We want to check if the memory is completely freed after output tensor is
  // out of scope
  {
    HostIrJit jit(std::move(hic));
    KernelArgumentHolder in_args;
    in_args.setCacheId(0);
    KernelArgumentHolder outs = jit.runWithInputs(in_args);
    EXPECT_EQ(outs.size(), 1);
    EXPECT_EQ(outs[0].as<at::Tensor>().sizes(), t2_sizes);
  }

  EXPECT_EQ(memoryAllocated(device_index), 0);
}

TEST_F(HostIrJitTest, DynamicSizedTensorAllocate) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* hic_in = makeSymbolicTensor(2);
  TensorView* hic_out = hic_in->split(0, 16)->split(0, 2);
  hic->addInput(hic_in);
  hic->addOutput(hic_out);
  auto* allocate =
      IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  hic->pushBackTopLevelExprs(allocate);

  HostIrJit jit(std::move(hic));
  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in = at::randn({64, 32}, options);
  in_args.push(in);
  KernelArgumentHolder outs = jit.runWithInputs(in_args);
  EXPECT_EQ(outs.size(), 1);
  auto out = outs[0].as<at::Tensor>();
  EXPECT_EQ(out.sizes(), std::vector<int64_t>({64, 32}));
  EXPECT_EQ(out.strides(), std::vector<int64_t>({32, 1}));
}

TEST_F(HostIrJitTest, Reorder) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* hic_in = makeSymbolicTensor(2);
  TensorView* hic_out = hic_in->reorder({1, 0});
  hic_out->setAllocationDomain(hic_in->getLoopDomain(), true);
  hic->addInput(hic_in);
  hic->addOutput(hic_out);
  auto* allocate =
      IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  hic->pushBackTopLevelExprs(allocate);

  HostIrJit jit(std::move(hic));
  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in = at::randn({64, 32}, options);
  in_args.push(in);
  KernelArgumentHolder outs = jit.runWithInputs(in_args);
  EXPECT_EQ(outs.size(), 1);
  auto out = outs[0].as<at::Tensor>();
  EXPECT_EQ(out.sizes(), std::vector<int64_t>({64, 32}));
  EXPECT_EQ(out.strides(), std::vector<int64_t>({1, 64}));
}

TEST_F(HostIrJitTest, Permute) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* hic_in = makeSymbolicTensor(2);
  TensorView* hic_out = permute(hic_in, {1, 0});
  hic_out->setAllocationDomain(hic_out->getLoopDomain(), true);
  hic->addInput(hic_in);
  hic->addOutput(hic_out);
  auto* allocate =
      IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  hic->pushBackTopLevelExprs(allocate);

  HostIrJit jit(std::move(hic));
  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in = at::randn({64, 32}, options);
  in_args.push(in);
  KernelArgumentHolder outs = jit.runWithInputs(in_args);
  EXPECT_EQ(outs.size(), 1);
  auto out = outs[0].as<at::Tensor>();
  EXPECT_EQ(out.sizes(), std::vector<int64_t>({32, 64}));
  EXPECT_EQ(out.strides(), std::vector<int64_t>({64, 1}));
}

TEST_F(HostIrJitTest, AllocationDomainReorder) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* hic_in = makeSymbolicTensor(2);
  TensorView* hic_out = set(hic_in);
  hic_out->setAllocationDomain({hic_out->axis(1), hic_out->axis(0)}, true);
  hic->addInput(hic_in);
  hic->addOutput(hic_out);
  auto* allocate =
      IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  hic->pushBackTopLevelExprs(allocate);

  HostIrJit jit(std::move(hic));
  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in = at::randn({64, 32}, options);
  in_args.push(in);
  KernelArgumentHolder outs = jit.runWithInputs(in_args);
  EXPECT_EQ(outs.size(), 1);
  auto out = outs[0].as<at::Tensor>();
  EXPECT_EQ(out.sizes(), std::vector<int64_t>({64, 32}));
  EXPECT_EQ(out.strides(), std::vector<int64_t>({1, 64}));
}

TEST_F(HostIrJitTest, BroadcastTest) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  auto broadcast_tv = TensorViewBuilder()
                          .ndims(3)
                          .shape({64, 1, 32})
                          .expanded({false, true, false})
                          .dtype(DataType::Float)
                          .build();
  auto expand_tv = TensorViewBuilder()
                       .ndims(3)
                       .shape({64, 32, 16})
                       .expanded({true, true, false})
                       .dtype(DataType::Float)
                       .build();
  hic->addOutput(broadcast_tv);
  hic->addOutput(expand_tv);
  auto* allocate_broadcast =
      IrBuilder::create<kir::Allocate>(broadcast_tv, MemoryType::Global);
  auto* allocate_expand =
      IrBuilder::create<kir::Allocate>(expand_tv, MemoryType::Global);
  hic->pushBackTopLevelExprs(allocate_broadcast);
  hic->pushBackTopLevelExprs(allocate_expand);

  HostIrJit jit(std::move(hic));
  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  KernelArgumentHolder outs = jit.runWithInputs(in_args);
  EXPECT_EQ(outs.size(), 2);
  auto out_broadcast = outs[0].as<at::Tensor>();
  auto out_expand = outs[1].as<at::Tensor>();

  EXPECT_EQ(out_broadcast.sizes(), std::vector<int64_t>({64, 1, 32}));
  EXPECT_EQ(out_broadcast.strides(), std::vector<int64_t>({32, 0, 1}));

  EXPECT_EQ(out_expand.sizes(), std::vector<int64_t>({64, 32, 16}));
  EXPECT_EQ(out_expand.strides(), std::vector<int64_t>({0, 0, 1}));
}

} // namespace hir

} // namespace nvfuser

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

TEST_F(HostIrJitTest, ConstantSizedTensorAllocate) {
  std::vector<std::vector<int64_t>> sizes;
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  int num_tensors = std::rand() % 10 + 1;
  // random size generation
  for (int i = 0; i < num_tensors; i++) {
    std::vector<int64_t> size;
    for (int j = 0; j < std::rand() % 7 + 1; j++) {
      size.push_back(std::rand() % 64 + 1);
    }
    sizes.push_back(size);
  }

  for (int i = 0; i < num_tensors; i++) {
    TensorView* tv = makeConcreteTensor(sizes[i]);
    tv->setMemoryType(MemoryType::Global);
    auto* allocate = IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
    hic->pushBackTopLevelExprs(allocate);
    hic->addOutput(tv);
  }

  HostIrJit jit(std::move(hic));
  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  KernelArgumentHolder outs = jit.runWithInputs(in_args);
  EXPECT_EQ(outs.size(), num_tensors);
  for (int i = 0; i < num_tensors; i++) {
    auto out = outs[i].as<at::Tensor>();
    EXPECT_EQ(out.sizes(), sizes[i])
        << "Tensor " << i << " sizes are not equal";

    std::deque<int64_t> strides;
    strides.push_back(1);
    for (size_t j = sizes[i].size() - 1; j > 0; j--) {
      strides.push_front(strides.front() * out.sizes()[j]);
    }
    for (size_t j = 0; j < out.strides().size(); j++) {
      EXPECT_EQ(out.strides()[j], strides[j])
          << "Tensor " << i << " strides are not equal";
    }
  }
}

TEST_F(HostIrJitTest, Deallocate) {
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

  HostIrJit jit(std::move(hic));
  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  KernelArgumentHolder outs = jit.runWithInputs(in_args);
  EXPECT_EQ(outs.size(), 0);

  EXPECT_EQ(memoryAllocated(device_index), 0);
}

TEST_F(HostIrJitTest, DynamicSizedTensorAllocate) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* hic_in = makeSymbolicTensor(2);
  TensorView* hic_out = hic_in->split(0, 16)->split(0, 2);
  hic->addInput(hic_in);
  hic->addOutput(hic_out);
  auto* allocate = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
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

TEST_F(HostIrJitTest, Permute) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* hic_in = makeSymbolicTensor(2);
  TensorView* hic_out = hic_in->reorder({1, 0});
  hic_out->setAllocationDomain(hic_in->getLoopDomain(),true);
  hic->addInput(hic_in);
  hic->addOutput(hic_out);
  auto* allocate = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
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

} // namespace hir

} // namespace nvfuser

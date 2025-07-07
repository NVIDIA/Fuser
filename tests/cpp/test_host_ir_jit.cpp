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
#include <host_ir/jit.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <random>

namespace nvfuser {

namespace hir {

using HostIrJitTest = NVFuserTest;
// Build with: python setup.py install --build-with-host-ir-jit
// Run with: NVFUSER_ENABLE=host_ir_lowering ./bin/test_host_ir_jit
// --gtest_filter=HostIrJitTest.TestJITAtenCall
TEST_F(HostIrJitTest, TestJITAtenCall) {
  // Fusion fusion;
  // FusionGuard fg(&fusion);
  // TensorView* in = makeSymbolicTensor(2);
  // fusion.addInput(in);

  // TensorView* out = set(in);
  // fusion.addOutput(out);

  // auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // at::Tensor t0 = at::randn({32, 32}, options);
  // auto ke = std::make_unique<KernelExecutor>();
  // ke->setGroupId(0);
  // ke->compile(&fusion, {t0});

  // auto hic = std::make_unique<HostIrContainer>(1);
  // FusionGuard::setCurFusion(hic.get());

  // hic->addKernelExecutor(std::move(ke));

  // IrCloner ir_cloner(hic.get());
  // auto hic_in = ir_cloner.clone(in);
  // auto hic_out = ir_cloner.clone(out);

  // hic->addInput(hic_in);
  // hic->addOutput(hic_out);

  // // Adjust the number of allocates and calls to each allocate
  // int num_allocates = 10;
  // int num_calls_per_allocate = 10;
  // std::vector<kir::Allocate*> allocates;
  // for(int i = 0; i < num_allocates; i++) {
  //   auto* allocate =
  //     IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  //   allocates.push_back(allocate);
  //   hic->pushBackTopLevelExprs(allocate);
  // }
  
  // auto* cache_id = IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
  // auto launch_kernel = IrBuilder::create<LaunchKernel>(
  //     0,
  //     LaunchParams(),
  //     CompileParams(),
  //     std::vector<Val*>{hic_in},
  //     std::vector<Val*>{hic_out},
  //     cache_id);

  // hic->pushBackTopLevelExprs(launch_kernel);

  // HostIrJit jit(hic.get());
  // for(auto* allocate : allocates) {
  //   for(int i = 0; i < num_calls_per_allocate; i++) {
  //     int first_dim = std::rand() % 100;
  //     int second_dim = std::rand() % 100;
  //     auto allocated_t = jit.allocate(allocate, {first_dim, second_dim}, {second_dim, 1});
  //     EXPECT_EQ(allocated_t.sizes(), at::IntArrayRef({first_dim, second_dim}));
  //     EXPECT_EQ(allocated_t.strides(), at::IntArrayRef({second_dim, 1}));
  //   }
  // }
}

TEST_F(HostIrJitTest, TestJITRunFullGraph) {
  //  Fusion fusion;
  // FusionGuard fg(&fusion);
  // TensorView* in = makeSymbolicTensor(2);
  // fusion.addInput(in);

  // TensorView* out = set(in);
  // fusion.addOutput(out);

  // auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // at::Tensor t0 = at::randn({32, 32}, options);
  // auto ke = std::make_unique<KernelExecutor>();
  // ke->setGroupId(0);
  // ke->compile(&fusion, {t0});

  // auto hic = std::make_unique<HostIrContainer>(1);
  // FusionGuard::setCurFusion(hic.get());

  // hic->addKernelExecutor(std::move(ke));

  // IrCloner ir_cloner(hic.get());
  // auto hic_in = ir_cloner.clone(in);
  // auto hic_out = ir_cloner.clone(out);

  // hic->addInput(hic_in);
  // hic->addOutput(hic_out);

  // // Adjust the number of allocates and calls to each allocate
  // int num_allocates = 2;
  // std::vector<kir::Allocate*> allocates;
  // for(int i = 0; i < num_allocates; i++) {
  //   auto* allocate =
  //     IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  //   allocates.push_back(allocate);
  //   hic->pushBackTopLevelExprs(allocate);
  // }
  
  // auto* cache_id = IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
  // auto launch_kernel = IrBuilder::create<LaunchKernel>(
  //     0,
  //     LaunchParams(),
  //     CompileParams(),
  //     std::vector<Val*>{hic_in},
  //     std::vector<Val*>{hic_out},
  //     cache_id);

  // hic->pushBackTopLevelExprs(launch_kernel);
  // std::unordered_map<Val*, PolymorphicValue> inputs;
  // HostIrJit jit(hic.get());

  // int first_dim = std::rand() % 100;
  // int second_dim = std::rand() % 100;
  // inputs[hic_in] = at::empty({first_dim, second_dim}, options);

  // auto result = jit.runFullGraph(hic.get(), inputs);
  // for(size_t i = 0; i < result.size(); i++) {
  //   EXPECT_EQ(result[i].sizes(), inputs[hic_in].as<at::Tensor>().sizes());
  //   EXPECT_EQ(result[i].strides(), inputs[hic_in].as<at::Tensor>().strides());
  // }
}

TEST_F(HostIrJitTest, TestJITRunFullGraphStrideInference) {
  //  Fusion fusion;
  // FusionGuard fg(&fusion);
  // TensorView* in = makeSymbolicTensor(2);
  // fusion.addInput(in);

  // TensorView* out = set(in);
  // out->split(0, 2);
  // out->split(1, 4);
  // out->setAllocationDomain({out->axis(0), out->axis(1), out->axis(3),out->axis(2)}, std::vector<std::optional<bool>>(out->getLoopDomain().size(), true));
  // fusion.addOutput(out);

  // auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // at::Tensor t0 = at::randn({32, 32}, options);
  // auto ke = std::make_unique<KernelExecutor>();
  // ke->setGroupId(0);
  // ke->compile(&fusion, {t0});

  // auto hic = std::make_unique<HostIrContainer>(1);
  // FusionGuard::setCurFusion(hic.get());

  // hic->addKernelExecutor(std::move(ke));

  // IrCloner ir_cloner(hic.get());
  // auto hic_in = ir_cloner.clone(in);
  // auto hic_out = ir_cloner.clone(out);

  // hic->addInput(hic_in);
  // hic->addOutput(hic_out);
  // // Adjust the number of allocates and calls to each allocate
  // int num_allocates = 2;
  // std::vector<kir::Allocate*> allocates;
  // for(int i = 0; i < num_allocates; i++) {
  //   auto* allocate =
  //     IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  //   allocates.push_back(allocate);
  //   hic->pushBackTopLevelExprs(allocate);
  // }
  
  // auto* cache_id = IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
  // auto launch_kernel = IrBuilder::create<LaunchKernel>(
  //     0,
  //     LaunchParams(),
  //     CompileParams(),
  //     std::vector<Val*>{hic_in},
  //     std::vector<Val*>{hic_out},
  //     cache_id);

  // hic->pushBackTopLevelExprs(launch_kernel);
  // std::unordered_map<Val*, PolymorphicValue> inputs;
  // HostIrJit jit(hic.get());

  // int first_dim = 16;
  // int second_dim = 8;
  // inputs[hic_in] = at::empty({first_dim, second_dim}, options);

  // auto result = jit.runFullGraph(hic.get(), inputs);
  // std::cout << "result[0].sizes(): " << result[0].sizes() << std::endl;
  // std::cout << "result[0].strides(): " << result[0].strides() << std::endl;
}
} // namespace hir

} // namespace nvfuser

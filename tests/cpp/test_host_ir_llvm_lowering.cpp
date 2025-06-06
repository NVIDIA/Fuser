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
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <host_ir/lower_to_llvm.h>

namespace nvfuser {

namespace hir {

using testing::Contains;
using HostIrLLVMTest = NVFuserTest;


TEST_F(HostIrLLVMTest, Allocation1) {

  // Fusion Definition
  Fusion fusion;
  FusionGuard fg(&fusion);
  int n1 = 31, n2 = 29, h = 64, w = 104, c = 21;
  auto tv0 = makeContigTensor(3); // [N1, N2, H*W*C]
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);
  tv1->merge(0);
  tv1->split(1, w);
  tv1->split(1, h);
  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(1);
  // [N, H*W, C]
  tv1->setAllocationDomain({tv1->axis(0), tv1->axis(1), tv1->axis(2)}, {true, true, true});

  // LLVM JIT Compile
  auto allocation_domain = tv1->getAllocationDomain();
  auto logical_domain = tv1->getLogicalDomain();
  std::unique_ptr<llvm::orc::LLJIT> JIT = llvm_jit_init(4);
  llvm_jit_compile_shape_infer(JIT, fusion, logical_domain, logical_domain);
  llvm_jit_compile_stride_infer(JIT, fusion, allocation_domain, logical_domain);

  auto func_infer_shape = ExitOnErr(JIT->lookup("infer_shape"));
  auto func_infer_stride = ExitOnErr(JIT->lookup("infer_stride"));
  FuncType func_shape = func_infer_shape.toPtr<FuncType>();
  FuncType func_stride = func_infer_stride.toPtr<FuncType>();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({n1, n2, h*w*c}, options);
  at::Tensor output_tensor = aten_output_allocation(func_shape, func_stride, t0, logical_domain.size());
  print_tensor_info(output_tensor);
}

TEST_F(HostIrLLVMTest, Allocation2) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(5);
  fusion.addInput(in);
  in->merge(0,1)->split(0,4)->merge(0,1)->split(0,2);
  TensorView* out = set(in);
  out->merge(0,1)->split(0,8)->merge(0,1)->split(0,2);
  fusion.addOutput(out);
  // Input Tensor
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8,8,16,32,16}, options);

  // LLVM JIT Compile
  out->setAllocationDomain(out->getLoopDomain(), {true, true, true, true, true});
  auto in_logical_domain = in->getLogicalDomain();
  auto out_logical_domain = out->getLogicalDomain();
  auto allocation_domain = out->getMaybeAllocationDomain();
  // do a forward declaration 
  std::unique_ptr<llvm::orc::LLJIT> JIT = llvm_jit_init(4);
  llvm_jit_compile_shape_infer(JIT, fusion, in_logical_domain, out_logical_domain);
  llvm_jit_compile_stride_infer(JIT, fusion, allocation_domain, out_logical_domain);
  auto func_infer_shape = ExitOnErr(JIT->lookup("infer_shape"));
  auto func_infer_stride = ExitOnErr(JIT->lookup("infer_stride"));
  FuncType func_shape = func_infer_shape.toPtr<FuncType>();
  FuncType func_stride = func_infer_stride.toPtr<FuncType>();

  at::Tensor output_tensor = aten_output_allocation(func_shape, func_stride, t0, out_logical_domain.size());
  
  print_tensor_info(output_tensor);

  // HostIr Compile
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
  // auto allocate = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  // auto* cache_id = IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
  // auto launch_kernel = IrBuilder::create<LaunchKernel>(
  //     0,
  //     LaunchParams(),
  //     CompileParams(),
  //     std::vector<Val*>{hic_in},
  //     std::vector<Val*>{hic_out},
  //     cache_id);
  // hic->pushBackTopLevelExprs(allocate);
  // hic->pushBackTopLevelExprs(launch_kernel);
  // HostIrEvaluator hie(std::move(hic));
  // auto outputs = hie.runWithInput({{hic_in, t0}});

  // Compare the output tensor
  // print_compare_tensor(t0, outputs[0].as<at::Tensor>());
  // EXPECT_TRUE(outputs[0].as<at::Tensor>().equal(output_tensor));
}

TEST_F(HostIrLLVMTest, Allocation3) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(5);
  fusion.addInput(in);
  in->merge(0,1)->split(0,4)->merge(0,1)->split(0,2);
  TensorView* out = set(in);
  out->merge(0,1)->split(0,8)->merge(0,1)->split(0,2);
  out->reorder({{1, 2, 3, 4, 0}});
  out->setAllocationDomain(out->getLoopDomain(), {true, true, true, true, true});
  fusion.addOutput(out);
  auto in_logical_domain = in->getLogicalDomain();
  auto out_logical_domain = out->getLogicalDomain();
  auto allocation_domain = out->getMaybeAllocationDomain();
  std::unique_ptr<llvm::orc::LLJIT> JIT = llvm_jit_init(4);
  llvm_jit_compile_shape_infer(JIT, fusion, in_logical_domain, out_logical_domain);
  llvm_jit_compile_stride_infer(JIT, fusion, allocation_domain, out_logical_domain);
  auto func_infer_shape = ExitOnErr(JIT->lookup("infer_shape"));
  auto func_infer_stride = ExitOnErr(JIT->lookup("infer_stride"));
  FuncType func_shape = func_infer_shape.toPtr<FuncType>();
  FuncType func_stride = func_infer_stride.toPtr<FuncType>();
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8,8,16,32,16}, options);
  at::Tensor output_tensor = aten_output_allocation(func_shape, func_stride, t0, out_logical_domain.size());
  print_tensor_info(output_tensor);
}


} // namespace hir

} // namespace nvfuser

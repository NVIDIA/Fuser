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
using HostIrEvaluatorTest = NVFuserTest;

TEST_F(HostIrEvaluatorTest, LaunchKernel4) {
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
  tv1->printTransforms();

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
  at::Tensor t0 = at::randn({2,n1, n2, c * h * w,4}, options);
  at::Tensor output_tensor = aten_output_allocation(func_shape, func_stride, t0, logical_domain.size());
  print_compare_tensor(output_tensor, t0);
}

// TEST_F(HostIrEvaluatorTest, LaunchKernel3) {
//     // Simple reshape example
//     Fusion fusion;
//     FusionGuard fg(&fusion);

//     auto tv0 = makeSymbolicTensor(4);
//     #define OUTPUT_DIM 2
//     #define INPUT_DIM 4
//     fusion.addInput(tv0);

//     // Shape of tv0 is assumed to be [4, 8], which is then reshaped to [32]
//     auto tv1 = permute(tv0, {1, 0, 2, 3});
//     tv1->merge(2,3);
//     tv1->split(0, 4);
//     // tv1->merge(0,1);
//     tv1->setAllocationDomain({tv1->getLoopDomain().begin(), tv1->getLoopDomain().end()}, {true, true, true, true});
//     fusion.addOutput(tv1);

//     auto logical_domain = tv1->getLogicalDomain();
//     auto allocation_domain = tv1->getAllocationDomain();
//     std::cout << "logical_domain size: " << logical_domain.size() << std::endl;
//     std::cout << "allocation_domain size: " << allocation_domain.size() << std::endl;
//     std::unique_ptr<llvm::orc::LLJIT> JIT = llvm::orc::llvm_jit_compile_stride_infer(fusion, allocation_domain, logical_domain);
//     if(JIT == nullptr){
//       std::cout << "JIT is nullptr" << std::endl;
//       return;
//     }
//     auto ke = std::make_unique<KernelExecutor>();
//     ke->setGroupId(0);
//     auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
//     at::Tensor t0 = at::randn({4, 8, 16, 32}, options);
//     ke->compile(&fusion, {t0});
//     auto func_infer_stride = llvm::orc::ExitOnErr(JIT->lookup("infer_stride"));
//     using FuncType = void (*)(const int64_t* input, int64_t input_len, int64_t* output, int64_t output_len);
//     FuncType func_stride = func_infer_stride.toPtr<FuncType>();
//     auto outputs = llvm::orc::stride_infer_runtime(func_stride, t0);
//     for(auto output : outputs){
//       std::cout << "output: " << output << std::endl;
//     }
// }


// run with the following command: NVFUSER_ENABLE=host_ir_lowering ./bin/test_host_ir --gtest_filter=HostIrEvaluatorTest.LaunchKernel2
TEST_F(HostIrEvaluatorTest, LaunchKernel2) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(5);
  fusion.addInput(in);
  in->merge(0,1)->split(0,4)->merge(0,1);
  TensorView* out = set(in);
  out->merge(0,1)->split(0,8)->merge(0,1);
  fusion.addOutput(out);
  out->setAllocationDomain({out->axis(0), out->axis(1), out->axis(2), out->axis(3)}, {true, true, true, true});
  out->printTransforms();
  auto in_logical_domain = in->getLogicalDomain();
  auto out_logical_domain = out->getLogicalDomain();
  auto allocation_domain = out->getAllocationDomain();
  std::unique_ptr<llvm::orc::LLJIT> JIT = llvm_jit_init(4);
  llvm_jit_compile_shape_infer(JIT, fusion, in_logical_domain, out_logical_domain);
  llvm_jit_compile_stride_infer(JIT, fusion, allocation_domain, out_logical_domain);
  
  auto func_infer_shape = ExitOnErr(JIT->lookup("infer_shape"));
  auto func_infer_stride = ExitOnErr(JIT->lookup("infer_stride"));

  FuncType func_shape = func_infer_shape.toPtr<FuncType>();
  FuncType func_stride = func_infer_stride.toPtr<FuncType>();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4,8,16,32,16}, options);
  at::Tensor output_tensor = aten_output_allocation(func_shape, func_stride, t0, out_logical_domain.size());
  
  print_compare_tensor(output_tensor, t0);
  auto ke = std::make_unique<KernelExecutor>();
  ke->setGroupId(0);
  ke->compile(&fusion, {t0});
  auto outputs = ke->run({t0}, {output_tensor}, LaunchParams(), CompileParams());
  EXPECT_TRUE(outputs[0].as<at::Tensor>().equal(t0));
}

} // namespace hir

} // namespace nvfuser

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

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

// Print tensor info
void print_tensor_info(const at::Tensor& t){
  std::cout << "Tensor dtype: " << t.dtype() << "\n";
  std::cout << "Shape: " << t.sizes() << "\n"; 
  std::cout << "Strides: " << t.strides() << "\n";
  std::cout << "Is Contiguous: " << t.is_contiguous() << "\n";
  std::cout << "Device: " << t.device() << "\n";
  std::cout << "Data ptr: " << t.data_ptr() << "\n";
}

void print_iter_domain(const std::vector<IterDomain*>& iter_domain, const std::string& name){
  std::cout << name << ": ";
  for(auto* id : iter_domain){
    std::cout << id->toString() << " ";
  }
  std::cout << std::endl;
}

TEST_F(HostIrLLVMTest, Allocation1) {
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
  tv1->setAllocationDomain(tv1->getLoopDomain(), {true, true, true});
  print_iter_domain(tv1->getLoopDomain(), "Output Loop Domain");
  print_iter_domain(tv1->getLogicalDomain(), "Input Logical Domain");
  // LLVM JIT Compile
  HostIrLlvmJit jit(4);
  jit.compile(tv1);

  // Input Tensor
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({n1, n2, h*w*c}, options);

  // LLVM JIT Run Allocation
  at::Tensor output_tensor = jit.allocateOutputTensor({t0});

  // Print Output Tensor Info
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
  print_iter_domain(in->getLogicalDomain(), "Input Logical Domain");
  print_iter_domain(out->getLoopDomain(), "Output Loop Domain");
  out->setAllocationDomain(out->getLoopDomain(), {true, true, true, true, true});
  // Input Tensor
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8,8,16,32,16}, options);
  print_iter_domain(in->getLogicalDomain(), "Input Logical Domain");
  print_iter_domain(out->getLoopDomain(), "Output Loop Domain");
  // LLVM JIT Compile
  HostIrLlvmJit jit(4);
  jit.compile(out);

  // LLVM JIT Run Allocation
  auto output_tensor = jit.allocateOutputTensor({t0});

  // Print Output Tensor Info
  print_tensor_info(output_tensor);
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

  // Input Tensor
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8,8,16,32,16}, options);

  // LLVM JIT Compile
  HostIrLlvmJit jit(4);
  jit.compile(out);

  // LLVM JIT Run Allocation
  auto output_tensor = jit.allocateOutputTensor({t0});

  // Print Output Tensor Info
  print_tensor_info(output_tensor);
}

TEST_F(HostIrLLVMTest, Allocation4) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  int N = 16, H = 16, W = 16, C = 16;
  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  // [N, H, W, C]
  tv0->merge(0,1);
  // [N * H, W, C]
  tv0->split(2,4);
  // [N * H, W, C/4, 4]
  tv0->split(3,2);
  // [N * H, W, C/4, 2, 2]
  tv0->merge(1,2);
  // [N * H, W * C/4, 2, 2]
  TensorView* tv1 = set(tv0);
  
  tv1->setAllocationDomain(tv1->getLoopDomain(), {true, true, true, true});

  TensorView* tv2 = makeContigConcreteTensor({N, H, W});
  fusion.addInput(tv2);
  auto tv3 = broadcast(tv2, {false, false, false, true});
  auto tv4 = add(tv2, tv3);

  fusion.addOutput(tv4);

  // Input Tensor
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({N, H, W, C}, options);
  at::Tensor t1 = at::randn({N, H, W}, options);
  std::cout << "check point 1" << std::endl;
  // LLVM JIT Compile
  HostIrLlvmJit jit(4);
  jit.compile(tv4);
  tv4->setAllocationDomain(tv4->getLoopDomain(), {true, true, true, true});
  std::cout << "check point 2" << std::endl;
  tv4->printTransforms();
  // LLVM JIT Run Allocation
  auto output_tensor = jit.allocateOutputTensor({t0, t1});

  // Print Output Tensor Info
  print_tensor_info(output_tensor);
}

TEST_F(HostIrLLVMTest, Allocate5) {
  int N = 32, H = 32, W = 32, C = 32;
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  tv0->merge(0,1);

  tv0->split(2,4);
  auto tv1 = set(tv0);
  tv1->commitLeafToLogical();
  tv1->merge(0,1);
  tv1->setAllocationDomain(tv1->getLoopDomain(),{true, true, true});
  fusion.addOutput(tv1);
  // Input Tensor
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({N, H, W, C}, options);

  HostIrLlvmJit jit(4);
  jit.compile(tv1);
  // LLVM JIT Run Allocation
  auto output_tensor = jit.allocateOutputTensor({t0});
  // Print Output Tensor Info
  print_tensor_info(output_tensor);
}

} // namespace hir

} // namespace nvfuser

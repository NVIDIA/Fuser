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
#include <random>

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
  KernelArgumentHolder outs1 = jit.runWithInput({{hic_in, in}});
  auto out1 = outs1[0].as<at::Tensor>();

  EXPECT_EQ(out1.sizes(), in.sizes()) << "Sizes are not equal:\n"
                                      << "in = " << in << "\n"
                                      << "out1 = " << out1;
  EXPECT_EQ(out1.strides(), in.strides()) << "Strides are not equal:\n"
                                          << "in = " << in << "\n"
                                          << "out1 = " << out1;
  EXPECT_EQ(at::equal(out1, in), true) << "Tensors are not equal:\n"
                                       << "in = " << in << "\n"
                                       << "out1 = " << out1;

  KernelArgumentHolder in_args;
  in_args.setCacheId(0);
  in_args.push(in);
  KernelArgumentHolder outs2 = jit.runWithInputs(in_args);
  auto out2 = outs2[0].as<at::Tensor>();
  EXPECT_EQ(out2.sizes(), in.sizes()) << "Sizes are not equal:\n"
                                      << "in = " << in << "\n"
                                      << "out2 = " << out2;
  EXPECT_EQ(out2.strides(), in.strides()) << "Strides are not equal:\n"
                                          << "in = " << in << "\n"
                                          << "out2 = " << out2;
  EXPECT_EQ(at::equal(out2, in), true) << "Tensors are not equal:\n"
                                       << "in = " << in << "\n"
                                       << "out2 = " << out2;
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
  EXPECT_EQ(jit.getHostIrContainer().inputs().size(), num_inputs);
  EXPECT_EQ(jit.getHostIrContainer().outputs().size(), num_inputs);
  EXPECT_EQ(jit.inputs().size(), num_inputs);
  EXPECT_EQ(jit.outputs().size(), num_inputs);
}

} // namespace hir

} // namespace nvfuser

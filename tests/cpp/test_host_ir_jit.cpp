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
  int run_iterations = 10;
  for (int i = 0; i < run_iterations; i++) {
    int dim0 = std::rand() % 100 + 1;
    int dim1 = std::rand() % 100 + 1;
    at::Tensor t0 = at::randn({dim0, dim1}, options);
    auto output_args = jit.runWithInput({{hic_in, t0}});
    auto output = output_args[0].as<at::Tensor>();

    EXPECT_EQ(output.sizes(), t0.sizes());
    EXPECT_EQ(output.strides(), t0.strides());
  }
}

} // namespace hir

} // namespace nvfuser

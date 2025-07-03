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
TEST_F(HostIrJitTest, Allocate) {
  auto hic = std::make_unique<HostIrContainer>(1);
  FusionGuard::setCurFusion(hic.get());

  auto hic_in = makeSymbolicTensor(2);
  auto hic_out = set(hic_in);

  hic->addInput(hic_in);
  hic->addOutput(hic_out);

  // Adjust the number of allocates and calls to each allocate
  int num_allocates = 10;
  int num_calls_per_allocate = 10;
  for (int i = 0; i < num_allocates; i++) {
    auto* allocate =
        IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
    hic->pushBackTopLevelExprs(allocate);
  }

  HostIrJit jit(hic.get());
  for (auto* expr : hic->topLevelExprs()) {
    if (auto* allocate = dynamic_cast<kir::Allocate*>(expr)) {
      for (int i = 0; i < num_calls_per_allocate; i++) {
        int first_dim = std::rand() % 100;
        int second_dim = std::rand() % 100;
        auto allocated_t =
            jit.allocate(allocate, {first_dim, second_dim}, {second_dim, 1});
        EXPECT_EQ(
            allocated_t.sizes(), at::IntArrayRef({first_dim, second_dim}));
        EXPECT_EQ(allocated_t.strides(), at::IntArrayRef({second_dim, 1}));
      }
    }
  }
}

} // namespace hir

} // namespace nvfuser

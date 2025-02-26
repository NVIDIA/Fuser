// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <fusion.h>
#include <kernel_ir_dispatch.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <gtest/gtest.h>


namespace nvfuser {

using RemoveTrivialOpsTest = NVFuserTest;

template<typename OpType>
class KernelContainsExpr : kir::IrVisitor {
 public:
  static bool check(const kir::Kernel* kernel) {
    KernelContainsExpr checker;
    checker.handle(kernel->topLevelExprs());
    return checker.has_op_;
  }
 private:
  using kir::IrVisitor::handle;

  void handle(OpType* op) final {
    has_op_ = true;
  }

 private:
  bool has_op_ = false;
};

// Test that we remove a trivial gmem->gmem broadcast at lowering
TEST_F(RemoveTrivialOpsTest, Broadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3});
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true, false});
  auto tv2 = neg(tv1);
  fusion.addOutput(tv2);
  
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);
  std::vector<c10::IValue> inputs{t0};

  {
    // In this case we do not remove the broadcast since it is G->L not G->G
    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<BroadcastOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    // Setting tv1 to Global means the BroadcastOp is now G->G. This means that
    // when we lower it, we will be able to safely remove it
    tv1->setMemoryType(MemoryType::Global);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_FALSE(KernelContainsExpr<BroadcastOp>::check(kernel));
    EXPECT_EQ(kernel->summary().global_allocations.size(), 0)
        << "Expected to have no intermediate global allocations";
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    NVF_ERROR(tv1->getMemoryType() == MemoryType::Global);
    // Transpose the input's allocation domain
    tv0->setAllocationDomain(
        {
            tv0->getLogicalDomain().at(1),
            tv0->getLogicalDomain().at(0),
        },
        {true, true});

    KernelExecutor ke;

    std::vector<c10::IValue> transposed_inputs{at::randn({3, 2}, options).t()};
    ke.compile(&fusion, transposed_inputs);
    // We do not remove the broadcast because the allocation domains do not
    // match
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<BroadcastOp>::check(kernel));
    EXPECT_EQ(kernel->summary().global_allocations.size(), 1)
        << "Expected to have one intermediate global allocation";
    auto outputs = ke.run(transposed_inputs);
    testValidate(&fusion, outputs, transposed_inputs, __LINE__, __FILE__);
  }

  {
    // Reset tv0's allocation domain but set tv1 as output
    tv0->setAllocationDomain(tv0->getLogicalDomain(), {true, true});
    fusion.addOutput(tv1);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    // We do not remove the broadcast because the output is a fusion output
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<BroadcastOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }
}

// TODO: tests for permutations, squeeze, and chain of ops

} // namespace nvfuser

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

template <typename OpType>
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

// Test that we remove a trivial gmem->gmem set at lowering unless it is an
// alias.
TEST_F(RemoveTrivialOpsTest, Set) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = neg(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);
  std::vector<c10::IValue> inputs{t0};

  {
    // In this case we do not remove the squeeze since it is G->L not G->G
    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    // Setting tv1 to Global means the LoadStoreOp is now G->G. This means that
    // when we lower it, we will be able to safely remove it
    tv1->setMemoryType(MemoryType::Global);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_FALSE(KernelContainsExpr<LoadStoreOp>::check(kernel));
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
        true);

    KernelExecutor ke;

    std::vector<c10::IValue> transposed_inputs{at::randn({3, 2}, options).t()};
    ke.compile(&fusion, transposed_inputs);
    // We do not remove the set because the allocation domains do not match
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    EXPECT_EQ(kernel->summary().global_allocations.size(), 1)
        << "Expected to have one intermediate global allocation";
    auto outputs = ke.run(transposed_inputs);
    testValidate(&fusion, outputs, transposed_inputs, __LINE__, __FILE__);
  }

  {
    // Reset tv0's allocation domain but add a new output which reuses the
    // buffer of the input tv0
    tv0->setAllocationDomain(tv0->getLogicalDomain(), true);
    auto* tv3 = exp(tv2);
    fusion.addOutput(tv3);
    fusion.aliasOutputToInput(tv3, tv0, AllocationType::ReuseBuffer);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    // We do not remove the squeeze because the input is target of an io alias
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    // Run with copy of inputs since we will overwrite the original inputs,
    // complicating validation
    std::vector<c10::IValue> inputs_copy{t0.clone()};
    auto outputs = ke.run(inputs_copy);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }
}

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
        true);

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
    tv0->setAllocationDomain(tv0->getLogicalDomain(), true);
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

// Test that we remove a trivial gmem->gmem squeeze at lowering
TEST_F(RemoveTrivialOpsTest, Squeeze) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 1, 3});
  fusion.addInput(tv0);
  auto tv1 = squeeze(tv0, {1});
  auto tv2 = neg(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 1, 3}, options);
  std::vector<c10::IValue> inputs{t0};

  {
    // In this case we do not remove the squeeze since it is G->L not G->G
    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    // NOTE: SqueezeOp is converted to LoadStoreOp by the loadStoreOpInserter
    // lowering pass
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    // Setting tv1 to Global means the SqueezeOp is now G->G. This means that
    // when we lower it, we will be able to safely remove it
    tv1->setMemoryType(MemoryType::Global);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_FALSE(KernelContainsExpr<LoadStoreOp>::check(kernel));
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
            tv0->getLogicalDomain().at(2),
            tv0->getLogicalDomain().at(1),
            tv0->getLogicalDomain().at(0),
        },
        true);

    KernelExecutor ke;

    std::vector<c10::IValue> transposed_inputs{
        at::randn({3, 2}, options).t().unsqueeze(1)};
    ke.compile(&fusion, transposed_inputs);
    // We do not remove the broadcast because the allocation domains do not
    // match
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    EXPECT_EQ(kernel->summary().global_allocations.size(), 1)
        << "Expected to have one intermediate global allocation";
    auto outputs = ke.run(transposed_inputs);
    testValidate(&fusion, outputs, transposed_inputs, __LINE__, __FILE__);
  }

  {
    // Reset tv0's allocation domain but set tv1 as output
    tv0->setAllocationDomain(tv0->getLogicalDomain(), true);
    fusion.addOutput(tv1);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    // We do not remove the squeeze because the output is a fusion output
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }
}

// Test that we remove a gmem->gmem permute whose allocation domain is
// unpermuted at lowering
TEST_F(RemoveTrivialOpsTest, Permute) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3});
  fusion.addInput(tv0);
  // This is a logical transpose. It implies data shuffling unless the
  // allocation domain is also transposed.
  auto tv1 = transpose(tv0);
  auto tv2 = neg(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);
  std::vector<c10::IValue> inputs{t0};

  tv1->setAllocationDomain(
      {
          tv1->getLogicalDomain().at(1),
          tv1->getLogicalDomain().at(0),
      },
      true);

  {
    // In this case we do not remove the permute since it is G->L not G->G
    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    // Setting tv1 to Global means the LoadStoreOp is now G->G. This means that
    // when we lower it, we will be able to safely remove it
    tv1->setMemoryType(MemoryType::Global);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_FALSE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    EXPECT_EQ(kernel->summary().global_allocations.size(), 0)
        << "Expected to have no intermediate global allocations";
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    NVF_ERROR(tv1->getMemoryType() == MemoryType::Global);
    tv1->setAllocationDomain(tv1->getLogicalDomain(), true);

    KernelExecutor ke;

    ke.compile(&fusion, inputs);
    // We do not remove the transpose because the allocation domains do not
    // match
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    EXPECT_EQ(kernel->summary().global_allocations.size(), 1)
        << "Expected to have one intermediate global allocation";
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    // Reset tv1's allocation domain but also set it as an output
    tv1->setAllocationDomain(
        {
            tv1->getLogicalDomain().at(1),
            tv1->getLogicalDomain().at(0),
        },
        true);
    fusion.addOutput(tv1);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    // We do not remove the transpose because the output is a fusion output
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }
}

// Test that we remove a gmem->gmem->gmem broadcast+squeeze at lowering
TEST_F(RemoveTrivialOpsTest, BroadcastSqueeze) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3});
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true, false});
  auto tv2 = squeeze(tv1, {1});
  auto tv3 = neg(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);
  std::vector<c10::IValue> inputs{t0};

  {
    // In this case we do not remove the broadcast since it is G->L not G->G,
    // and the squeeze is L->L
    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    // NOTE: SqueezeOp is converted to LoadStoreOp by the loadStoreOpInserter
    // lowering pass
    EXPECT_TRUE(KernelContainsExpr<BroadcastOp>::check(kernel));
    EXPECT_TRUE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    // Set G->G->G so that both ops get removed
    tv1->setMemoryType(MemoryType::Global);
    tv2->setMemoryType(MemoryType::Global);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_FALSE(KernelContainsExpr<BroadcastOp>::check(kernel));
    EXPECT_FALSE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    EXPECT_EQ(kernel->summary().global_allocations.size(), 0)
        << "Expected to have no intermediate global allocations";
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }

  {
    NVF_ERROR(tv1->getMemoryType() == MemoryType::Global);
    NVF_ERROR(tv2->getMemoryType() == MemoryType::Global);
    // Transpose the input's allocation domain
    tv0->setAllocationDomain(
        {
            tv0->getLogicalDomain().at(1),
            tv0->getLogicalDomain().at(0),
        },
        true);

    KernelExecutor ke;

    std::vector<c10::IValue> transposed_inputs{at::randn({3, 2}, options).t()};
    ke.compile(&fusion, transposed_inputs);
    // We do not remove the broadcast because the allocation domains do not
    // match
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<BroadcastOp>::check(kernel));
    EXPECT_FALSE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    EXPECT_EQ(kernel->summary().global_allocations.size(), 1)
        << "Expected to have one intermediate global allocation";
    auto outputs = ke.run(transposed_inputs);
    testValidate(&fusion, outputs, transposed_inputs, __LINE__, __FILE__);
  }

  {
    // Reset tv0's allocation domain but set tv1 as output
    tv0->setAllocationDomain(tv0->getLogicalDomain(), true);
    fusion.addOutput(tv1);

    KernelExecutor ke;
    ke.compile(&fusion, inputs);
    // We do not remove the broadcast because the output is a fusion output, but
    // we _do_ remove the squeeze
    const kir::Kernel* kernel = ke.compiledKernel()->kernel();
    EXPECT_TRUE(KernelContainsExpr<BroadcastOp>::check(kernel));
    EXPECT_FALSE(KernelContainsExpr<LoadStoreOp>::check(kernel));
    auto outputs = ke.run(inputs);
    testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
  }
}

} // namespace nvfuser

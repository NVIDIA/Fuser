// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <dynamic_transform.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using TopKDynamicTest = NVFuserTest;

// Test Case 1: TopK Detection Validation (Most Basic Test)
// Purpose: Verify TopK with symbolic K is detected by dynamic transform
// Input: Fusion with TopK(maybe_symbolic=true)
// Expected: DynamicTransformInitialInfo.isDynamic() == true
TEST_F(TopKDynamicTest, DynamicTransformDetection) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create symbolic input tensor
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // Create symbolic K parameter
  auto k = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(k);

  // Create TopK operation with symbolic K
  auto topk_result = topk(tv0, k, 0);

  fusion.addOutput(topk_result.values);
  fusion.addOutput(topk_result.indices);

  // Test detection
  auto initial_info = DynamicTransform::getInitialInfo(&fusion);

  // Verify TopK is detected as dynamic operation
  EXPECT_TRUE(initial_info.isDynamic())
      << "Fusion with symbolic TopK should be detected as dynamic";

  // Verify TopK tensors are tracked
  EXPECT_EQ(initial_info.getDynamicTopKTensorViews().size(), 1)
      << "Should track 1 TopK operation";

  // Verify the tracked tensor is the values output
  auto tracked_tv = initial_info.getDynamicTopKTensorViews().at(0);
  EXPECT_EQ(tracked_tv, topk_result.values)
      << "Should track TopK values tensor";

  // Verify TopK dimension has symbolic IterType
  auto logical_domain = topk_result.values->getLogicalDomain();
  EXPECT_TRUE(logical_domain[0]->isSymbolic())
      << "TopK dimension should have Symbolic IterType initially";
}

// Test Case 2: TopK K=1 Broadcast Concretization
// Purpose: Verify K=1 creates Broadcast IterType after concretization
// Input: TopK with symbolic K, runtime K=1
// Expected: TopK dimension becomes IterType::Broadcast
TEST_F(TopKDynamicTest, K1BroadcastConcretization) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create symbolic input tensor [4, 8, 16]
  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  // Create symbolic K parameter
  auto k = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(k);

  // Create TopK operation on last dimension with symbolic K
  auto topk_result = topk(tv0, k);

  fusion.addOutput(topk_result.values);
  fusion.addOutput(topk_result.indices);

  // Verify initial symbolic state
  auto initial_info = DynamicTransform::getInitialInfo(&fusion);

  EXPECT_TRUE(initial_info.isDynamic())
      << "TopK with symbolic K should be detected as dynamic";

  EXPECT_EQ(initial_info.getDynamicTopKTensorViews().size(), 1)
      << "Should detect 1 TopK operation with symbolic K";

  // Test concretization with K=1
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8, 16}, options);

  KernelArgumentHolder args({t0, 1}); // K=1 for broadcast test
  auto expr_eval = executor_utils::bindInputs(args, &fusion);
  DynamicTransformConcretizationInfo conc_info(&initial_info, &expr_eval);

  EXPECT_EQ(conc_info.getTopKIterTypes().size(), 1)
      << "Should analyze 1 TopK operation";

  EXPECT_EQ(conc_info.getTopKIterTypes().at(0).second, IterType::Broadcast)
      << "K=1 should result in Broadcast IterType";

  // Test concretization
  DynamicTransform::concretizeFusion(&fusion, &conc_info);

  // Verify results
  auto values_logical =
      fusion.outputs()[0]->as<TensorView>()->getLogicalDomain();
  auto indices_logical =
      fusion.outputs()[1]->as<TensorView>()->getLogicalDomain();

  EXPECT_EQ(values_logical.back()->getIterType(), IterType::Broadcast)
      << "Values tensor TopK dimension should be Broadcast after "
         "concretization";
  EXPECT_EQ(indices_logical.back()->getIterType(), IterType::Broadcast)
      << "Indices tensor TopK dimension should be Broadcast after "
         "concretization";
}

// Test Case 3: TopK K>1 Iteration Concretization
// Purpose: Verify K>1 creates Iteration IterType after concretization
// Input: TopK with symbolic K, runtime K=5
// Expected: TopK dimension becomes IterType::Iteration
TEST_F(TopKDynamicTest, KGreaterThan1IterationConcretization) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create symbolic input tensor [4, 8, 16]
  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  // Create symbolic K parameter
  auto k = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(k);

  // Create TopK operation on last dimension with symbolic K
  auto topk_result = topk(tv0, k, -1);

  fusion.addOutput(topk_result.values);
  fusion.addOutput(topk_result.indices);

  // Verify initial symbolic state
  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  EXPECT_TRUE(initial_info.isDynamic());
  EXPECT_EQ(initial_info.getDynamicTopKTensorViews().size(), 1);

  // Test concretization with K=5
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8, 16}, options);

  KernelArgumentHolder args({t0, 5}); // K=5 for iteration test
  auto expr_eval = executor_utils::bindInputs(args, &fusion);
  DynamicTransformConcretizationInfo conc_info(&initial_info, &expr_eval);

  // Verify analysis determines Iteration for K>1
  EXPECT_EQ(conc_info.getTopKIterTypes().size(), 1);
  EXPECT_EQ(conc_info.getTopKIterTypes().at(0).second, IterType::Iteration)
      << "K>1 should result in Iteration IterType";

  // Test concretization
  DynamicTransform::concretizeFusion(&fusion, &conc_info);

  // Verify results
  auto values_logical =
      fusion.outputs()[0]->as<TensorView>()->getLogicalDomain();
  auto indices_logical =
      fusion.outputs()[1]->as<TensorView>()->getLogicalDomain();

  EXPECT_EQ(values_logical.back()->getIterType(), IterType::Iteration)
      << "Values tensor TopK dimension should be Iteration after "
         "concretization";
  EXPECT_EQ(indices_logical.back()->getIterType(), IterType::Iteration)
      << "Indices tensor TopK dimension should be Iteration after "
         "concretization";
}

// Test Case 4: Multiple TopK Operations
// Purpose: Verify multiple TopK operations in same fusion
// Input: Fusion with multiple TopK operations, different K values
// Expected: Each TopK concretized independently and correctly
TEST_F(TopKDynamicTest, MultipleTopKOperations) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create symbolic input tensor [8, 12, 16]
  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  // Create two symbolic K parameters
  auto k1 = IrBuilder::create<Val>(DataType::Int);
  auto k2 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(k1);
  fusion.addInput(k2);

  // Create first TopK operation on dimension 0
  auto topk1_result = topk(tv0, k1, 0);

  // Create second TopK operation on dimension 2 of first result
  auto topk2_result = topk(topk1_result.values, k2, 2);

  fusion.addOutput(topk2_result.values);
  fusion.addOutput(topk2_result.indices);

  // Verify detection finds both TopK operations
  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  EXPECT_TRUE(initial_info.isDynamic());
  EXPECT_EQ(initial_info.getDynamicTopKTensorViews().size(), 2)
      << "Should detect 2 TopK operations";

  // Test concretization with K1=1 (Broadcast), K2=4 (Iteration)
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({8, 12, 16}, options);

  KernelArgumentHolder args({t0, 1, 4}); // K1=1, K2=4
  auto expr_eval = executor_utils::bindInputs(args, &fusion);
  DynamicTransformConcretizationInfo conc_info(&initial_info, &expr_eval);

  // Verify analysis handles both operations correctly
  EXPECT_EQ(conc_info.getTopKIterTypes().size(), 2);

  // First TopK: K1=1 should be Broadcast
  // Second TopK: K2=4 should be Iteration
  auto topk_types = conc_info.getTopKIterTypes();
  std::sort(
      topk_types.begin(),
      topk_types.end()); // Sort by index for predictable order

  EXPECT_EQ(topk_types[0].second, IterType::Broadcast)
      << "First TopK (K=1) should be Broadcast";
  EXPECT_EQ(topk_types[1].second, IterType::Iteration)
      << "Second TopK (K=4) should be Iteration";

  // Test concretization
  DynamicTransform::concretizeFusion(&fusion, &conc_info);

  // Verify results
  auto values_logical =
      fusion.outputs()[0]->as<TensorView>()->getLogicalDomain();
  auto indices_logical =
      fusion.outputs()[1]->as<TensorView>()->getLogicalDomain();

  // Final output should have:
  // - Dimension 0: Broadcast (from first TopK with K=1)
  // - Dimension 2: Iteration (from second TopK with K=4)
  EXPECT_EQ(values_logical[0]->getIterType(), IterType::Broadcast)
      << "First TopK dimension should be Broadcast";
  EXPECT_EQ(values_logical[2]->getIterType(), IterType::Iteration)
      << "Second TopK dimension should be Iteration";
}

// Test Case 5: Integration with Existing Dynamic Operations
// Purpose: Verify TopK works alongside other dynamic operations
// Input: Fusion with TopK + Reshape operations
// Expected: All operations concretized correctly without interference
TEST_F(TopKDynamicTest, TopKThenReshape) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create symbolic input tensor
  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  // Create symbolic parameters
  auto k = IrBuilder::create<Val>(DataType::Int);
  auto new_size = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(k);
  fusion.addInput(new_size);

  // Apply TopK first
  auto topk_result = topk(tv0, k, 1);

  // Apply resize to another dimension
  auto reshaped =
      reshape(topk_result.values, {new_size, IrBuilder::create<Val>(-1)});

  fusion.addOutput(reshaped);
  fusion.addOutput(topk_result.indices);

  // Verify both dynamic operations are detected
  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  EXPECT_TRUE(initial_info.isDynamic());

  // Should detect TopK operation
  EXPECT_EQ(initial_info.getDynamicTopKTensorViews().size(), 1)
      << "Should detect TopK operation";

  // Should also detect reshape operation
  EXPECT_EQ(initial_info.getDynamicReshapedTensorViews().size(), 1)
      << "Should detect reshape operation";

  // Test concretization
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({6, 10, 8}, options);

  KernelArgumentHolder args({t0, 3, 12}); // K=3, new_size=12
  auto expr_eval = executor_utils::bindInputs(args, &fusion);
  DynamicTransformConcretizationInfo conc_info(&initial_info, &expr_eval);

  // Verify both operations are analyzed
  EXPECT_EQ(conc_info.getTopKIterTypes().size(), 1);
  EXPECT_EQ(conc_info.getReshapeTransforms().size(), 1);

  // TopK with K=3 should be Iteration
  EXPECT_EQ(conc_info.getTopKIterTypes().at(0).second, IterType::Iteration);

  // Test concretization
  DynamicTransform::concretizeFusion(&fusion, &conc_info);

  // Verify results
  auto values_logical =
      fusion.outputs()[0]->as<TensorView>()->getLogicalDomain();
  auto indices_logical =
      fusion.outputs()[1]->as<TensorView>()->getLogicalDomain();

  // Dimension 1 should be Iteration from TopK
  EXPECT_EQ(values_logical.at(1)->getIterType(), IterType::Iteration)
      << "TopK dimension should be Iteration";
  EXPECT_EQ(indices_logical.at(1)->getIterType(), IterType::Iteration)
      << "TopK dimension should be Iteration";
}

// Test Case 6: Integration with Existing Dynamic Operations
// Purpose: Verify TopK works alongside other dynamic operations
// Input: Fusion with Reshape + TopK operations
// Expected: All operations concretized correctly without interference
TEST_F(TopKDynamicTest, ReshapeThenTopK) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create symbolic input tensor
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // Create symbolic parameters
  auto new_size_outer = IrBuilder::create<Val>(DataType::Int);
  auto k = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(new_size_outer);
  fusion.addInput(k);

  auto reshaped = reshape(tv0, {new_size_outer, IrBuilder::create<Val>(-1)});

  auto topk_result = topk(reshaped, k);

  fusion.addOutput(topk_result.indices);

  // Verify both dynamic operations are detected
  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  EXPECT_TRUE(initial_info.isDynamic());

  // Should detect TopK operation
  EXPECT_EQ(initial_info.getDynamicTopKTensorViews().size(), 1)
      << "Should detect TopK operation";

  // Should also detect reshape operation
  EXPECT_EQ(initial_info.getDynamicReshapedTensorViews().size(), 1)
      << "Should detect reshape operation";

  // Test concretization
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);

  KernelArgumentHolder args({t0, 4, 1}); // new_size_outer=4, K=1
  auto expr_eval = executor_utils::bindInputs(args, &fusion);
  DynamicTransformConcretizationInfo conc_info(&initial_info, &expr_eval);

  // Verify both operations are analyzed
  EXPECT_EQ(conc_info.getTopKIterTypes().size(), 1);
  EXPECT_EQ(conc_info.getReshapeTransforms().size(), 1);

  // TopK with K=1 should be Broadcast
  EXPECT_EQ(conc_info.getTopKIterTypes().at(0).second, IterType::Broadcast);

  // Test concretization
  DynamicTransform::concretizeFusion(&fusion, &conc_info);

  // Verify results
  auto values_logical =
      fusion.outputs()[0]->as<TensorView>()->getLogicalDomain();

  // Dimension 1 should be Broadcast from TopK
  EXPECT_EQ(values_logical.at(1)->getIterType(), IterType::Broadcast)
      << "TopK dimension should be Iteration";
}

// TopK producer has a symbolic iter domain, but the TopKOp itself
// should not be symbolic as its K parameter is static
TEST_F(TopKDynamicTest, DynamicReshapeThenStaticTopK) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create symbolic input tensor
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // Create symbolic parameters
  auto new_size_outer = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(new_size_outer);

  auto reshaped = reshape(tv0, {new_size_outer, IrBuilder::create<Val>(-1)});

  // Apply static TopK
  auto topk_result = topk(reshaped, IrBuilder::create<Val>(3));

  fusion.addOutput(topk_result.indices);

  fusion.printMath();

  // Verify dynamic operations are detected
  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  EXPECT_TRUE(initial_info.isDynamic());

  // TopK should be static
  EXPECT_TRUE(initial_info.getDynamicTopKTensorViews().empty())
      << "Should not have dynamic TopK operation";

  // Should detect reshape operation
  EXPECT_EQ(initial_info.getDynamicReshapedTensorViews().size(), 1)
      << "Should detect reshape operation";
}

} // namespace nvfuser

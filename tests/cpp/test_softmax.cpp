// SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <ir/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

// A simple example demonstrating basic tensor addition using FusionExecutorCache
TEST_F(NVFuserTest, BasicTensorAddition_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensors
  auto tv0 = makeSymbolicTensor(2); // 2D tensor
  auto tv1 = makeSymbolicTensor(2); // 2D tensor
  
  // Register as inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  
  // Perform addition
  auto tv2 = add(tv0, tv1);
  
  // Register as output
  fusion.addOutput(tv2);
}

// Test demonstrating transform propagation and its benefits
TEST_F(NVFuserTest, TransformPropagation_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create symbolic input tensor
  auto tv0 = makeSymbolicTensor(2); // 2D tensor (i0, i1)
  fusion.addInput(tv0);
  
  // Create a computational pipeline: tv0 -> tv1 -> tv2 -> tv3
  // This creates a chain of dependent tensors to demonstrate propagation
  auto tv1 = mul(tv0, IrBuilder::create<Val>(2.0)); // tv1 = tv0 * 2
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0)); // tv2 = tv1 + 1
  auto tv3 = mul(tv2, IrBuilder::create<Val>(3.0)); // tv3 = tv2 * 3
  
  fusion.addOutput(tv3);

  // STAGE 1: Before any transformations
  std::cout << "\n==== STAGE 1: BEFORE ANY TRANSFORMATIONS ====\n";
  std::cout << "\n--- Fusion IR ---\n";
  fusion.print();
  std::cout << "\n--- Mathematical Operations ---\n";
  fusion.printMath();
  std::cout << "\n--- CUDA Kernel ---\n";
  fusion.printKernel();
  
  // Schedule following the recommended merge→split→parallelize pattern for pointwise ops
  
  // 1. Merge all dimensions to create a single flattened dimension
  tv3->merge(0);  // Merge dimensions 0 and 1, creating a single dimension (i0*i1)
  
  // 2. Split for parallelism (working inside-out)
  // First split for vectorization (innermost - constant size 4)
  tv3->split(0, 4);  // Now [(i0*i1)/4, 4]
  
  // Then split for thread parallelism
  tv3->split(0, 128);  // Now [(i0*i1)/(4*128), 128, 4]
  
  // 3. Parallelize
  tv3->axis(0)->parallelize(ParallelType::BIDx);    // Outer dimension -> blocks
  tv3->axis(1)->parallelize(ParallelType::TIDx);    // Middle dimension -> threads
  tv3->axis(2)->parallelize(ParallelType::Vectorize); // Inner dimension -> vectorize
  
  // STAGE 2: After scheduling tv3 but before propagation
  std::cout << "\n==== STAGE 2: AFTER SCHEDULING TV3 BUT BEFORE PROPAGATION ====\n";
  std::cout << "\n--- Fusion IR ---\n";
  fusion.print();
  std::cout << "\n--- Tensor Transforms ---\n";
  fusion.printTransforms();
  std::cout << "\n--- CUDA Kernel ---\n";
  fusion.printKernel();
  
  // Print descriptions of tensor states
  std::cout << "\n--- Description of Tensor States Before Propagation ---\n";
  std::cout << "TV3 (output tensor): Transformed with merge, splits, and parallelization\n";
  std::cout << "TV2 (intermediate tensor): No transformations applied yet\n";
  std::cout << "TV1 (intermediate tensor): No transformations applied yet\n";
  
  // 4. Propagate the transformations to producer tensors (tv2, tv1)
  TransformPropagator propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);
  
  // STAGE 3: After transform propagation but before parallelizeAllLike
  std::cout << "\n==== STAGE 3: AFTER TRANSFORM PROPAGATION BUT BEFORE PARALLELIZEALLLIKE ====\n";
  std::cout << "\n--- Fusion IR ---\n";
  fusion.print();
  std::cout << "\n--- Tensor Transforms ---\n";
  fusion.printTransforms();
  std::cout << "\n--- CUDA Kernel ---\n";
  fusion.printKernel();
  
  // Print descriptions of tensor states
  std::cout << "\n--- Description of Tensor States After Propagation ---\n";
  std::cout << "TV3 (output tensor): Has both transformations and parallelization\n";
  std::cout << "TV2 (intermediate tensor): Has transformations but no parallelization yet\n";
  std::cout << "TV1 (intermediate tensor): Has transformations but no parallelization yet\n";
  
  // 5. Propagate parallelization from consumer to producers
  scheduler_utils::parallelizeAllLike(tv3);
  
  // STAGE 4: After parallelizeAllLike
  std::cout << "\n==== STAGE 4: AFTER PARALLELIZEALLLIKE ====\n";
  std::cout << "\n--- Fusion IR ---\n";
  fusion.print();
  std::cout << "\n--- Tensor Transforms ---\n";
  fusion.printTransforms();
  std::cout << "\n--- CUDA Kernel ---\n";
  fusion.printKernel();
  std::cout << std::endl;
}

// Test exploring cacheAfter with vectorization
TEST_F(NVFuserTest, CacheAfterVectorization_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 1. Define input tensor and initial operations
  auto tv0_global = TensorViewBuilder().ndims(2).dtype(DataType::Float).contiguity(true).build();
  fusion.addInput(tv0_global);

  auto tv_temp_output = mul(tv0_global, IrBuilder::create<Val>(2.0));
  auto tv1_output = add(tv_temp_output, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1_output);

  std::cout << "\n==== STEP A: After cacheAfter(), before tv0_local scheduling ====\n";
  auto tv0_local = tv0_global->cacheAfter();
  std::cout << "\n--- Math (Step A) ---\n";
  fusion.printMath();
  std::cout << "\n--- CUDA Kernel (Step A) ---\n";
  fusion.printKernel();
  std::cout << std::endl;

  std::cout << "\n==== STEP B: tv0_local set to Global Memory ====\n";
  tv0_local->setMemoryType(MemoryType::Global);
  std::cout << "\n--- Math (Step B) ---\n";
  fusion.printMath();
  std::cout << "\n--- CUDA Kernel (Step B) ---\n";
  fusion.printKernel();
  std::cout << std::endl;

  std::cout << "\n==== STEP C: tv0_local merge(0) and split(0, 4) ====\n";
  tv0_local->merge(0);
  tv0_local->split(0, 4);
  std::cout << "\n--- Math (Step C) ---\n";
  fusion.printMath();
  std::cout << "\n--- CUDA Kernel (Step C) ---\n";
  fusion.printKernel();
  std::cout << std::endl;

  std::cout << "\n==== STEP D: tv0_local vectorize axis(1) ====\n";
  tv0_local->axis(1)->parallelize(ParallelType::Vectorize);
  std::cout << "\n--- Math (Step D) ---\n";
  fusion.printMath();
  std::cout << "\n--- CUDA Kernel (Step D) ---\n";
  fusion.printKernel();
  std::cout << std::endl;

  std::cout << "\n==== STEP E: tv0_local parallelize TIDx on axis(0) ====\n";
  tv0_local->axis(0)->parallelize(ParallelType::TIDx);
  std::cout << "\n--- Math (Step E) ---\n";
  fusion.printMath();
  std::cout << "\n--- CUDA Kernel (Step E) ---\n";
  fusion.printKernel();
  std::cout << std::endl;

  std::cout << "\n==== STEP F: parallelizeAllLike(tv0_local, consumers) ====\n";
  scheduler_utils::parallelizeAllLike(tv0_local, {tv_temp_output, tv1_output});
  std::cout << "\n--- Math (Step F) ---\n";
  fusion.printMath();
  std::cout << "\n--- CUDA Kernel (Step F) ---\n";
  fusion.printKernel();
  std::cout << std::endl;
}

} // namespace nvfuser 
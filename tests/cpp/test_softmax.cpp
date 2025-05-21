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

// Test exploring cacheAfter (for input) and cacheBefore (for output) with vectorization
TEST_F(NVFuserTest, CacheAfterBeforeVectorization_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int vec_factor = 4;
  const int tid_x_threads = 128;

  auto tv0_global = TensorViewBuilder().ndims(2).dtype(DataType::Float).contiguity(true).build();
  fusion.addInput(tv0_global);

  auto tv_temp_math = mul(tv0_global, IrBuilder::create<Val>(2.0));
  auto tv1_output_global = add(tv_temp_math, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1_output_global);

  // --- Part 1: Caching Setup ---
  auto tv0_local = tv0_global->cacheAfter(); // Cache for input
  // tv_temp_math will now use tv0_local

  tv1_output_global->cacheBefore(); // Cache for output
  // tv1_output_local is now defined by add(tv_temp_math, 1.0)
  // tv1_output_global is now defined by Set(tv1_output_local)

  // --- Part 2: Base Scheduling (on tv0_local) ---
  // Schedule tv0_local. This structure and BIDx/TIDx parallelization will be propagated.
  tv0_local->merge(0);
  tv0_local->split(0, vec_factor);      // Innermost axis for vectorization, e.g., axis 2 after next split
  tv0_local->split(0, tid_x_threads); // Middle axis for TIDx, e.g., axis 1
                                     // Outermost axis for BIDx, e.g., axis 0

  tv0_local->axis(0)->parallelize(ParallelType::BIDx);
  tv0_local->axis(1)->parallelize(ParallelType::TIDx);
  // tv0_local->axis(2) is not yet vectorized.

  // --- Part 3: Single Propagation Pass (from tv0_local) ---
  // Propagate structure and BIDx/TIDx from tv0_local to all its consumers.
  // tv_temp_math and tv1_output_local should inherit the 3-axis domain
  // with a scalar innermost dimension of size vec_factor.
  // tv1_output_global, being a Set from tv1_output_local, should also get this structure.
  TransformPropagator propagator(tv0_local);
  MaxLogicalDomainInfoSpanningTree(tv0_local).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv0_local);

  // --- Part 4: Targeted Vectorization ---
  // Vectorize the LOAD into tv0_local
  tv0_local->axis(2)->parallelize(ParallelType::Vectorize);

  // Vectorize the STORE from tv1_output_local to tv1_output_global.
  // This is done by vectorizing the corresponding axis on tv1_output_global itself.
  // tv1_output_local (defined by math) keeps its scalar innermost domain for its definition.
  tv1_output_global->axis(2)->parallelize(ParallelType::Vectorize);

  // --- Part 5: Print final Math IR and CUDA Kernel ---
  std::cout << "\n==== FINAL KERNEL (CacheAfter+CacheBefore, Vectorized Load/Store) ====\n";
  std::cout << "\n--- Final Math IR ---\n";
  fusion.printMath();
  std::cout << "\n--- Final CUDA Kernel ---\n";
  fusion.printKernel();
  std::cout << std::endl;

  // TODO: Add executeAndValidate if needed for this specific exploration phase,
  // or rely on kernel printout for now. Consider input sizes for validation.
}

// Case study for documenting Softmax implementation and optimization
TEST_F(NVFuserTest, SoftmaxImplementationCaseStudy_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Define a 2D input tensor (e.g., Batch x Features)
  // For Softmax, reduction is typically done along the last dimension (Features).
  auto tv_input = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(tv_input);

  // Reduction axis: last dimension (axis 1 for a 2D tensor)
  const int reduction_axis = 1;
  std::vector<int> axes_to_reduce = {reduction_axis};
  bool keep_dim = true; // Keep dimension for broadcasting compatibility

  // 1. Find max value for numerical stability: max_val = max(input, reduction_axis)
  auto tv_max_val = max(tv_input, axes_to_reduce, keep_dim);

  // 2. Subtract max_val from input: exp_input_intermediate = input - broadcast(max_val)
  //    Need to broadcast tv_max_val back to the shape of tv_input
  //    Broadcast axes for max_val (which is [Batch, 1]) to become [Batch, Features]
  //    will be along axis 1.
  std::vector<bool> broadcast_dims_for_max(tv_input->nDims(), false);
  broadcast_dims_for_max[reduction_axis] = true;
  auto tv_max_val_broadcasted = broadcast(tv_max_val, broadcast_dims_for_max);
  auto tv_exp_input_intermediate = sub(tv_input, tv_max_val_broadcasted);

  // 3. Compute exponent: exp_input = exp(exp_input_intermediate)
  auto tv_exp_input = exp(tv_exp_input_intermediate);

  // 4. Sum exponents: sum_exp = sum(exp_input, reduction_axis)
  auto tv_sum_exp = sum(tv_exp_input, axes_to_reduce, keep_dim);

  // 5. Divide by sum: output = exp_input / broadcast(sum_exp)
  //    Need to broadcast tv_sum_exp back to the shape of tv_exp_input
  std::vector<bool> broadcast_dims_for_sum(tv_input->nDims(), false);
  broadcast_dims_for_sum[reduction_axis] = true;
  auto tv_sum_exp_broadcasted = broadcast(tv_sum_exp, broadcast_dims_for_sum);
  auto tv_output = div(tv_exp_input, tv_sum_exp_broadcasted);

  fusion.addOutput(tv_output);

  // --- Initial (Naive) Implementation Printouts ---
  std::cout << "\n==== SOFTMAX CASE STUDY: NAIVE IMPLEMENTATION ====\n";
  std::cout << "\n--- Initial Math IR (Softmax) ---\n";
  fusion.printMath();
  std::cout << "\n--- Initial CUDA Kernel (Softmax) ---\n";
  fusion.printKernel();
  std::cout << std::endl;

  // TODO: Add validation logic for this softmax implementation
  // at::Tensor at_input = at::randn({BATCH_SIZE, FEATURE_SIZE}, tensor_options);
  // FusionExecutor fe;
  // fe.compileFusion(&fusion, {at_input});
  // auto outputs = fe.runFusion({at_input});
  // auto at_softmax_output = at::softmax(at_input, reduction_axis);
  // testValidate(&fusion, outputs, {at_input}, {at_softmax_output}, __LINE__, __FILE__);
}

} // namespace nvfuser 
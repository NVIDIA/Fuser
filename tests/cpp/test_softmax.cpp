// SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/tools/inlining.h>
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

// Very Simple nvFuser Example: Adding two 1D tensors
TEST_F(NVFuserTest, VerySimpleExample_CUDA) {
  // 1. Create a Fusion object: This holds our computation.
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  // 2. FusionGuard: Activates the fusion for operations in this scope.
  FusionGuard fg(fusion);

  // 3. Define Inputs: Two 1D symbolic tensors.
  //    "Symbolic" means their size isn't fixed yet.
  auto tv0 = makeSymbolicTensor(1); // 1D tensor
  auto tv1 = makeSymbolicTensor(1); // 1D tensor
  
  //    Register them as inputs to the fusion.
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  
  // 4. Define Operation: Add the two tensors.
  auto tv2 = add(tv0, tv1);
  
  // 5. Define Output: Register the result tensor as an output.
  //    This is crucial! nvFuser only compiles what leads to an output.
  fusion->addOutput(tv2);
  
  // 6. Print (Optional): See what nvFuser generates (unscheduled).
  std::cout << "\n==== VERY SIMPLE EXAMPLE ====\n";
  std::cout << "--- Mathematical Operations ---\n";
  fusion->printMath(); // Shows: T2 = T0 + T1
  std::cout << "--- Generated CUDA Kernel (Unscheduled) ---\n";
  fusion->printKernel(); // Shows a basic, unoptimized CUDA kernel
  std::cout << std::endl;

  // 7. Execute the fusion using FusionExecutorCache
  
  // Create FusionExecutorCache to compile and run the fusion
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  
  // Create input tensors with actual data
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::ones({10}, options);  // 10 elements, all 1's
  at::Tensor input2 = at::ones({10}, options) * 2;  // 10 elements, all 2's
  
  // Setup input arguments
  std::vector<c10::IValue> inputs = {input1, input2};
  
  // Run the fusion
  auto outputs = executor_cache.runFusionWithInputs(inputs);
  
  // Print the compiled CUDA code
  std::cout << "\n==== EXECUTED FUSION WITH INPUTS ====\n";
  std::cout << "Input 1: " << input1 << std::endl;
  std::cout << "Input 2: " << input2 << std::endl;
  std::cout << "Output: " << outputs[0] << std::endl;
  
  std::cout << "\n==== COMPILED CUDA KERNEL ====\n";
  std::cout << executor_cache.getMostRecentCode() << std::endl;
}

// Test exploring inlining with reduction operations
TEST_F(NVFuserTest, InliningReductionExample_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a 2D input tensor
  auto tv0 = makeSymbolicTensor(2);  // Input 2D tensor [I, J]
  fusion.addInput(tv0);
  
  // Step 1: Simple unary operation - multiply by 2
  auto tv1 = mul(tv0, IrBuilder::create<Val>(2.0));  // tv1[i,j] = tv0[i,j] * 2
  
  // Step 2: Reduction operation - sum along the first dimension (axis 0)
  auto tv2 = sum(tv1, {0});  // tv2[j] = sum(tv1[:,j])
  
  // Step 3: Another unary operation - add 1.0 to the result
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));  // tv3[j] = tv2[j] + 1
  
  fusion.addOutput(tv3);

  // STAGE 1: Before any inlining
  std::cout << "\n==== STAGE 1: BEFORE INLINING ====\n";
  std::cout << "\n--- Fusion IR ---\n";
  fusion.print();
  std::cout << "\n--- CUDA Kernel (default schedule) ---\n";
  fusion.printKernel();
  
  // STAGE 2: Apply inlineMost to tv1 (producer of tv2 reduction)
  std::cout << "\n==== STAGE 2: INLINE TV1 USING INLINEMOST ====\n";
  std::cout << "This inlines the computation of tv1 at the innermost loop position of tv2\n";
  
  std::vector<TensorView*> to_inline_1 = {tv1};
  inlineMost(to_inline_1);
  
  std::cout << "\n--- Fusion IR After Inlining tv1 ---\n";
  fusion.print();
  std::cout << "\n--- CUDA Kernel ---\n";
  fusion.printKernel();
  
  // STAGE 3: Apply inlineMost to tv2 (producer of tv3)
  // Note: Typically, we don't inline reduction outputs, but this demonstrates
  // the behavior for educational purposes
  std::cout << "\n==== STAGE 3: INLINE TV2 (REDUCTION OUTPUT) USING INLINEMOST ====\n";
  std::cout << "This shows what happens when we inline a reduction output using inlineMost\n";
  
  std::vector<TensorView*> to_inline_2 = {tv2};
  inlineMost(to_inline_2);
  
  std::cout << "\n--- Fusion IR After Inlining tv2 ---\n";
  fusion.print();
  std::cout << "\n--- CUDA Kernel ---\n";
  fusion.printKernel();
  
  // Key learnings about inlining with reductions
  std::cout << "\n==== KEY LEARNINGS: INLINING WITH REDUCTIONS ====\n";
  std::cout << "1. REDUCTION INPUTS: Using inlineMost for reduction inputs (tv1)\n";
  std::cout << "   - Computes values at the innermost loop position of the reduction\n";
  std::cout << "   - May enable efficient memory access patterns for reduction operations\n\n";
  
  std::cout << "2. REDUCTION OUTPUTS: Usually NOT recommended to inline reduction outputs (tv2)\n";
  std::cout << "   - When using inlineMost, may recompute the entire reduction for each use\n";
  std::cout << "   - With axis 0 reduction, this affects how memory is accessed\n\n";
  
  std::cout << "3. TRADE-OFFS WITH DIMENSION CHOICE:\n";
  std::cout << "   - Reducing over dimension 0 vs dimension 1 changes memory access patterns\n";
  std::cout << "   - Inlining strategy depends on which dimensions are reduced\n";
}

// Test exploring RFactor with reduction operations 
TEST_F(NVFuserTest, RFactorExample_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a 2D input tensor
  auto tv0 = makeSymbolicTensor(2);  // 2D input tensor [I, J]
  fusion.addInput(tv0);
  
  // Perform some computation before reduction
  auto tv1 = mul(tv0, IrBuilder::create<Val>(2.0));  // tv1[i,j] = tv0[i,j] * 2
  
  // Create a reduction operation along the second dimension (dim 1)
  auto tv2 = sum(tv1, {1});  // tv2[i] = sum(tv1[i,:])
  fusion.addOutput(tv2);

  // STAGE 1: Visualize the reduction before any transformations
  std::cout << "\n==== STAGE 1: ORIGINAL REDUCTION OPERATION ====\n";
  std::cout << "\n--- Fusion IR ---\n";
  fusion.print();
  std::cout << "\n--- Mathematical Operations ---\n";
  fusion.printMath();
  std::cout << "\n--- Default CUDA Kernel ---\n";
  fusion.printKernel();
  
  // STAGE 2: Split the reduction dimension to prepare for RFactor
  std::cout << "\n==== STAGE 2: SPLIT REDUCTION DIMENSION ====\n";
  
  // Split the reduction dimension into outer and inner parts
  // For example, split dim 1 into chunks of size 128
  tv2->split(1, 128);
  // Now tv2 has shape [I, R1o, R1i{128}], where:
  // - I is original dimension 0
  // - R1o is the outer part of the split reduction dimension
  // - R1i is the inner part with size 128
  
  std::cout << "After splitting tv2 along the reduction dimension:\n";
  std::cout << tv2->toString() << std::endl;
  
  // STAGE 3: Apply RFactor to the outer reduction dimension
  std::cout << "\n==== STAGE 3: APPLY RFACTOR ====\n";
  
  // RFactor transforms the outer reduction dimension into an iteration dimension
  // This creates an intermediate tensor that performs partial reductions
  auto tv3 = tv2->rFactor({1});
  
  std::cout << "After applying rFactor:\n";
  std::cout << "Intermediate tensor (tv3): " << tv3->toString() << std::endl;
  std::cout << "Output tensor (tv2): " << tv2->toString() << std::endl;
  
  // STAGE 4: Schedule the computation with appropriate computeAt and parallelization
  std::cout << "\n==== STAGE 4: SCHEDULE WITH COMPUTEAT AND PARALLELIZE ====\n";
  
  // Set up computeAt relationships for proper fusion
  tv0->computeAt(tv3, 2);  // Compute tv0 at the outer reduction level of tv3
  tv3->computeAt(tv2, 1);  // Compute tv3 at the I0 level of tv2
  
  // Parallelize the computation
  tv2->axis(0)->parallelize(ParallelType::BIDx);  // Parallelize outer dim across blocks
  tv2->axis(1)->parallelize(ParallelType::TIDx);  // Parallelize inner reduction across threads
  tv3->axis(2)->parallelize(ParallelType::TIDx);  // Parallelize the inner iteration dim across threads
  
  // Inline tv1 into tv3 for more efficient computation
  tv1->inlineAt(-1);  // Inline at innermost position
  
  std::cout << "\n--- Scheduled Fusion IR ---\n";
  fusion.print();
  std::cout << "\n--- Optimized CUDA Kernel ---\n";
  fusion.printKernel();
  
  // STAGE 5: Execute and validate the RFactored kernel
  std::cout << "\n==== STAGE 5: EXECUTE AND VALIDATE ====\n";
  
  // Create input tensor with test data
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({64, 128}, options);
  
  // Execute the kernel
  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});
  
  // Verify result against PyTorch reference
  auto reference = (input * 2).sum({1});
  bool result_correct = at::allclose(outputs[0], reference, 1e-5, 1e-5);
  
  std::cout << "Result validation: " << (result_correct ? "CORRECT" : "INCORRECT") << std::endl;
  
  // STAGE 6: Analyze RFactor benefits
  std::cout << "\n==== STAGE 6: RFACTOR BENEFITS ANALYSIS ====\n";
  std::cout << "1. Parallelization Benefits:\n";
  std::cout << "   - Without RFactor: Each thread block would compute one output element\n";
  std::cout << "   - With RFactor: Multiple thread blocks collaborate on large reductions\n";
  std::cout << "   - Threads share work by computing partial results at the inner dimension\n\n";
  
  std::cout << "2. Memory Pattern Benefits:\n";
  std::cout << "   - Splitting allows more efficient memory access patterns\n";
  std::cout << "   - The RFactored tensor (tv3) serves as an intermediate storage\n";
  std::cout << "   - Partial results are stored in a more cache-friendly way\n\n";
  
  std::cout << "3. Flexibility Benefits:\n";
  std::cout << "   - RFactor allows fine-grained control over how reductions are computed\n";
  std::cout << "   - Can adapt to different target hardware (e.g., more/fewer threads)\n";
  std::cout << "   - Enables multi-level reductions for large tensors\n";
}

} // namespace nvfuser 
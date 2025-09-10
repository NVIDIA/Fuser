// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <id_model/id_model.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <type.h>

namespace nvfuser {

class Tutorial : public NVFuserTest {
 protected:
  static void SetUpTestSuite() {
    verbose_ = getNvFuserEnv("TUTORIAL_VERBOSE");
  }

 protected:
  static bool verbose_;
};

bool Tutorial::verbose_ = false;

TEST_F(Tutorial, Memcpy) {
  // First, we define a fusion. A common pattern is:
  // - Declare a Fusion, which works as a container of expressions
  // - Setup inputs. Utility routines such as makeSymbolicTensor
  //   can be used to create tensors, which can be then registered as
  //   fusion inputs with Fusion::addInput
  // - Define operations with the registered inputs. For supported
  //   operations, see csrc/ops/all_ops.h.
  // - Most of operations that take tensors as inputs produce tensors
  //   as outputs, which can then be used as inputs to another
  //   operations
  // - Final outputs should be set as fusion outputs with
  //   Fusion::addOutput
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a 2D tensor of type float. It's "symbolic" as we do not
  // assume any specific shape except for that it's 2D.
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // Just create a copy
  TensorView* tv1 = set(tv0);
  fusion.addOutput(tv1);

  // End of fusion definition

  if (verbose_) {
    // Here's some common ways to inspect the fusion. These are not
    // necessary for running the fusion but should provide helpful
    // information for understanding how fusions are transformed.

    // Print a concise representation of the fusion exprssions
    fusion.printMath();

    // Generate and print a CUDA kernel. Notice that at this point the
    // genereated code is just a sequential kernel as we have not
    // scheduled the fusion yet, but it should be a valid CUDA kernel
    fusion.printKernel();
  }

  // Next, try running the fusion. First, we need to set up a sample
  // input tensor. Here, we create a 32x32 tensor initialized with
  // random float values.
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32, 32}, options);
  {
    // Next, lower the fusion to Kernel, generate CUDA kernel source and then
    // compile it with nvrtc. All of them are done by KernelExecutor
    KernelExecutor ke;
    ke.compile(&fusion, {t0});

    // KernelExecutor now has a compiled kernel, which can be executed as:
    auto outputs = ke.run({t0});
    // Note that this run is done using just one thread, which will be
    // corrected below.

    // To validate the output, we can just assert that the output is
    // equal to the input as this is just a copy fusion. More commonly,
    // though, testValidate is used to validate outputs while
    // automatically adjusting thresholds of valid deviations
    ASSERT_TRUE(outputs[0].as<at::Tensor>().equal(t0));

    // Next, instead of just running the fusion as is, we manually
    // schedule it so that it runs in parallel. In this case, we only
    // have one expression, i.e., the set expression, so we just need to
    // schedule tv1.

    // tv1 is a 2D tensor. Let its domain be [i0, i1]. What we are going
    // to do is to transform this 2D domain to the multi-dimensional
    // CUDA parallelism, i.e., a grid consisting of multiple thread
    // blocks, each of which consisting of multiple threads. A common
    // transformation pattern is to merge all of each axis to get a
    // flattened domain, and then split the domain to factor out axes
    // that are parallelized by threads and thread blocks.

    // For example, the current domain of tv1 looks like [i0, i1]. We
    // can merge the two axes by:
    tv1->merge(0, 1);
    // This creates a single axis that merges i0 and i1. Its extent is a
    // multiplication of the extents of i0 and i1, so we commonly
    // represent it as [i0*i1]. It can be also examined with:
    if (verbose_) {
      std::cout << tv1->toString() << std::endl;
    }

    // Next, we factor out a subdomain for threads in each thread
    // block.
    tv1->split(0, 256);
    // In this case, the flattened domain is now 2D domain with an inner
    // domain of extent 256 and an outer domain of extent i0*i1/256, so
    // the tensor should now look like [i0*i1/256, 256]. Note that in
    // reality we do ceil division as i0*i1 may not be divisible by
    // 256.
    if (verbose_) {
      std::cout << tv1->toString() << std::endl;
    }

    // Now that we have two domains, we can parallelize each domain
    // using IterDomain::parallelize(ParallelType). Specifically, to
    // parallelize the inner domain with threads, we can do:
    tv1->axis(1)->parallelize(ParallelType::TIDx);
    // Similarly, to paralllize the outer domain with thread blocks:
    tv1->axis(0)->parallelize(ParallelType::BIDx);
    // This way, the inner and outer axes are divided by blockDim.x
    // threads and gridDim.x blocks, respectively. Each element in each
    // axis is computed by one thread or one block, so this means that
    // the size of each thread block and a grid must match the size of
    // each domain. Specifically, blockDim.x and gridDim.x must be 256
    // and i0*i1/256, respectively.

    // Now that the fusion is parallelized, it can be examined again
    if (verbose_) {
      fusion.printMath();
      // Notice that the axes of tv1 are now printed with blockIdx.x and
      // threadIdx.x, which shows they are parallelized by the
      // respective parallel types

      // The CUDA kernel should look very differently as there should be
      // no for-loops
      fusion.printKernel();
    }
  }

  // Since the fusion is modified, we need to recompile it.
  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // This time, the kernel is launched with multiple threads and
  // thread blocks. Note that the launch configurations, i.e., the
  // thread block and grid shapes, are autoatically inferred from the
  // given inputs. To see how many threads are used, run this test
  // with NVFUSER_DUMP=launch_param
  auto outputs = ke.run({t0});

  ASSERT_TRUE(outputs[0].as<at::Tensor>().equal(t0));
}

TEST_F(Tutorial, Reduction) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a 2D tensor
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // Reduce the second dimension
  auto tv1 = sum(tv0, {1});
  fusion.addOutput(tv1);

  // At this point, nothing is parallelized. The reduction is done by
  // a single thread sequentially.

  if (verbose_) {
    fusion.printMath();
    fusion.printKernel();
  }

  // Block-parallel reduciton
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  if (verbose_) {
    fusion.printMath();
    fusion.printKernel();
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 1024}, options);
  at::Tensor ref = t0.sum({1});

  {
    KernelExecutor ke;
    ke.compile(&fusion);
    auto outputs = ke.run({t0});
    testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
  }

  // Next, use the same fusion but parallelize the reduction with
  // thread blocks
  tv1->axis(1)->parallelize(ParallelType::BIDx);

  if (verbose_) {
    fusion.printMath();
    fusion.printKernel();
  }

  {
    KernelExecutor ke;
    ke.compile(&fusion);
    auto outputs = ke.run({t0});
    testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
  }

  // We can also parallelize the first axis as well. For example,
  // here's how threadIdx.x is used for the reduction and threadIdx.y
  // is used for the outer non-reduction domain
  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  if (verbose_) {
    fusion.printMath();
    fusion.printKernel();
  }

  {
    KernelExecutor ke;
    ke.compile(&fusion);
    // Running this fusion, however, should fail as it would require
    // thread blocks of shape 1024x10, i.e., the same shape as the
    // input tensor, which is too large in CUDA.
    //
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_ANY_THROW(ke.run({t0}));

    // Try again with a smaller input. This should launch a kernel
    // with thread blocks of shape 32x10
    at::Tensor t1 = at::randn({10, 32}, options);
    auto outputs = ke.run({t1});
    testValidate(&fusion, outputs, {t0}, {t1.sum({1})}, __LINE__, __FILE__);
  }

  // We can of course mix BIDx and TIDx.
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  if (verbose_) {
    fusion.printMath();
    fusion.printKernel();
  }

  {
    KernelExecutor ke;
    ke.compile(&fusion);
    // The original input should not fail in this case. The kernel
    // will be launched with 10 thread blocks, each of which has 1024
    // threads. Try running this test with NVFUSER_DUMP=launch_param
    // to see the launch configuration of each kernel lauch
    auto outputs = ke.run({t0});
    testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
  }
}

TEST_F(Tutorial, ReductionRFactor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Just a very simple reduction of 1D tensor
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  if (verbose_) {
    fusion.printMath();
  }

  // Multiple scheduling examples will be created so each time a copy
  // of the fusion is used
  {
    Fusion fusion_copy = fusion;
    FusionGuard fg_copy(&fusion_copy);

    // Note that tv1 is not a valid pointer for fusion_copy
    auto tv1_copy = fusion_copy.outputs().at(0)->as<TensorView>();

    // A common pattern of reductions in CUDA involves multiple steps of
    // reductions, where the first step is a per-thread local reduction,
    // followed by a block reduction of the per-thread partial results,
    // and also potentially followed by a grid reduction of the
    // per-block partial results. Here's an example with a two-step
    // reduciton:
    //
    // // Step 1: Per-thread reduction
    // float partial_result = 0;
    // for (int i = threadIdx.x; i += blockDim.x; i < N) {
    //   partial_result += input[i];
    // }
    //
    // // Step 2: Accumulation within each thread block
    // __shared__ float shared_buf[blockDim.x];
    // shared_buf[threadIdx.x] = partial_result;
    // __syncthreads();
    // float final_result = 0;
    // // Accumulation of the partila result in a naive sequntial way.
    // if (threadIdx.x == 0) {
    //   for (int i = 0; i < blockDim.x; ++i) {
    //     final_result += shared_buf[i];
    //   }
    // }
    //
    // To reproduce the multi-step reduction pattern in nvFuser, a
    // fusion transformaiton called reduction rfactor is used. The basic
    // idea is to split a reduction domain such that each of the output
    // domains of the split is separately reduced. For example, tv1 can
    // be transformed from a 2D tensor to a 3D tensor as follows:
    //
    // tv0: [i0]
    // tv1: [r1]
    tv1_copy->split(0, 1024);
    // tv1: [r1/1024, r1024]
    //
    // Both of the two inner domains are reduction domains, and we first
    // want to reduce the second domain, i.e., r1/1024, by each thread
    // independently, and then reduce the other reduction domain by a
    // block reduction. This can be done as follows:
    TensorView* tv2 = tv1_copy->rFactor({0});

    // The fusion math should now look like:
    //
    // tv0: root = logical = [i{i0}]
    // tv2 = reduction(tv0): root = [r{i0}], logical = [r{i0/1024}, i{1024}]
    // tv1 = reduction(tv2): root = logical = [r{1024}]
    if (verbose_) {
      fusion_copy.print();
    }
    // Notice that the reduction operation is now split into two
    // operations, where the first one takes care of the first domain, and the
    // second one finishes up the remaining domain. The final values of
    // tv1 is not alterned, but just its computation is changed. (More
    // strictly, though, since floating-point addition is not
    // associative, the final result will not be exactly the same due to
    // rounding errors)

    // To realize the parallelization as we sketched above, we can
    // use TIDx for both of tv1 and tv2 as follows:
    tv1_copy->axis(0)->parallelize(ParallelType::TIDx);
    tv2->axis(1)->parallelize(ParallelType::TIDx);

    // At this point, tv2 is a TIDx-parallelized operation of multiple
    // independent reductions. There will be 1024 threads, each of which
    // reduces the first axis of size r1/1024.
    // tv1 is also parallelized by TIDx, but unlike tv2 the reduction
    // domain is parallelized, so it becomes a block-reduction
    // operation.
    if (verbose_) {
      fusion_copy.printMath();
      fusion_copy.printKernel();
    }

    // Let's run the scheduled fusion
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({10000}, options);
    at::Tensor ref = t0.sum({0});

    KernelExecutor ke;
    ke.compile(&fusion_copy);

    // Since the size of the input is 10000, which is split by a
    // factor of 1024, the first per-thread reduction is done for
    // ceilDiv(10000, 1024) = 10 elements.
    auto outputs = ke.run({t0});
    testValidate(&fusion_copy, outputs, {t0}, {ref}, __LINE__, __FILE__);
  }

  // We can further increase the parallelism by splitting the
  // reduction domain into three
  {
    Fusion fusion_copy = fusion;
    FusionGuard fg_copy(&fusion_copy);

    auto tv1_copy = fusion_copy.outputs().at(0)->as<TensorView>();

    // First, split for TIDx of 1024 threads
    tv1_copy->split(0, 1024);
    // Next, split for BIDx of 100 thread blocks
    tv1_copy->split(0, 100);
    // tv1: [r0/1024/100, r100, r1024]

    // Factoring out per-thread reduction
    auto tv2 = tv1_copy->rFactor({1});
    // tv2: [i0/1024/100, r100, i1024]
    // tv1: [r0/1024/100, r1024]

    // Factoring out block reduction
    auto tv3 = tv1_copy->rFactor({1});
    // tv2: [i0/1024/100, r100, i1024]
    // tv3: [i0/1024/100, r1024]
    // tv1: [r0/1024/100]

    // Parallelize each operation as follows
    // tv2: [bidx(i0/1024/100), r100, tidx(i1024)]
    // tv3: [bidx(i0/1024/100), tidx(r1024)]
    // tv1: [bidx(r0/1024/100)]
    tv2->axis(0)->parallelize(ParallelType::BIDx);
    tv3->axis(0)->parallelize(ParallelType::BIDx);
    tv1_copy->axis(0)->parallelize(ParallelType::BIDx);
    tv2->axis(2)->parallelize(ParallelType::TIDx);
    tv3->axis(1)->parallelize(ParallelType::TIDx);
    // Note that this could be also done more easily using
    // scheduler_utils::parallelizeAllLike.

    if (verbose_) {
      fusion_copy.printMath();
      fusion_copy.printKernel();
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    // Notice we use a larger input. The same size as before can be
    // used, but some threads will be idle.
    at::Tensor t0 = at::randn({10000000}, options);
    at::Tensor ref = t0.sum({0});

    KernelExecutor ke;
    ke.compile(&fusion_copy);

    auto outputs = ke.run({t0});
    testValidate(&fusion_copy, outputs, {t0}, {ref}, __LINE__, __FILE__);
  }
}

TEST_F(Tutorial, Reshape) {
  {
    // Simple reshape example
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(2);
    fusion.addInput(tv0);

    // Shape of tv0 is assumed to be [4, 8], which is then reshaped to [32]
    auto tv1 = reshape(tv0, {4, 8}, {32});
    fusion.addOutput(tv1);

    if (verbose_) {
      // Notice that tv1 has root and logical domains. The root domain
      // should consist of two IterDomains, whreas the logical domain
      // consists of a single IterDomain that is an output of a merge
      // operation of the two root IterDomains
      fusion.print();
    }

    // Check if the tv1 domains are generated as expected
    ASSERT_TRUE(tv1->hasRoot());
    ASSERT_EQ(tv1->getLogicalDomain().size(), 1);
    ASSERT_TRUE(tv1->getLogicalDomain().at(0)->definition()->isA<Merge>());
    Merge* tv1_merge = tv1->getLogicalDomain().at(0)->definition()->as<Merge>();
    ASSERT_EQ(tv1_merge->inner(), tv1->getRootDomain().at(1));
    ASSERT_EQ(tv1_merge->outer(), tv1->getRootDomain().at(0));
  }

  {
    // Reshape example with broadcast domains
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Create a 3D tensor with a broadcast domain
    auto tv0 = makeConcreteTensor({1, -1, -1});
    fusion.addInput(tv0);

    // tv0 is first squeezed and then reshaped and unsqueezed
    auto tv1 = reshape(tv0, {1, 2, 3}, {3, 2, 1});
    fusion.addOutput(tv1);

    if (verbose_) {
      fusion.print();
    }

    // The fusion should look like:
    //
    // tv1 = unsqueeze(reshape(squeeze(tv0)));
    ASSERT_TRUE(tv1->definition()->isA<BroadcastOp>());
    auto reshape_output = tv1->definition()->input(0)->as<TensorView>();
    ASSERT_TRUE(reshape_output->definition()->isA<ReshapeOp>());
    auto squeeze_output =
        reshape_output->definition()->input(0)->as<TensorView>();
    ASSERT_TRUE(squeeze_output->definition()->isA<SqueezeOp>());

    ASSERT_TRUE(reshape_output->hasRoot());
    ASSERT_EQ(reshape_output->getLogicalDomain().size(), 2);
    ASSERT_TRUE(
        reshape_output->getLogicalDomain().at(0)->definition()->isA<Split>());
    auto reshape_output_split =
        reshape_output->getLogicalDomain().at(0)->definition()->as<Split>();
    ASSERT_EQ(
        reshape_output_split->outer(),
        reshape_output->getLogicalDomain().at(0));
    ASSERT_EQ(
        reshape_output_split->inner(),
        reshape_output->getLogicalDomain().at(1));
    ASSERT_TRUE(reshape_output_split->in()->definition()->isA<Merge>());
    auto reshape_output_merge =
        reshape_output_split->in()->definition()->as<Merge>();
    ASSERT_EQ(
        reshape_output_merge->outer(), reshape_output->getRootDomain().at(0));
    ASSERT_EQ(
        reshape_output_merge->inner(), reshape_output->getRootDomain().at(1));

    // So far, the fusion has transformations as part of its
    // definition. It can be further extended with scheduling transformations.
    reshape_output->merge(0, 1);
    reshape_output->split(0, 128);

    ASSERT_TRUE(
        reshape_output->getLoopDomain().at(0)->definition()->isA<Split>());
    ASSERT_EQ(
        reshape_output->getLoopDomain()
            .at(0)
            ->definition()
            ->as<Split>()
            ->inner(),
        reshape_output->getLoopDomain().at(1));
    ASSERT_TRUE(reshape_output->getLoopDomain()
                    .at(0)
                    ->definition()
                    ->as<Split>()
                    ->in()
                    ->definition()
                    ->isA<Merge>());
    ASSERT_EQ(
        reshape_output->getLoopDomain()
            .at(0)
            ->definition()
            ->as<Split>()
            ->in()
            ->definition()
            ->as<Merge>()
            ->outer(),
        reshape_output->getLogicalDomain().at(0));
    ASSERT_EQ(
        reshape_output->getLoopDomain()
            .at(0)
            ->definition()
            ->as<Split>()
            ->in()
            ->definition()
            ->as<Merge>()
            ->inner(),
        reshape_output->getLogicalDomain().at(1));

    // Here's how we propagate the transformations of reshape_output
    // to all other tensors in the fusion
    TransformPropagatorWithCheck propagator(reshape_output);
    MaxLogicalDomainInfoSpanningTree(reshape_output).traverse(&propagator);

    // Now, all tensors, including those before the reshape op, should
    // be transformed to 2D tensors with an inner domain of extent
    // 128.
    if (verbose_) {
      fusion.print();
    }

    // Notice that all transformations of the reshape tensor,
    // including both the reshape and scheduling transformations, are
    // propagated. For example, squeeze_output should have the merge and split
    // for the reshape, followed by another merge and split for
    // scheduling. Specifically:
    //
    // Root domain: [b0, i1, i2]
    // merge(1, 2) -> [b0, i1*i2]
    // outer split(1, 3) -> [b0, 3, i1*i2/3]
    // merge(1, 2) -> [b0, 3*i1*i2/3]
    // split(1, 128) -> [b0, 3*i1*i2/3/128, 128]
    ASSERT_TRUE(
        squeeze_output->getLoopDomain().at(0)->definition()->isA<Split>());
    auto squeeze_output_second_split =
        squeeze_output->getLoopDomain().at(0)->definition()->as<Split>();
    ASSERT_EQ(
        squeeze_output_second_split->outer(),
        squeeze_output->getLoopDomain().at(0));
    ASSERT_EQ(
        squeeze_output_second_split->inner(),
        squeeze_output->getLoopDomain().at(1));

    ASSERT_TRUE(squeeze_output_second_split->in()->definition()->isA<Merge>());
    auto squeeze_output_second_merge =
        squeeze_output_second_split->in()->definition()->as<Merge>();

    ASSERT_TRUE(
        squeeze_output_second_merge->outer()->definition()->isA<Split>());
    auto squeeze_output_first_split =
        squeeze_output_second_merge->outer()->definition()->as<Split>();
    ASSERT_EQ(
        squeeze_output_first_split->outer(),
        squeeze_output_second_merge->outer());
    ASSERT_EQ(
        squeeze_output_first_split->inner(),
        squeeze_output_second_merge->inner());

    ASSERT_TRUE(squeeze_output_first_split->in()->definition()->isA<Merge>());
    auto squeeze_output_first_merge =
        squeeze_output_first_split->in()->definition()->as<Merge>();
    ASSERT_EQ(
        squeeze_output_first_merge->outer(),
        squeeze_output->getLogicalDomain().at(0));
    ASSERT_EQ(
        squeeze_output_first_merge->inner(),
        squeeze_output->getLogicalDomain().at(1));

    // Note that all the transformations of squeeze_output are scheduling
    // transformations, thus it should not have a root domain
    ASSERT_FALSE(squeeze_output->hasRoot());
  }
}

// Demonstration of using IdModel for analyzing equivalence of reshape ops
TEST_F(Tutorial, IdModelReshapeAnalysis) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  // Use the static reshape to avoid reshape concretization.
  //
  // While the reshape operations are equivalent, we don't know if the
  // two inputs are the same as there's no op allowing inference of
  // equivalence (e.g., tv0 + tv1)
  auto tv2 = reshape(tv0, {10, 20}, {20, 10});
  auto tv3 = reshape(tv1, {10, 20}, {20, 10});

  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  IdModel id_model(&fusion);
  id_model.buildExactGraph();
  ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

  // As mentioned above, we don't know any relationship between tv0
  // and tv1, so they should not be mapped.
  for (const auto i : arange(tv0->getLogicalDomain().size())) {
    ASSERT_FALSE(exact_graph.disjointValSets().strictAreMapped(
        tv0->getLogicalDomain().at(i), tv1->getLogicalDomain().at(i)));
  }

  // Thus, the outputs of the reshape ops are not mapped either
  for (const auto i : arange(tv2->nDims())) {
    ASSERT_FALSE(exact_graph.disjointValSets().strictAreMapped(
        tv2->axis(i), tv3->axis(i)));
  }

  // Now, suppose we can say the inputs are exactly mapped. We
  // can manually add mappings:
  for (const auto i : arange(tv0->getLogicalDomain().size())) {
    exact_graph.mapVals(
        tv0->getLogicalDomain().at(i), tv1->getLogicalDomain().at(i));
  }

  // Now, tv2 and tv3 should be fully mapped, including their root,
  // intermediate and loop domains.

  // Check the root domains.
  for (const auto i : arange(tv2->getRootDomain().size())) {
    ASSERT_TRUE(exact_graph.disjointValSets().strictAreMapped(
        tv2->getRootDomain().at(i), tv3->getRootDomain().at(i)));
  }

  // The reshape consists of a merge and split. The output of the
  // merge should be mapped as well
  ASSERT_TRUE(exact_graph.disjointValSets().strictAreMapped(
      tv2->getRootDomain().at(0)->uses().at(0)->as<Merge>()->out(),
      tv3->getRootDomain().at(0)->uses().at(0)->as<Merge>()->out()));

  // The next operation is split. Its outputs, which are the loop
  // domains, should be mapped too.
  for (const auto i : arange(tv2->nDims())) {
    ASSERT_TRUE(exact_graph.disjointValSets().strictAreMapped(
        tv2->axis(i), tv3->axis(i)));
  }
}

TEST_F(Tutorial, BasicTMA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  // This tutorial uses copy kernels to demonstrate how to schedule TMA. Please
  // note that this is not a guide on how to use TMA to achieve SOL. Instead, it
  // is a demonstration on the degree of freedoms we have in a TMA schedule and
  // how a schedule is translated into generated code in the kernel. I also want
  // the readers to focus on the schedule of TMA. How the other part of the
  // kernel is scheduled is not important here, and indeed, I just randomly
  // picked one schedule for it without any meaning. For an example about TMA
  // load, please focus on the schedule of the shared memory tensor. For an
  // example about TMA store, please focus on the allocation domain of the
  // shared memory tensor and the fusion output.

  CompileParams index32bit{DataType::Int32, 255, false};

  {
    // Example 1:
    // Similar to how we generally schedule pointwise fusion, in this example,
    // we treat the fusion as 1D and uses 1D TMA to load data to shared memory.
    // We use one TMA instruction to load the entire CTA tile.
    // CTA tile size = TMA tile size = 256
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto input = makeContigTensor(3);
    fusion.addInput(input);
    auto output = set(input);
    fusion.addOutput(output);

    TensorView* smem_cache =
        input->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
    smem_cache->setMemoryType(MemoryType::Shared);

    // For TMA load, both the shared memory layout and the loop nest and
    // parallelization of TMA are specified by the consumer: smem_cache

    // Step 1: define TMA domain
    // Because we want to treat the entire tensor as 1D, we define the TMA
    // domain as [I0*I1*I2]
    smem_cache->merge(0);
    smem_cache->merge(0);
    // Note that the TMA domain only exist in people's mind, there is no need to
    // set anything here.

    // Step 2: define box
    smem_cache->split(0, 256);
    // [I0*I1*I2/256, 256]
    // partitioned IterDomain: I0*I1*I2
    // coordinate IterDomain: I0*I1*I2/256
    // box IterDomain: 256

    // Step 3: define tile
    // We use dense tile here, so tile == box. Nothing to do here.

    // Step 4: schedule the shared memory tensor
    // By default, the allocation domain is the logical domain, which is already
    // in good shape for this case.

    // Step 5: schedule the consumer tensor
    smem_cache->axis(0)->parallelize(ParallelType::BIDx);
    smem_cache->axis(1)->parallelize(ParallelType::Bulk);
    // [BIDx, Bulk]

    // Schedule the smem->gmem part
    output->merge(0);
    output->merge(0);
    output->split(0, 256);
    output->axis(0)->parallelize(ParallelType::BIDx);
    output->axis(1)->parallelize(ParallelType::TIDx);

    if (verbose_) {
      fusion.print();
      fusion.printKernel();
      // TMA will be generated like:
      // Note that the coordinate is in number of items, smem address is in
      // bytes
      //
      // if (threadIdx.x == 0) {
      //   Hopper::cpAsyncBulkTensorTileG2S(
      //       coordinate = {256 * blockIdx.x},
      //       smem_addr = toSmem(T2));
      // }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    std::vector<int64_t> shape(3, 300);
    auto t = at::randn(shape, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t}, {}, index32bit);
    auto outputs = ke.run({t});
    ASSERT_TRUE(at::equal(t, outputs[0].as<at::Tensor>()));
  }

  {
    // Example 2:
    // Similar to example 1, we treat the fusion as 1D and uses 1D TMA to load
    // data to shared memory. But this time, instead of using 1 TMA instruction
    // to load the entire CTA tile, we use 4 TMA instructions. We use a for loop
    // to launch these 4 instructions
    // CTA tile size = 4 * TMA tile size = 1024
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto input = makeContigTensor(3);
    fusion.addInput(input);
    auto output = set(input);
    fusion.addOutput(output);

    TensorView* smem_cache =
        input->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
    smem_cache->setMemoryType(MemoryType::Shared);

    // For TMA load, both the shared memory layout and the loop nest and
    // parallelization of TMA are specified by the consumer: smem_cache

    // Step 1: define TMA domain
    // Because we want to treat the entire tensor as 1D, we define the TMA
    // domain as [I0*I1*I2]
    smem_cache->merge(0);
    smem_cache->merge(0);
    // Note that the TMA domain only exist in people's mind, there is no need to
    // set anything here.

    // Step 2: define box
    smem_cache->split(0, 256);
    // [I0*I1*I2/256, 256]
    // partitioned IterDomain: I0*I1*I2
    // coordinate IterDomain: I0*I1*I2/256
    // box IterDomain: 256

    // Step 3: define tile
    // We use dense tile here, so tile == box. Nothing to do here.

    // Step 4: schedule the shared memory tensor
    // By default, the allocation domain is the logical domain, which is already
    // in good shape for this case.

    // Step 5: schedule the consumer tensor
    smem_cache->split(0, 4);
    // [I0*I1*I2/256/4, 4, 256]
    smem_cache->axis(0)->parallelize(ParallelType::BIDx);
    smem_cache->axis(2)->parallelize(ParallelType::Bulk);
    // [BIDx, Serial, Bulk]

    // Schedule the smem->gmem part
    output->merge(0);
    output->merge(0);
    output->split(0, 256);
    output->split(0, 4);
    output->axis(0)->parallelize(ParallelType::BIDx);
    output->axis(2)->parallelize(ParallelType::TIDx);

    if (verbose_) {
      fusion.print();
      fusion.printKernel();
      // TMA will be generated like:
      // Note that the coordinate is in number of items, smem address is in
      // bytes
      //
      // for (nvfuser_index_t i8 = 0; i8 < 4; ++i8) {
      //   if (threadIdx.x == 0) {
      //     Hopper::cpAsyncBulkTensorTileG2S(
      //         coordinate = {1024 * blockIdx.x + 256 * i8},
      //         smem_addr = (toSmem(T2) + 1024 * i8));
      //   }
      // }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    std::vector<int64_t> shape(3, 300);
    auto t = at::randn(shape, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t}, {}, index32bit);
    auto outputs = ke.run({t});
    ASSERT_TRUE(at::equal(t, outputs[0].as<at::Tensor>()));
  }

  {
    // Example 3:
    // Similar to example 2, we treat the fusion as 1D and uses 1D TMA to load
    // data to shared memory, and we use 4 TMA instructions to load the entire
    // CTA tile. However, instead of using a for loop to launch these 4
    // instructions, we parallelize these 4 instructions to TIDx.
    // CTA tile size = 4 * TMA tile size = 1024
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto input = makeContigTensor(3);
    fusion.addInput(input);
    auto output = set(input);
    fusion.addOutput(output);

    TensorView* smem_cache =
        input->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
    smem_cache->setMemoryType(MemoryType::Shared);

    // For TMA load, both the shared memory layout and the loop nest and
    // parallelization of TMA are specified by the consumer: smem_cache

    // Step 1: define TMA domain
    // Because we want to treat the entire tensor as 1D, we define the TMA
    // domain as [I0*I1*I2]
    smem_cache->merge(0);
    smem_cache->merge(0);
    // Note that the TMA domain only exist in people's mind, there is no need to
    // set anything here.

    // Step 2: define box
    smem_cache->split(0, 256);
    // [I0*I1*I2/256, 256]
    // partitioned IterDomain: I0*I1*I2
    // coordinate IterDomain: I0*I1*I2/256
    // box IterDomain: 256

    // Step 3: define tile
    // We use dense tile here, so tile == box. Nothing to do here.

    // Step 4: schedule the shared memory tensor
    // By default, the allocation domain is the logical domain, which is already
    // in good shape for this case.

    // Step 5: schedule the consumer tensor
    smem_cache->split(0, 4);
    // [I0*I1*I2/256/4, 4, 256]
    smem_cache->axis(0)->parallelize(ParallelType::BIDx);
    smem_cache->axis(1)->parallelize(ParallelType::TIDx);
    smem_cache->axis(2)->parallelize(ParallelType::Bulk);
    // [BIDx, TIDx, Bulk]

    // Schedule the smem->gmem part
    output->merge(0);
    output->merge(0);
    output->split(0, 256);
    output->split(0, 4);
    output->axis(0)->parallelize(ParallelType::BIDx);
    output->axis(2)->parallelize(ParallelType::TIDx);

    if (verbose_) {
      fusion.print();
      fusion.printKernel();
      // TMA will be generated like:
      // Note that the coordinate is in number of items, smem address is in
      // bytes
      //
      // if (threadIdx.x < 4) {
      //   Hopper::cpAsyncBulkTensorTileG2S(
      //       coordinate = {1024 * blockIdx.x + 256 * threadIdx.x},
      //       smem_addr = (toSmem(T2) + 1024 * threadIdx.x));
      // }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    std::vector<int64_t> shape(3, 300);
    auto t = at::randn(shape, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t}, {}, index32bit);
    auto outputs = ke.run({t});
    ASSERT_TRUE(at::equal(t, outputs[0].as<at::Tensor>()));
  }

  {
    // Example 4: Similar to example 3, except that we are using TMA for store
    // instead of load.
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto input = makeContigTensor(3);
    fusion.addInput(input);
    auto output = set(input);
    fusion.addOutput(output);

    TensorView* smem_cache =
        output->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);
    smem_cache->setMemoryType(MemoryType::Shared);

    // For TMA store, the loop nest and parallelization is specified in the
    // consumer `output`, and the shared memory layout is specified in the
    // allocation dimain of `smem_cache`.

    // Step 1: define TMA domain
    // Because we want to treat the entire tensor as 1D, we define the TMA
    // domain as [I0*I1*I2]
    output->merge(0);
    output->merge(0);
    // Note that the TMA domain only exist in people's mind, there is no need to
    // set anything here.

    // Step 2: define box
    output->split(0, 256);
    // [I0*I1*I2/256, 256]
    // partitioned IterDomain: I0*I1*I2
    // coordinate IterDomain: I0*I1*I2/256
    // box IterDomain: 256

    // Step 3: define tile
    // We use dense tile here, so tile == box. Nothing to do here.

    // Step 4: schedule the shared memory tensor
    // By default, the allocation domain is the logical domain, which is already
    // in good shape for this case.

    // Step 5: schedule the consumer tensor
    output->split(0, 4);
    // [I0*I1*I2/256/4, 4, 256]
    output->axis(0)->parallelize(ParallelType::BIDx);
    output->axis(1)->parallelize(ParallelType::TIDx);
    output->axis(2)->parallelize(ParallelType::Bulk);
    // [BIDx, TIDx, Bulk]

    // Schedule the gmem->smem part
    smem_cache->merge(0);
    smem_cache->merge(0);
    smem_cache->split(0, 256);
    smem_cache->split(0, 4);
    smem_cache->axis(0)->parallelize(ParallelType::BIDx);
    smem_cache->axis(2)->parallelize(ParallelType::TIDx);

    if (verbose_) {
      fusion.print();
      fusion.printKernel();
      // TMA will be generated like:
      // Note that the coordinate is in number of items, smem address is in
      // bytes
      //
      // if (threadIdx.x < 4) {
      //   Hopper::cpAsyncBulkTensorTileS2G(
      //       coordinate = {1024 * blockIdx.x + 256 * threadIdx.x},
      //       smem_addr = (toSmem(T2) + 1024 * threadIdx.x));
      // }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    std::vector<int64_t> shape(3, 300);
    auto t = at::randn(shape, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t}, {}, index32bit);
    auto outputs = ke.run({t});
    ASSERT_TRUE(at::equal(t, outputs[0].as<at::Tensor>()));
  }

  {
    // Example 5: Still the same copy kernel of 3D tensor, but this time, we
    // want to do tiling on the inner two dimensions. The first dimension is
    // treated as a "batch" dimension. We use CTA tile (64, 64), and TMA tile
    // (32, 32), so we need 4 TMA instructions to load the entire CTA tile.
    // We want to use two threads, and each thread issue two TMA instructions.
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto input = makeContigTensor(3);
    fusion.addInput(input);
    auto output = set(input);
    fusion.addOutput(output);

    TensorView* smem_cache =
        input->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
    smem_cache->setMemoryType(MemoryType::Shared);

    // For TMA load, both the shared memory layout and the loop nest and
    // parallelization of TMA are specified by the consumer: smem_cache

    // Step 1: define TMA domain
    // For this case, we want to treat all three dimensions separately.
    // TMA domain: [I0, I1, I2]
    // Note that the TMA domain only exist in people's mind, there is no need to
    // set anything here.

    // Step 2: define box
    smem_cache->split(2, 32);
    smem_cache->split(1, 32);
    // [I0, I1/32, 32, I2/32', 32']
    // Box dimensions defined by partitioning: I1 and I2
    //   partitioned IterDomain: I1, I2
    //   coordinate IterDomain: I1/32, I2/32'
    //   box IterDomain: 32, 32'
    // Box dimension defined by compositing: I0
    //   coordinate IterDomain: I0
    //   box IterDomain: no box IterDomain, so implicit size 1

    // Step 3: define tile
    // We use dense tile here, so tile == box. Nothing to do here.

    // Step 4: schedule the shared memory tensor
    // By default, the allocation domain is the logical domain. The default
    // value does not work for this case, because this way, tile will not be
    // contiguous in shared memory.
    // [I0, I1/32, 32, I2/32', 32']
    smem_cache->split(3, 2);
    smem_cache->split(1, 2);
    // [I0, I1/32/2, 2, 32, I2/32'/2', 2', 32']
    smem_cache->reorder({{3, -2}, {2, -4}});
    // [I0, I1/32/2, I2/32'/2', 2, 2', 32, 32']
    smem_cache->setAllocationDomain(smem_cache->getLoopDomain(), true);

    // Step 5: schedule the consumer tensor
    // [I0, I1/32/2, I2/32'/2', 2, 2', 32, 32']
    smem_cache->axis(0)->parallelize(ParallelType::BIDx);
    smem_cache->axis(1)->parallelize(ParallelType::BIDy);
    smem_cache->axis(2)->parallelize(ParallelType::BIDz);
    smem_cache->axis(3)->parallelize(ParallelType::TIDx);
    smem_cache->axis(5)->parallelize(ParallelType::Bulk);
    smem_cache->axis(6)->parallelize(ParallelType::Bulk);
    // [BIDx, BIDy, BIDz, TIDx, Serial, Bulk, Bulk]

    // Schedule the smem->gmem part
    output->split(2, 32);
    output->split(1, 32);
    output->split(3, 2);
    output->split(1, 2);
    output->reorder({{3, -2}, {2, -4}});
    output->axis(0)->parallelize(ParallelType::BIDx);
    output->axis(1)->parallelize(ParallelType::BIDy);
    output->axis(2)->parallelize(ParallelType::BIDz);
    output->merge(3);
    output->axis(3)->parallelize(ParallelType::TIDx);

    if (verbose_) {
      fusion.print();
      fusion.printKernel();
      // TMA will be generated like:
      // Note that the coordinate is in number of items, smem address is in
      // bytes.Also note that coordinate is in column major, so inner dims
      // goes first
      //
      // for (nvfuser_index_t i13 = 0; i13 < 2; ++i13) {
      //   if (threadIdx.x < 2) {
      //     Hopper::cpAsyncBulkTensorTileG2S(
      //         coordinate =
      //             {64 * blockIdx.z + 32 * i13,
      //              64 * blockIdx.y + 32 * threadIdx.x,
      //              blockIdx.x},
      //         smem_addr = toSmem(T2) + 8192 * threadIdx.x + 4096 * i13);
      //   }
      // }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    std::vector<int64_t> shape(3, 300);
    auto t = at::randn(shape, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t}, {}, index32bit);
    auto outputs = ke.run({t});
    ASSERT_TRUE(at::equal(t, outputs[0].as<at::Tensor>()));
  }

  {
    // Example 6: Similar to example 5, but we are using TMA for store instead
    // of load.
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto input = makeContigTensor(3);
    fusion.addInput(input);
    auto output = set(input);
    fusion.addOutput(output);

    TensorView* smem_cache =
        output->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);
    smem_cache->setMemoryType(MemoryType::Shared);

    // For TMA store, the loop nest and parallelization is specified in the
    // consumer `output`, and the shared memory layout is specified in the
    // allocation dimain of `smem_cache`.

    // Step 1: define TMA domain
    // For this case, we want to treat all three dimensions separately.
    // TMA domain: [I0, I1, I2]
    // Note that the TMA domain only exist in people's mind, there is no need to
    // set anything here.

    // Step 2: define box
    output->split(2, 32);
    output->split(1, 32);
    // [I0, I1/32, 32, I2/32', 32']
    // Box dimensions defined by partitioning: I1 and I2
    //   partitioned IterDomain: I1, I2
    //   coordinate IterDomain: I1/32, I2/32'
    //   box IterDomain: 32, 32'
    // Box dimension defined by compositing: I0
    //   coordinate IterDomain: I0
    //   box IterDomain: no box IterDomain, so implicit size 1

    // Step 3: define tile
    // We use dense tile here, so tile == box. Nothing to do here.

    // Step 4: schedule the shared memory tensor
    // By default, the allocation domain is the logical domain. The default
    // value does not work for this case, because this way, tile will not be
    // contiguous in shared memory.
    // [I0, I1, I2]
    smem_cache->split(2, 32);
    smem_cache->split(1, 32);
    // [I0, I1/32, 32, I2/32', 32']
    smem_cache->split(3, 2);
    smem_cache->split(1, 2);
    // [I0, I1/32/2, 2, 32, I2/32'/2', 2', 32']
    smem_cache->reorder({{3, -2}, {2, -4}});
    // [I0, I1/32/2, I2/32'/2', 2, 2', 32, 32']
    smem_cache->setAllocationDomain(smem_cache->getLoopDomain(), true);

    // Step 5: schedule the consumer tensor.
    // Because we are not inlining anything in this example, we do not care
    // about the order of IterDomains.
    // [I0, I1/32, 32, I2/32', 32']
    output->split(3, 2);
    output->split(1, 2);
    // [I0, I1/32/2, 2, 32, I2/32'/2', 2', 32']
    output->axis(0)->parallelize(ParallelType::BIDx);
    output->axis(1)->parallelize(ParallelType::BIDy);
    output->axis(2)->parallelize(ParallelType::TIDx);
    output->axis(3)->parallelize(ParallelType::Bulk);
    output->axis(4)->parallelize(ParallelType::BIDz);
    output->axis(6)->parallelize(ParallelType::Bulk);
    // [BIDx, BIDy, TIDx, Bulk, BIDz, Serial, Bulk]

    // Schedule the gmem->smem part
    smem_cache->merge(-2);
    smem_cache->axis(0)->parallelize(ParallelType::BIDx);
    smem_cache->axis(1)->parallelize(ParallelType::BIDy);
    smem_cache->axis(2)->parallelize(ParallelType::BIDz);
    smem_cache->axis(-1)->parallelize(ParallelType::TIDx);

    if (verbose_) {
      fusion.print();
      fusion.printKernel();
      // TMA will be generated like:
      // Note that the coordinate is in number of items, smem address is in
      // bytes.Also note that coordinate is in column major, so inner dims
      // goes first
      //
      // for (nvfuser_index_t i19 = 0; i19 < 2; ++i19) {
      //   if (threadIdx.x < 2) {
      //     Hopper::cpAsyncBulkTensorTileS2G(
      //         coordinate =
      //             {64 * blockIdx.z + 32 * i19,
      //              64 * blockIdx.y + 32 * threadIdx.x,
      //              blockIdx.x},
      //         smem_addr = toSmem(T2) + 8192 * threadIdx.x + 4096 * i19);
      //   }
      // }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    std::vector<int64_t> shape(3, 300);
    auto t = at::randn(shape, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t}, {}, index32bit);
    auto outputs = ke.run({t});
    ASSERT_TRUE(at::equal(t, outputs[0].as<at::Tensor>()));
  }
}

TEST_F(Tutorial, VectorizeStorePointwiseTMA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  CompileParams index32bit{DataType::Int32, 255, false};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr at::ScalarType dtype = at::ScalarType::Float;

  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  auto tv1 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  // Create cache_tvs
  auto tv0a = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  auto tv1a = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  auto tv2b = tv2->cacheBefore();

  tv0a->setMemoryType(MemoryType::Shared);
  tv1a->setMemoryType(MemoryType::Shared);

  auto reference_tv = tv2;

  // Step 1: Create tma domain
  // Use the root domain as TMA domain
  //   root domain: [I0, I1]

  constexpr int64_t num_threads = 128;
  constexpr int64_t vectorization = 2;
  constexpr int64_t tma_tile = num_threads * vectorization;
  constexpr int64_t num_stages = 4;
  constexpr int64_t num_ctas_for_hopper = 132;

  // Step 2: Create Box
  // After TMA domain creation
  //         split: [I0, I3, 256]
  reference_tv->split(-1, tma_tile);
  //         split: [I2, 4, I3, 256]
  reference_tv->split(0, num_stages);

  // Step 3: Create Tile
  // Do nothing here because box == tile

  // Step 4: Schedule Shared Memory Tensor
  //         split: [I2, 4, I3, 128, 2]
  reference_tv->split(-1, vectorization);
  //         split: [I4, 132, 4, I3, 128, 2]
  reference_tv->split(0, num_ctas_for_hopper);
  //         reorder: [I4, 132, I3, 4, 128, 2]
  reference_tv->reorder({{3, 2}, {2, 3}});

  // Transform Operations between cache operations and output reference
  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree(reference_tv).traverse(&propagator);

  // Propagate common parallel dimensions
  reference_tv->axis(1)->parallelize(ParallelType::BIDx);
  scheduler_utils::parallelizeAllLike(reference_tv);

  tv2b->axis(-2)->parallelize(ParallelType::TIDx);

  // Vectorization for writing results to gmem
  reference_tv->axis(-3)->parallelize(ParallelType::Unroll);
  reference_tv->axis(-2)->parallelize(ParallelType::TIDx);
  reference_tv->axis(-1)->parallelize(ParallelType::Vectorize);

  // Apply bulk type to TMA tensors
  tv0a->axis(-1)->parallelize(ParallelType::Bulk);
  tv0a->axis(-2)->parallelize(ParallelType::Bulk);
  tv0a->axis(-3)->parallelize(ParallelType::Bulk);

  tv1a->axis(-1)->parallelize(ParallelType::Bulk);
  tv1a->axis(-2)->parallelize(ParallelType::Bulk);
  tv1a->axis(-3)->parallelize(ParallelType::Bulk);

  // ComputeAt
  inlineMost();

  if (verbose_) {
    fusion->printMath();
    fusion->printKernel();
  }

  constexpr int dim0 = 16384, dim1 = 16384;
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);
  at::Tensor at_tv1 = at::randn({dim0, dim1}, options);

  // Compile with KernelExecutor directly to avoid scheduling
  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0, at_tv1}, {}, index32bit);
  auto outputs = ke.run({at_tv0, at_tv1});

  auto at_output = at_tv0 + at_tv1;
  testValidate(
      fusion.get(), outputs, {at_tv0, at_tv1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(Tutorial, PointwiseBroadcastTMA) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  CompileParams index32bit{DataType::Int32, 255, false};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr at::ScalarType dtype = at::ScalarType::Float;

  auto tv0 = makeContigTensor(3, aten_to_data_type(dtype));
  auto tv1 = TensorViewBuilder()
                 .ndims(4)
                 .shape({-1, -1, -1, -1})
                 .contiguity({true, false, true, true})
                 .dtype(aten_to_data_type(dtype))
                 .build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv0, {true, false, false, false});
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  // Create cache_tvs
  auto tv0a = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  auto tv1a = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  auto tv3b = tv3->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);

  tv0a->setMemoryType(MemoryType::Shared);
  tv1a->setMemoryType(MemoryType::Shared);
  tv3b->setMemoryType(MemoryType::Shared);

  auto reference_tv = tv3;

  // Step 1: Create tma domain
  //   root domain: [I0, I1, I2, I3]
  //    TMA domain: [I0, I1, I4]
  reference_tv->merge(-2, -1);

  // Step 2: Define TMA Box
  //         split: [I0, I1, I5, 256]
  reference_tv->split(-1, 256);

  // Step 3: Define Tile
  // Do nothing here because tile == box.

  // Step 4: Schedule Shared Memory Tensor
  //         merge: [I10, I5, 256]
  reference_tv->merge(0, 1);
  //         split: [I10, I7, 4, 256]
  reference_tv->split(-2, 4);
  //         merge: [I11, 4, 256]
  reference_tv->merge(0, 1);

  // Transform Operations between cache operations and output reference
  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree(reference_tv).traverse(&propagator);

  // Define Parallelization Schema
  // Intermediate Tensors
  tv3b->axis(0)->parallelize(ParallelType::BIDx);
  tv3b->axis(1)->parallelize(ParallelType::Unroll);
  tv3b->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  // TMA Tensors
  tv1a->axis(0)->parallelize(ParallelType::BIDx);
  tv1a->axis(1)->parallelize(ParallelType::TIDx);
  tv1a->axis(2)->parallelize(ParallelType::Bulk);

  tv0a->axis(0)->parallelize(ParallelType::BIDx);
  tv0a->axis(1)->parallelize(ParallelType::TIDx);
  tv0a->axis(2)->parallelize(ParallelType::Bulk);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(2)->parallelize(ParallelType::Bulk);

  // ComputeAt
  inlineMost();

  if (verbose_) {
    fusion->printMath();
    fusion->printKernel();
  }

  constexpr int dim0 = 32, dim1 = 2, dim2 = 4, dim3 = 256;
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim1, dim2, dim3}, options);
  at::Tensor at_tv1 = at::randn({dim0, dim1, dim2, dim3}, options);

  // Compile with KernelExecutor directly to avoid scheduling
  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0, at_tv1}, {}, index32bit);
  auto outputs = ke.run({at_tv0, at_tv1});

  auto at_output = at_tv0 + at_tv1;
  testValidate(
      fusion.get(), outputs, {at_tv0, at_tv1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(Tutorial, TMABankConflictFreeTranspose) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto input = makeContigTensor(2);
  fusion.addInput(input);
  auto output = transpose(input, 0, 1);
  fusion.addOutput(output);

  // Change the fusion to input->smem->register->smem->output
  // where the smem->register part does the transpose
  TensorView* input_smem_cache =
      input->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  input_smem_cache->setMemoryType(MemoryType::Shared);
  TensorView* output_smem_cache =
      output->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);
  output_smem_cache->setMemoryType(MemoryType::Shared);
  TensorView* output_reg_cache = output_smem_cache->cacheBefore();

  using Options =
      scheduler_utils::BoundedDirectionalTransformPropagator::Options;

  // Create 32x32 tile. Each CTA has one tile, and the entire tile will be
  // loaded to shared memory by TMA, and stored back to global memory by TMA.

  // [I1, I0]
  output->split(1, 32);
  output->split(0, 32);
  output->reorder({{-2, 0}});
  output->merge(0);
  // [I0/32 * I1/32', 32', 32]
  output->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, 32', 32]
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      output, -1, {input}, Options{}.propagateParallelType());

  // For fusion output, we just use TMA to store the entire tile back to global
  // memory. There is no need to further schedule the output tensor.
  output->axis(1)->parallelize(ParallelType::Bulk);
  output->axis(2)->parallelize(ParallelType::Bulk);
  // [BIDx, Bulk, Bulk]

  // output_smem_cache and output_reg_cache are scheduled in the same way.
  // We use each warp to load one column of input_smem_cache. We vectorize
  // the load to 16 byte, and use 8 warps to load all these 8 columns. And
  // when we write to output_smem_cache, we unroll the write. Each warp writes
  // one row in output_smem_cache in each iteration, so there is no bank
  // conflict.
  // [BIDx, 32', 32]
  output_smem_cache->setAllocationDomain(
      output_smem_cache->getLoopDomain(), true);
  output_smem_cache->split(1, 4);
  // [BIDx, 8', 4', 32]
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      output_smem_cache, -1, {input});
  output_smem_cache->merge(1, 3);
  // [BIDx, 256, 4']
  output_smem_cache->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      output_smem_cache,
      -1,
      {input_smem_cache},
      Options{}.propagateParallelType());
  output_smem_cache->axis(2)->parallelize(ParallelType::Unroll);
  output_reg_cache->axis(2)->parallelize(ParallelType::Vectorize);
  output_reg_cache->setAllocationDomain(
      output_reg_cache->getLoopDomain(), true);

  // Schedule the memory format for 128 byte swizzle
  // [BIDx, 8', 4', 32]
  input_smem_cache->reorder({{-1, 1}});
  // [BIDx, 32, 8', 4']
  input_smem_cache->split(1, 8);
  // [BIDx, 4, 8, 8', 4']
  input_smem_cache->swizzle(SwizzleType::XOR, 2, 3);
  // [BIDx, 4, 8, 8', 4']
  input_smem_cache->setAllocationDomain(
      input_smem_cache->getLoopDomain(), true);
  input_smem_cache->axis(1)->parallelize(ParallelType::Bulk);
  input_smem_cache->axis(2)->parallelize(ParallelType::Bulk);
  input_smem_cache->axis(3)->parallelize(ParallelType::Bulk);
  input_smem_cache->axis(4)->parallelize(ParallelType::Bulk);
  // [BIDx, Bulk, Bulk, Bulk, Bulk]

  if (verbose_) {
    fusion.print();
    fusion.printKernel();
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t = at::randn({10000, 10000}, options);
  KernelExecutor ke;
  CompileParams index32bit{DataType::Int32, 255, false};
  ke.compile(&fusion, {t}, {}, index32bit);
  auto outputs = ke.run({t});
  ASSERT_TRUE(at::equal(t.t(), outputs[0].as<at::Tensor>()));
}

} // namespace nvfuser

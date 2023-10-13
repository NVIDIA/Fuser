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
#include <inlining.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <test/utils.h>
#include <test/validator.h>
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
  std::vector<c10::IValue> aten_inputs = {t0};

  // Next, lower the fusion to Kernel, generate CUDA kernel source and then
  // compile it with nvrtc. All of them are done by FusionExecutor
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  // FusionExecutor now has a compiled kernel, which can be executed as:
  std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
  // Note that this run is done using just one thread, which will be
  // corrected below.

  // To validate the output, we can just assert that the output is
  // equal to the input as this is just a copy fusion. More commonly,
  // though, testValidate is used to validate outputs while
  // automatically adjusting thresholds of valid deviations
  ASSERT_TRUE(outputs[0].equal(t0));

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

  // Since the fusion is modified, we need to recompile it.
  FusionExecutor fe2;
  fe2.compileFusion(&fusion, aten_inputs);

  // This time, the kernel is launched with multiple threads and
  // thread blocks. Note that the launch configurations, i.e., the
  // thread block and grid shapes, are autoatically inferred from the
  // given inputs. To see how many threads are used, run this test
  // with NVFUSER_DUMP=launch_param
  outputs = fe2.runFusion(aten_inputs);

  ASSERT_TRUE(outputs[0].equal(t0));
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
  std::vector<c10::IValue> aten_inputs = {t0};
  at::Tensor ref = t0.sum({1});

  {
    FusionExecutor fe;
    fe.compileFusion(&fusion);
    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
    testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
  }

  // Next, use the same fusion but parallelize the reduction with
  // thread blocks
  tv1->axis(1)->parallelize(ParallelType::BIDx);

  if (verbose_) {
    fusion.printMath();
    fusion.printKernel();
  }

  {
    FusionExecutor fe;
    fe.compileFusion(&fusion);
    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
    testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
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
    FusionExecutor fe;
    fe.compileFusion(&fusion);
    // Running this fusion, however, should fail as it would require
    // thread blocks of shape 1024x10, i.e., the same shape as the
    // input tensor, which is too large in CUDA.
    //
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_ANY_THROW(fe.runFusion(aten_inputs));

    // Try again with a smaller input. This should launch a kernel
    // with thread blocks of shape 32x10
    at::Tensor t1 = at::randn({10, 32}, options);
    std::vector<at::Tensor> outputs = fe.runFusion({t1});
    testValidate(
        &fusion, outputs, aten_inputs, {t1.sum({1})}, __LINE__, __FILE__);
  }

  // We can of course mix BIDx and TIDx.
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  if (verbose_) {
    fusion.printMath();
    fusion.printKernel();
  }

  {
    FusionExecutor fe;
    fe.compileFusion(&fusion);
    // The original input should not fail in this case. The kernel
    // will be launched with 10 thread blocks, each of which has 1024
    // threads. Try running this test with NVFUSER_DUMP=launch_param
    // to see the launch configuration of each kernel lauch
    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
    testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
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
    // tv2[r1/1024, i1024] = tv0[i0]
    // tv1[r1024] = tv2[r1/1024, i1024]
    //
    if (verbose_) {
      fusion_copy.printMath();
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
    std::vector<c10::IValue> aten_inputs = {t0};
    at::Tensor ref = t0.sum({0});

    FusionExecutor fe;
    fe.compileFusion(&fusion_copy);

    // Since the size of the input is 10000, which is split by a
    // factor of 1024, the first per-thread reduction is done for
    // ceilDiv(10000, 1024) = 10 elements.
    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
    testValidate(&fusion_copy, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
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
    std::vector<c10::IValue> aten_inputs = {t0};
    at::Tensor ref = t0.sum({0});

    FusionExecutor fe;
    fe.compileFusion(&fusion_copy);

    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
    testValidate(&fusion_copy, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
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
      // Notice that tv1 has root and rfactor domains. The root domain
      // should consist of two IterDomains, whreas the rfactor domain
      // consists of a single IterDomain that is an output of a merge
      // operation of the two root IterDomains
      fusion.print();
    }

    // Check if the tv1 domains are generated as expected
    ASSERT_TRUE(tv1->hasRFactor());
    ASSERT_EQ(tv1->getRFactorDomain().size(), 1);
    ASSERT_TRUE(tv1->getRFactorDomain().at(0)->definition()->isA<Merge>());
    Merge* tv1_merge = tv1->getRFactorDomain().at(0)->definition()->as<Merge>();
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
    ASSERT_TRUE(reshape_output->definition()->isA<ViewOp>());
    auto squeeze_output =
        reshape_output->definition()->input(0)->as<TensorView>();
    ASSERT_TRUE(squeeze_output->definition()->isA<SqueezeOp>());

    ASSERT_TRUE(reshape_output->hasRFactor());
    ASSERT_EQ(reshape_output->getRFactorDomain().size(), 2);
    ASSERT_TRUE(
        reshape_output->getRFactorDomain().at(0)->definition()->isA<Split>());
    auto reshape_output_split =
        reshape_output->getRFactorDomain().at(0)->definition()->as<Split>();
    ASSERT_EQ(
        reshape_output_split->outer(),
        reshape_output->getRFactorDomain().at(0));
    ASSERT_EQ(
        reshape_output_split->inner(),
        reshape_output->getRFactorDomain().at(1));
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
        reshape_output->getLeafDomain().at(0)->definition()->isA<Split>());
    ASSERT_EQ(
        reshape_output->getLeafDomain()
            .at(0)
            ->definition()
            ->as<Split>()
            ->inner(),
        reshape_output->getLeafDomain().at(1));
    ASSERT_TRUE(reshape_output->getLeafDomain()
                    .at(0)
                    ->definition()
                    ->as<Split>()
                    ->in()
                    ->definition()
                    ->isA<Merge>());
    ASSERT_EQ(
        reshape_output->getLeafDomain()
            .at(0)
            ->definition()
            ->as<Split>()
            ->in()
            ->definition()
            ->as<Merge>()
            ->outer(),
        reshape_output->getRFactorDomain().at(0));
    ASSERT_EQ(
        reshape_output->getLeafDomain()
            .at(0)
            ->definition()
            ->as<Split>()
            ->in()
            ->definition()
            ->as<Merge>()
            ->inner(),
        reshape_output->getRFactorDomain().at(1));

    // Here's how we propagate the transformations of reshape_output
    // to all other tensors in the fusion
    TransformPropagatorWithCheck propagator(reshape_output);
    MaxRootDomainInfoSpanningTree(reshape_output).traverse(&propagator);

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
        squeeze_output->getLeafDomain().at(0)->definition()->isA<Split>());
    auto squeeze_output_second_split =
        squeeze_output->getLeafDomain().at(0)->definition()->as<Split>();
    ASSERT_EQ(
        squeeze_output_second_split->outer(),
        squeeze_output->getLeafDomain().at(0));
    ASSERT_EQ(
        squeeze_output_second_split->inner(),
        squeeze_output->getLeafDomain().at(1));

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
        squeeze_output->getRootDomain().at(0));
    ASSERT_EQ(
        squeeze_output_first_merge->inner(),
        squeeze_output->getRootDomain().at(1));

    // Note that all the transformations of squeeze_output are scheduling
    // transformations, thus it should not have a rfactor domain
    ASSERT_FALSE(squeeze_output->hasRFactor());
  }
}

} // namespace nvfuser

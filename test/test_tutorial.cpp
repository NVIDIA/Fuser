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

} // namespace nvfuser

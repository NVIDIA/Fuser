# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
from nvfuser_direct import (
    FusionDefinition,
    ParallelType,
    TensorView,
    Merge,
    Split,
    BroadcastOp,
    SqueezeOp,
    ReshapeOp,
)

verbose_ = False


def test_tutorial_memcpy():
    # First, we define a fusion. A common pattern is:
    # - Declare a Fusion, which works as a container of expressions using
    #   with context manager.
    # - Setup inputs. fd.define_tensor can be used to manually create tensors.
    #   fd.from_pytorch will create a TensorView given a pytorch tensor. Fusion
    #   registration is automatic.
    # - Define operations with the registered inputs.
    #   For supported operations, run:
    #   >>> import nvfuser_direct
    #   >>> fd = nvfuser_direct.FusionDefinition()
    #   >>> help(fd.ops)
    # - Most of operations that take tensors as inputs produce tensors
    #   as outputs, which can then be used as inputs to another
    #   operations.
    # - Final outputs should be set as fusion outputs with fd.add_output

    with FusionDefinition() as fd:
        # Create a 2D tensor of type float. It's "symbolic" as we do not
        # assume any specific shape except for that it's 2D.
        tv0 = fd.define_tensor(shape=[-1, -1])

        # Just create a copy
        tv1 = fd.ops.set(tv0)
        fd.add_output(tv1)

    if verbose_:
        # Here's some common ways to inspect the fusion. These are not
        # necessary for running the fusion but should provide helpful
        # information for understanding how fusions are transformed.

        # Print a concise representation of the fusion exprssions
        print(fd.fusion.print_math())

        # Generate and print a CUDA kernel. Notice that at this point the
        # genereated code is just a sequential kernel as we have not
        # scheduled the fusion yet, but it should be a valid CUDA kernel
        print(fd.fusion.print_kernel())

    # Next, try running the fusion. First, we need to set up a sample
    # input tensor. Here, we create a 32x32 tensor initialized with
    # random float values.

    t0 = torch.randn(32, 32, dtype=torch.float, device="cuda:0")

    # Next, lower the Fusion to Kernel, generate CUDA kernel source and then
    # compile it with nvrtc. After compilation, KernelExecutor now has a
    # compiled kernel, which can be executed as:
    outputs = fd.manual_execute([t0])

    # Note that this run is done using just one thread, which will be
    # corrected below.

    # To validate the output, we can just assert that the output is
    # equal to the input as this is just a copy fusion. More commonly,
    # though, fd.validate is used to validate outputs while
    # automatically adjusting thresholds of valid deviations.
    assert outputs[0].equal(t0)


def test_tutorial_memcpy_scheduled():
    # test_tutorial_memcpy_scheduled is a continuation from test_tutorial_memcpy

    # Instead of just running the fusion as is, we manually schedule it so that
    # it runs in parallel. In this case, we only have one expression, so we
    # just need to schedule tv1.

    # tv1 is a 2D tensor. Let its domain be [i0, i1]. We are going transform
    # this 2D domain to a CUDA Grid and Block. Specifically, a grid consisting
    # of multiple thread blocks, each of which containin multiple threads. A
    # common transformation pattern is to merge all of each axis to get a
    # flattened domain, and then split the domain to factor out axes that are
    # parallelized by threads and thread blocks.

    # In python, we can only modify the FusionDefinition inside a with context.
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(shape=[-1, -1])
        tv1 = fd.ops.set(tv0)
        fd.add_output(tv1)

        # For example, the current domain of tv1 looks like [i0, i1]. We can
        # merge the two axes by:
        tv1.merge(0, 1)

        # This creates a single axis that merges i0 and i1. Its extent is a
        # multiplication of the extents of i0 and i1, so we commonly represent
        # it as [i0 * i1]. It can be also examined with:
        if verbose_:
            print(tv1)

        # Next, we factor out a subdomain for threads in each thread block.
        tv1.split(0, 256)

        # In this case, the flattened domain is now 2D domain with an inner
        # domain of extent 256 and an outer domain of extent i0*i1/256, so the
        # tensor should now look like [i0*i1/256, 256]. Note that in reality we
        # do ceiling division as i0 * i1 may not be divisible by 256.
        if verbose_:
            print(tv1)

        # Now that we have two domains, we can parallelize each domain using
        # IterDomain.parallelize(ParallelType). Specifically, to parallelize the
        # inner domain with threads, we can do:
        tv1.axis(1).parallelize(ParallelType.block_x)
        # Similarly, to paralllize the outer domain with thread blocks:
        tv1.axis(0).parallelize(ParallelType.grid_x)
        # This way, the inner and outer axes are divided by blockDim.x threads
        # and gridDim.x blocks, respectively. Each element in each axis is
        # computed by one thread or one block, so this means that the size of
        # each thread block and a grid must match the size of each domain.
        # blockDim.x and gridDim.x must be 256 and i0*i1/256.

    # Now that the fusion is parallelized, it can be examined again.
    if verbose_:
        print(fd.fusion.print_math())
        # Notice that the axes of tv1 are now printed with blockIdx.x and
        # threadIdx.x, which shows they are parallelized by the
        # respective parallel types.

        # The CUDA kernel should look very differently as there should be no
        # for-loops.
        print(fd.fusion.print_kernel())

    # This time, the kernel is launched with multiple threads and thread
    # blocks. Note that the thread block and grid shapes are inferred from the
    # given inputs. To see how many threads are used, run this test
    # with NVFUSER_DUMP=launch_param
    t0 = torch.randn(32, 32, dtype=torch.float, device="cuda:0")
    outputs = fd.manual_execute([t0])
    assert outputs[0].equal(t0)


def test_tutorial_reduction():
    def fusion_func(fd: FusionDefinition) -> TensorView:
        # Create a 2D tensor
        tv0 = fd.define_tensor(shape=[-1, -1])

        # Reduce the second dimension
        tv1 = fd.ops.sum(tv0, dims=[1])
        fd.add_output(tv1)

        return tv1

    with FusionDefinition() as fd0:
        ref_tv = fusion_func(fd0)

        # At this point, nothing is parallelized. The reduction is done by
        # a single thread sequentially.

        if verbose_:
            print(fd0.fusion.print_math())
            print(fd0.fusion.print_kernel())

        # Block-parallel reduction
        ref_tv.axis(1).parallelize(ParallelType.block_x)

        if verbose_:
            print(fd0.fusion.print_math())
            print(fd0.fusion.print_kernel())

    t0 = torch.randn(10, 1024, dtype=torch.float, device="cuda:0")
    ref = t0.sum(dim=1)

    fd0.manual_validate([t0], [ref])

    # Create another FusionDefinition with same math but different schedule.
    with FusionDefinition() as fd1:
        ref_tv = fusion_func(fd1)

        # Next, use the same fusion but parallelize the reduction with
        # thread blocks
        ref_tv.axis(1).parallelize(ParallelType.grid_x)

        if verbose_:
            print(fd1.fusion.print_math())
            print(fd1.fusion.print_kernel())

    fd1.manual_validate([t0], [ref])

    # Create another FusionDefinition with same math but different schedule.
    with FusionDefinition() as fd2:
        ref_tv = fusion_func(fd2)

        # We can also parallelize the first axis as well. For example,
        # here's how threadIdx.x is used for the reduction and threadIdx.y
        # is used for the outer non-reduction domain
        ref_tv.axis(0).parallelize(ParallelType.block_y)
        ref_tv.axis(1).parallelize(ParallelType.block_x)

        if verbose_:
            print(fd2.fusion.print_math())
            print(fd2.fusion.print_kernel())

    # Running this fusion, however, should fail as it would require thread
    # blocks of shape 1024x10, i.e., the same shape as the input tensor, which
    # is too large in CUDA.
    with pytest.raises(RuntimeError):
        fd2.manual_validate([t0], [ref])

    # Try again with a smaller input. This should launch a kernel
    # with thread blocks of shape 32x10
    t1 = torch.randn(10, 32, dtype=torch.float, device="cuda:0")
    fd2.manual_validate([t1], [t1.sum(dim=1)])

    # Create another FusionDefinition with same math but different schedule.
    with FusionDefinition() as fd3:
        ref_tv = fusion_func(fd3)

        # We can of course mix BIDx and TIDx.
        ref_tv.axis(0).parallelize(ParallelType.grid_x)
        ref_tv.axis(1).parallelize(ParallelType.block_x)

        if verbose_:
            print(fd3.fusion.print_math())
            print(fd3.fusion.print_kernel())

    # The original input should not fail in this case. The kernel will be
    # launched with 10 thread blocks, each of which has 1024 threads. Try
    # running this test with NVFUSER_DUMP=launch_param to see the launch
    # configuration of each kernel lauch
    fd3.manual_validate([t0], [ref])


def test_tutorial_reduction_rfactor():
    # Just a very simple reduction of 1D tensor
    def fusion_func(fd: FusionDefinition) -> TensorView:
        tv0 = fd.define_tensor(shape=[-1])
        tv1 = fd.ops.sum(tv0, dims=[0])
        fd.add_output(tv1)
        return tv1

    # Create separate fusions because of multiple schedules
    with FusionDefinition() as fd0:
        tv1 = fusion_func(fd0)

        # A common pattern of reductions in CUDA involves multiple steps of
        # reductions, where the first step is a per-thread local reduction,
        # followed by a block reduction of the per-thread partial results,
        # and also potentially followed by a grid reduction of the
        # per-block partial results. Here's an example with a two-step
        # reduction:
        #
        # // Step 1: Per-thread reduction
        # float partial_result = 0;
        # for (int i = threadIdx.x; i += blockDim.x; i < N) {
        #   partial_result += input[i];
        # }
        #
        # // Step 2: Accumulation within each thread block
        # __shared__ float shared_buf[blockDim.x];
        # shared_buf[threadIdx.x] = partial_result;
        # __syncthreads();
        # float final_result = 0;
        # // Accumulation of the partila result in a naive sequntial way.
        # if (threadIdx.x == 0) {
        #   for (int i = 0; i < blockDim.x; ++i) {
        #     final_result += shared_buf[i];
        #   }
        # }

        # To reproduce the multi-step reduction pattern in nvFuser, a fusion
        # transformation called reduction rfactor is used. The basic idea is to
        # split a reduction domain such that each of the output domains of the
        # split is separately reduced. For example, tv1 can be transformed from
        # a 2D tensor to a 3D tensor as follows:

        # tv0: [i0]
        # tv1: [r1]
        tv1.split(0, 1024)
        # tv1: [r1/1024, r1024]

        # Both of the two inner domains are reduction domains, and we first
        # want to reduce the second domain, i.e., r1/1024, by each thread
        # independently, and then reduce the other reduction domain by a
        # block reduction. This can be done as follows:
        tv2 = tv1.rfactor([0])

        # The fusion math should now look like:
        # tv0: root = logical = [i{i0}]
        # tv2 = reduction(tv0): root = [r{i0}], logical = [r{i0/1024}, i{1024}]
        # tv1 = reduction(tv2): root = logical = [r{1024}]
        if verbose_:
            print(fd0.fusion.print_math())

        # Notice that the reduction operation is now split into two operations,
        # where the first one takes care of the first domain, and the second one
        # finishes up the remaining domain. The final values of tv1 is not
        # altered, but its computation is changed. (More strictly, since
        # floating-point addition is not associative, the final result will not
        # be exactly the same due to rounding errors)

        # To realize the parallelization as we sketched above, we can
        # use TIDx for both of tv1 and tv2 as follows:
        tv1.axis(0).parallelize(ParallelType.block_x)
        tv2.axis(1).parallelize(ParallelType.block_x)

        # At this point, tv2 is a TIDx-parallelized operation of multiple
        # independent reductions. There will be 1024 threads, each of which
        # reduces the first axis of size r1/1024. tv1 is also parallelized by
        # TIDx, but unlike tv2 the reduction domain is parallelized, so it
        # becomes a block-reduction operation.
        if verbose_:
            print(fd0.fusion.print_math())
            print(fd0.fusion.print_kernel())

    # Let's run the scheduled fusion
    t0 = torch.randn(10000, dtype=torch.float, device="cuda:0")
    ref = t0.sum(dim=0)
    fd0.manual_validate([t0], [ref])

    # We can further increase the parallelism by splitting the reduction domain
    # into three
    with FusionDefinition() as fd1:
        tv1 = fusion_func(fd1)

        # First, split for TIDx of 1024 threads
        tv1.split(0, 1024)
        # Next, split for BIDx of 100 thread blocks
        tv1.split(0, 100)
        # tv1: [r0/1024/100, r100, r1024]

        # Factoring out per-thread reduction
        tv2 = tv1.rfactor([1])
        # tv2: [i0/1024/100, r100, i1024]
        # tv1: [r0/1024/100, r1024]

        # Factoring out block reduction
        tv3 = tv1.rfactor([1])
        # tv2: [i0/1024/100, r100, i1024]
        # tv3: [i0/1024/100, r1024]
        # tv1: [r0/1024/100]

        # Parallelize each operation as follows
        # tv2: [bidx(i0/1024/100), r100, tidx(i1024)]
        # tv3: [bidx(i0/1024/100), tidx(r1024)]
        # tv1: [bidx(r0/1024/100)]
        tv2.axis(0).parallelize(ParallelType.grid_x)
        tv3.axis(0).parallelize(ParallelType.grid_x)
        tv1.axis(0).parallelize(ParallelType.grid_x)
        tv2.axis(2).parallelize(ParallelType.block_x)
        tv3.axis(1).parallelize(ParallelType.block_x)
        # Note that this could be also done more easily using
        # scheduler_utils::parallelizeAllLike.

        if verbose_:
            print(fd1.fusion.print_math())
            print(fd1.fusion.print_kernel())

    t1 = torch.randn(10000000, dtype=torch.float, device="cuda:0")
    ref1 = t1.sum(dim=0)
    fd1.manual_validate([t1], [ref1])


def test_tutorial_reshape():
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(shape=[4, 8])

        # Shape of tv0 is assumed to be [4, 8], which is then reshaped to [32]
        tv1 = fd.ops.reshape(tv0, [32])
        fd.add_output(tv1)

    if verbose_:
        # Notice that tv1 has root and logical domains. The root domain has two
        # IterDomains, whereas the logical domain consists of a single
        # IterDomain that is an output of a merge operation of the two root
        # IterDomains.
        print(fd.fusion.print_math())

    # Check if the tv1 domains are generated as expected
    assert tv1.has_root()
    assert len(tv1.get_logical_domain()) == 1
    # In python, use type() function to check an object's class.
    # In CPP, use isA template function.
    tv1_merge = tv1.get_logical_domain()[0].definition()
    assert type(tv1_merge) is Merge
    assert tv1_merge.inner() == tv1.get_root_domain()[1]
    assert tv1_merge.outer() == tv1.get_root_domain()[0]

    # Reshape example with broadcast domains
    with FusionDefinition() as fd1:
        # Create a 3D tensor with a broadcast domain
        tv0 = fd1.define_tensor(shape=[1, 2, 3])

        # tv0 is first squeezed and then reshaped and unsqueezed
        tv1 = fd1.ops.reshape(tv0, [3, 2, 1])
        fd1.add_output(tv1)

        if verbose_:
            print(fd1.fusion.print_math())

        # The fusion should look like:
        #
        # tv1 = unsqueeze(reshape(squeeze(tv0)));
        assert type(tv1.definition()) is BroadcastOp
        reshape_output = tv1.definition().input(0)
        assert type(reshape_output.definition()) is ReshapeOp
        squeeze_output = reshape_output.definition().input(0)
        assert type(squeeze_output.definition()) is SqueezeOp

        assert reshape_output.has_root()
        assert len(reshape_output.get_logical_domain()) == 2
        assert type(reshape_output.get_logical_domain()[0].definition()) is Split
        reshape_output_split = reshape_output.get_logical_domain()[0].definition()
        assert reshape_output_split.outer() == reshape_output.get_logical_domain()[0]
        assert reshape_output_split.inner() == reshape_output.get_logical_domain()[1]
        assert type(reshape_output_split.input(0).definition()) is Merge
        reshape_output_merge = reshape_output_split.input(0).definition()
        assert reshape_output_merge.outer() == reshape_output.get_root_domain()[0]
        assert reshape_output_merge.inner() == reshape_output.get_root_domain()[1]

        # So far, the fusion has transformations as part of its definition. It can
        # be further extended with scheduling transformations.
        reshape_output.merge(0, 1)
        reshape_output.split(0, 128)

        assert type(reshape_output.get_loop_domain()[0].definition()) is Split
        assert (
            reshape_output.get_loop_domain()[0].definition().inner()
            == reshape_output.get_loop_domain()[1]
        )
        assert (
            type(reshape_output.get_loop_domain()[0].definition().input(0).definition())
            == Merge
        )
        assert (
            reshape_output.get_loop_domain()[0]
            .definition()
            .input(0)
            .definition()
            .outer()
            == reshape_output.get_logical_domain()[0]
        )
        assert (
            reshape_output.get_loop_domain()[0]
            .definition()
            .input(0)
            .definition()
            .inner()
            == reshape_output.get_logical_domain()[1]
        )

        # Here's how we propagate the transformations of reshape_output to all
        # other tensors in the fusion
        fd.sched.transform_like(reshape_output)

        # Now, all tensors, including those before the reshape op, should be
        # transformed to 2D tensors with an inner domain of extent 128.
        if verbose_:
            print(fd.fusion.print_math())

        # Notice that all transformations of the reshape tensor, including both the
        # reshape and scheduling transformations, are propagated. For example,
        # squeeze_output should have the merge and split for the reshape, followed
        # by another merge and split for scheduling. Specifically:
        #
        # Root domain: [b0, i1, i2]
        # merge(1, 2) -> [b0, i1*i2]
        # outer split(1, 3) -> [b0, 3, i1*i2/3]
        # merge(1, 2) -> [b0, 3*i1*i2/3]
        # split(1, 128) -> [b0, 3*i1*i2/3/128, 128]
        assert type(squeeze_output.get_loop_domain()[0].definition()) is Split
        squeeze_output_second_split = squeeze_output.get_loop_domain()[0].definition()
        assert (
            squeeze_output_second_split.outer() == squeeze_output.get_loop_domain()[0]
        )
        assert (
            squeeze_output_second_split.inner() == squeeze_output.get_loop_domain()[1]
        )

        assert type(squeeze_output_second_split.input(0).definition()) is Merge
        squeeze_output_second_merge = squeeze_output_second_split.input(0).definition()

        assert type(squeeze_output_second_merge.outer().definition()) is Split
        squeeze_output_first_split = squeeze_output_second_merge.outer().definition()
        assert squeeze_output_first_split.outer() == squeeze_output_second_merge.outer()
        assert squeeze_output_first_split.inner() == squeeze_output_second_merge.inner()

        assert type(squeeze_output_first_split.input(0).definition()) is Merge
        squeeze_output_first_merge = squeeze_output_first_split.input(0).definition()
        assert (
            squeeze_output_first_merge.outer() == squeeze_output.get_logical_domain()[0]
        )
        assert (
            squeeze_output_first_merge.inner() == squeeze_output.get_logical_domain()[1]
        )

        # Note that all the transformations of squeeze_output are scheduling
        # transformations, thus it should not have a root domain
        assert not squeeze_output.has_root()

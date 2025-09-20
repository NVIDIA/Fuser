# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
from nvfuser_direct import FusionDefinition, ParallelType, TensorView

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

# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
import itertools
import math
from dataclasses import dataclass
from nvfuser_direct import (
    FusionDefinition,
    IdMappingMode,
    ParallelType,
    TensorView,
    Merge,
    Split,
    BroadcastOp,
    SqueezeOp,
    ReshapeOp,
    LoadStoreOpType,
    MemoryType,
    DataType,
    CompileParams,
    KernelExecutor,
    SchedulerType,
    PythonProfiler,
)
from nvfuser_direct import idm, schedule

from python.direct_utils import (
    is_pre_hopper,
)

verbose_ = True


# A helper function to test heuristic schedulers with automatic scheduling
def check_auto_schedule(schedule_fn):
    """
    A decorator to validate a schedule_fn before applying it to a fusion.

    Args:
        schedule_fn: The function to apply the scheduler
    """
    # List of all scheduler heuristics for testing
    # NOTE We cannot iterate pybind11 enum directly, so we extract the entries here.
    all_scheduler_heuristics = [
        heuristic
        for heuristic, _ in SchedulerType.__entries.values()
        if not SchedulerType.none
    ]

    def inner_fn(fusion, selected_heuristic, inputs):
        """
        Helper function to validate a schedule_fn.

        Args:
            fusion: The Fusion object to schedule
            selected_heuristic: The SchedulerType expected to work
            inputs: Input tensors for the fusion
        """
        available_heuristics = schedule.find_compatible_schedulers(fusion, inputs)

        # Assume that only a single heuristic is available for fusion
        assert len(available_heuristics) == 1

        # Check that only selected heuristic is available as a scheduler
        assert set(available_heuristics) == set([selected_heuristic])

        # Double-check with can_schedule
        status, _ = schedule.can_schedule(fusion, selected_heuristic, inputs)
        assert status

        # Check that the other schedulers are not compatible with this fusion
        assert all(
            [
                not schedule.can_schedule(fusion, h, inputs)[0]
                for h in all_scheduler_heuristics
                if h is not selected_heuristic
            ]
        )
        return schedule_fn(fusion, selected_heuristic, inputs)

    return inner_fn


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


def test_tutorial_id_model_reshape_analysis():
    """
    Demonstration of using IdModel for analyzing equivalence of reshape ops
    """
    with FusionDefinition() as fd:
        # Use the static reshape to avoid reshape concretization.
        tv0 = fd.define_tensor(shape=[10, 20])
        tv1 = fd.define_tensor(shape=[10, 20])

        # While the reshape operations are equivalent, we do not know if the two
        # inputs are the same. There is not an operation allowing us to infer
        # equivalence. e.g., tv0 + tv1.
        tv2 = fd.ops.reshape(tv0, [20, 10])
        tv3 = fd.ops.reshape(tv1, [20, 10])
        fd.add_output(tv2)
        fd.add_output(tv3)

    id_model = idm.IdModel(fd.fusion)
    exact_graph = id_model.maybe_build_graph(IdMappingMode.exact)

    if verbose_:
        print(id_model)
        print(exact_graph)
        print(exact_graph.disjoint_val_sets())

    # As mentioned above, we do not know any relationship between tv0 and tv1.
    # They should not be mapped in exact graph.
    assert len(tv0.get_logical_domain()) == len(tv1.get_logical_domain())
    for tv0_id, tv1_id in zip(tv0.get_logical_domain(), tv1.get_logical_domain()):
        assert not exact_graph.disjoint_val_sets().strict_are_mapped(tv0_id, tv1_id)

    # Thus, the outputs of the reshape ops are not mapped either
    assert len(tv2.get_loop_domain()) == len(tv3.get_loop_domain())
    for tv2_id, tv3_id in zip(tv2.get_loop_domain(), tv3.get_loop_domain()):
        assert not exact_graph.disjoint_val_sets().strict_are_mapped(tv2_id, tv3_id)

    # Now, suppose we can say the inputs are exactly mapped. We can manually
    # add mappings:
    for tv0_id, tv1_id in zip(tv0.get_logical_domain(), tv1.get_logical_domain()):
        exact_graph.map_vals(tv0_id, tv1_id)

    # Now, tv2 and tv3 should be fully mapped, including their root,
    # intermediate and loop domains.

    # Check the root domains.
    assert len(tv2.get_root_domain()) == len(tv3.get_root_domain())
    for tv2_id, tv3_id in zip(tv2.get_root_domain(), tv3.get_root_domain()):
        assert exact_graph.disjoint_val_sets().strict_are_mapped(tv2_id, tv3_id)

    # The reshape consists of a merge and split. The output of the merge should
    # be mapped as well
    assert exact_graph.disjoint_val_sets().strict_are_mapped(
        tv2.get_root_domain()[0].uses()[0].output(0),
        tv3.get_root_domain()[0].uses()[0].output(0),
    )

    # The next operation is split. Its outputs, which are the loop domains,
    # should be mapped too.
    for tv2_id, tv3_id in zip(tv2.get_loop_domain(), tv3.get_loop_domain()):
        assert exact_graph.disjoint_val_sets().strict_are_mapped(tv2_id, tv3_id)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_basic_tma_example1(nvfuser_direct_test):
    """
    This tutorial uses copy kernels to demonstrate how to schedule TMA.
    Please note that this is not a guide on how to use TMA to achieve SOL.
    Instead, it is a demonstration on the degree of freedoms we have in a
    TMA schedule and how a schedule is translated into generated code in the
    kernel. I also want the readers to focus on the schedule of TMA. The
    other parts of the kernel that is scheduled is not important here.
    Indeed, I picked a random, valid schedule. For the example about TMA
    load, please focus on the schedule of the shared memory tensor. For the
    example about TMA store, please focus on the allocation domain of the
    shared memory tensor and the fusion output.
    """

    # In this example, we treat the fusion as 1D, which is similar to how we
    # generally schedule pointwise fusions. We use a single 1D TMA instruction to
    # load the entire CTA tile to shared memory.
    # CTA tile size = TMA tile size = 256
    with FusionDefinition() as fd:
        input = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        output = fd.ops.set(input)
        fd.add_output(output)

        smem_cache = input.cache_after(LoadStoreOpType.tma)
        smem_cache.set_memory_type(MemoryType.shared)

        # For TMA load, both the shared memory layout and the loop nest and
        # parallelization of TMA are specified by the consumer: smem_cache

        # Step 1: define TMA domain
        # We want to treat the entire tensor as 1D so define the TMA domain as
        # [I0*I1*I2]
        smem_cache.merge(0, 1)
        smem_cache.merge(0, 1)
        # Note that the TMA domain only exists in people's mind, there is no need to
        # set anything here.

        # Step 2: define box
        smem_cache.split(0, 256)
        # [I0*I1*I2/256, 256]
        # partitioned IterDomain: I0*I1*I2
        # coordinate IterDomain: I0*I1*I2/256
        # box IterDomain: 256

        # Step 3: define tile
        # We use dense tile here, so tile == box. Nothing to do here.

        # Step 4: schedule the shared memory tensor
        # By default, the allocation domain is the logical domain, which is already
        # in good shape for this case.

        # Step 5: schedule the consumer tensor
        smem_cache.axis(0).parallelize(ParallelType.grid_x)
        smem_cache.axis(1).parallelize(ParallelType.tma)
        # [BIDx, TMA]

        # Schedule the smem->gmem part
        output.merge(0, 1)
        output.merge(0, 1)
        output.split(0, 256)
        output.axis(0).parallelize(ParallelType.grid_x)
        output.axis(1).parallelize(ParallelType.block_x)

    if verbose_:
        print(fd.fusion.print_math())
        print(fd.fusion.print_kernel())
        # TMA will be generated like:
        # Note that the coordinate is in number of items, smem address is in
        # bytes
        #
        # if (threadIdx.x == 0) {
        #   Hopper::cpAsyncBulkTensorTileG2S(
        #       coordinate = {256 * blockIdx.x},
        #       smem_addr = toSmem(T2));
        # }

    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(5, 3, 300, dtype=torch.float, device="cuda:0")
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0], compile_params=index32bit)
    outputs = ke.run([t0])
    assert outputs[0].equal(t0)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_basic_tma_example2(nvfuser_direct_test):
    """
    Example 2:
     Similar to example 1, we treat the fusion as 1D and uses 1D TMA to load
     data to shared memory. But this time, instead of using 1 TMA instruction
     to load the entire CTA tile, we use 4 TMA instructions. We use a for loop
     to launch these 4 instructions
     CTA tile size = 4 * TMA tile size = 1024
    """
    with FusionDefinition() as fd:
        input = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        output = fd.ops.set(input)
        fd.add_output(output)

        smem_cache = input.cache_after(LoadStoreOpType.tma)
        smem_cache.set_memory_type(MemoryType.shared)

        # For TMA load, both the shared memory layout and the loop nest and
        # parallelization of TMA are specified by the consumer: smem_cache

        # Step 1: define TMA domain
        # We want to treat the entire tensor as 1D, so define the TMA domain as
        # [I0*I1*I2]
        smem_cache.merge(0, 1)
        smem_cache.merge(0, 1)
        # Note that the TMA domain only exist in people's mind, there is no need to
        # set anything here.

        # Step 2: define box
        smem_cache.split(0, 256)
        # [I0*I1*I2/256, 256]
        # partitioned IterDomain: I0*I1*I2
        # coordinate IterDomain: I0*I1*I2/256
        # box IterDomain: 256

        # Step 3: define tile
        # We use dense tile here, so tile == box. Nothing to do here.

        # Step 4: schedule the shared memory tensor
        # By default, the allocation domain is the logical domain, which is already
        # in good shape for this case.

        # Step 5: schedule the consumer tensor
        smem_cache.split(0, 4)
        # [I0*I1*I2/256/4, 4, 256]
        smem_cache.axis(0).parallelize(ParallelType.grid_x)
        smem_cache.axis(2).parallelize(ParallelType.tma)
        # [BIDx, Serial, TMA]

        # Schedule the smem->gmem part
        output.merge(0, 1)
        output.merge(0, 1)
        output.split(0, 256)
        output.split(0, 4)
        output.axis(0).parallelize(ParallelType.grid_x)
        output.axis(2).parallelize(ParallelType.block_x)

    if verbose_:
        print(fd.fusion.print_math())
        print(fd.fusion.print_kernel())
        # TMA will be generated like:
        # Note that the coordinate is in number of items, smem address is in
        # bytes
        #
        # for (nvfuser_index_t i8 = 0; i8 < 4; ++i8) {
        #   if (threadIdx.x == 0) {
        #     Hopper::cpAsyncBulkTensorTileG2S(
        #         coordinate = {1024 * blockIdx.x + 256 * i8},
        #         smem_addr = (toSmem(T2) + 1024 * i8));
        #   }
        # }

    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(5, 3, 300, dtype=torch.float, device="cuda:0")
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0], compile_params=index32bit)
    outputs = ke.run([t0])
    assert outputs[0].equal(t0)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_basic_tma_example3(nvfuser_direct_test):
    """
    Example 3:
     Similar to example 2, we treat the fusion as 1D and use 1D TMA to load data
     to shared memory. 4 TMA instructions are used to load the entire CTA tile.
     However, instead of using a for loop to launch these 4 instructions, we
     parallelize these 4 instructions to TIDx.
     CTA tile size = 4 * TMA tile size = 1024
    """

    with FusionDefinition() as fd:
        input = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        output = fd.ops.set(input)
        fd.add_output(output)

        smem_cache = input.cache_after(LoadStoreOpType.tma)
        smem_cache.set_memory_type(MemoryType.shared)

        # For TMA load, both the shared memory layout and the loop nest and
        # parallelization of TMA are specified by the consumer: smem_cache

        # Step 1: define TMA domain
        # Because we want to treat the entire tensor as 1D, we define the TMA
        # domain as [I0*I1*I2]
        smem_cache.merge(0, 1)
        smem_cache.merge(0, 1)
        # Note that the TMA domain only exist in people's mind, there is no need to
        # set anything here.

        # Step 2: define box
        smem_cache.split(0, 256)
        # [I0*I1*I2/256, 256]
        # partitioned IterDomain: I0*I1*I2
        # coordinate IterDomain: I0*I1*I2/256
        # box IterDomain: 256

        # Step 3: define tile
        # We use dense tile here, so tile == box. Nothing to do here.

        # Step 4: schedule the shared memory tensor
        # By default, the allocation domain is the logical domain, which is already
        # in good shape for this case.

        # Step 5: schedule the consumer tensor
        smem_cache.split(0, 4)
        # [I0*I1*I2/256/4, 4, 256]
        smem_cache.axis(0).parallelize(ParallelType.grid_x)
        smem_cache.axis(1).parallelize(ParallelType.block_x)
        smem_cache.axis(2).parallelize(ParallelType.tma)
        # [BIDx, TIDx, TMA]

        # Schedule the smem->gmem part
        output.merge(0, 1)
        output.merge(0, 1)
        output.split(0, 256)
        output.split(0, 4)
        output.axis(0).parallelize(ParallelType.grid_x)
        output.axis(2).parallelize(ParallelType.block_x)

        if verbose_:
            print(fd.fusion.print_math())
            print(fd.fusion.print_kernel())
            # TMA will be generated like:
            # Note that the coordinate is in number of items, smem address is in
            # bytes
            #
            # if (threadIdx.x < 4) {
            #   Hopper::cpAsyncBulkTensorTileG2S(
            #       coordinate = {1024 * blockIdx.x + 256 * threadIdx.x},
            #       smem_addr = (toSmem(T2) + 1024 * threadIdx.x));
            # }

    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(5, 3, 300, dtype=torch.float, device="cuda:0")
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0], compile_params=index32bit)
    outputs = ke.run([t0])
    assert outputs[0].equal(t0)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_basic_tma_example4(nvfuser_direct_test):
    """
    Example 4: Similar to example 3, except that we are using TMA for store
    instead of load.
    """

    with FusionDefinition() as fd:
        input = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        output = fd.ops.set(input)
        fd.add_output(output)

        smem_cache = output.cache_before(LoadStoreOpType.tma)
        smem_cache.set_memory_type(MemoryType.shared)

        # For TMA store, the loop nest and parallelization is specified in the
        # consumer `output`, and the shared memory layout is specified in the
        # allocation dimain of `smem_cache`.

        # Step 1: define TMA domain
        # Because we want to treat the entire tensor as 1D, we define the TMA
        # domain as [I0*I1*I2]
        output.merge(0, 1)
        output.merge(0, 1)
        # Note that the TMA domain only exist in people's mind, there is no need to
        # set anything here.

        # Step 2: define box
        output.split(0, 256)
        # [I0*I1*I2/256, 256]
        # partitioned IterDomain: I0*I1*I2
        # coordinate IterDomain: I0*I1*I2/256
        # box IterDomain: 256

        # Step 3: define tile
        # We use dense tile here, so tile == box. Nothing to do here.

        # Step 4: schedule the shared memory tensor
        # By default, the allocation domain is the logical domain, which is already
        # in good shape for this case.

        # Step 5: schedule the consumer tensor
        output.split(0, 4)
        # [I0*I1*I2/256/4, 4, 256]
        output.axis(0).parallelize(ParallelType.grid_x)
        output.axis(1).parallelize(ParallelType.block_x)
        output.axis(2).parallelize(ParallelType.tma)
        # [BIDx, TIDx, TMA]

        # Schedule the gmem->smem part
        smem_cache.merge(0, 1)
        smem_cache.merge(0, 1)
        smem_cache.split(0, 256)
        smem_cache.split(0, 4)
        smem_cache.axis(0).parallelize(ParallelType.grid_x)
        smem_cache.axis(2).parallelize(ParallelType.block_x)

        if verbose_:
            print(fd.fusion.print_math())
            print(fd.fusion.print_kernel())
            # TMA will be generated like:
            # Note that the coordinate is in number of items, smem address is in
            # bytes
            #
            # if (threadIdx.x < 4) {
            #   Hopper::cpAsyncBulkTensorTileS2G(
            #       coordinate = {1024 * blockIdx.x + 256 * threadIdx.x},
            #       smem_addr = (toSmem(T2) + 1024 * threadIdx.x));
            # }

    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(5, 3, 300, dtype=torch.float, device="cuda:0")
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0], compile_params=index32bit)
    outputs = ke.run([t0])
    assert outputs[0].equal(t0)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_basic_tma_example5(nvfuser_direct_test):
    """
    Example 5: Still the same copy kernel of 3D tensor, but this time, we
     want to do tiling on the inner two dimensions. The first dimension is
     treated as a "batch" dimension. We use CTA tile (64, 64), and TMA tile
     (32, 32), so we need 4 TMA instructions to load the entire CTA tile.
     We want to use two threads, and each thread issue two TMA instructions.
    """

    with FusionDefinition() as fd:
        input = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        output = fd.ops.set(input)
        fd.add_output(output)

        smem_cache = input.cache_after(LoadStoreOpType.tma)
        smem_cache.set_memory_type(MemoryType.shared)

        # For TMA load, both the shared memory layout and the loop nest and
        # parallelization of TMA are specified by the consumer: smem_cache

        # Step 1: define TMA domain
        # For this case, we want to treat all three dimensions separately.
        # TMA domain: [I0, I1, I2]
        # Note that the TMA domain only exist in people's mind, there is no need to
        # set anything here.

        # Step 2: define box
        smem_cache.split(2, 32)
        smem_cache.split(1, 32)
        # [I0, I1/32, 32, I2/32', 32']
        # Box dimensions defined by partitioning: I1 and I2
        #   partitioned IterDomain: I1, I2
        #   coordinate IterDomain: I1/32, I2/32'
        #   box IterDomain: 32, 32'
        # Box dimension defined by compositing: I0
        #   coordinate IterDomain: I0
        #   box IterDomain: no box IterDomain, so implicit size 1

        # Step 3: define tile
        # We use dense tile here, so tile == box. Nothing to do here.

        # Step 4: schedule the shared memory tensor
        # By default, the allocation domain is the logical domain. The default
        # value does not work for this case, because the tile will not be
        # contiguous in shared memory.
        # [I0, I1/32, 32, I2/32', 32']
        smem_cache.split(3, 2)
        smem_cache.split(1, 2)
        # [I0, I1/32/2, 2, 32, I2/32'/2', 2', 32']
        smem_cache.reorder({3: -2, 2: -4})
        # [I0, I1/32/2, I2/32'/2', 2, 2', 32, 32']
        smem_cache.set_allocation_domain(
            smem_cache.get_loop_domain(), new_contiguity=True
        )

        # Step 5: schedule the consumer tensor
        # [I0, I1/32/2, I2/32'/2', 2, 2', 32, 32']
        smem_cache.axis(0).parallelize(ParallelType.grid_x)
        smem_cache.axis(1).parallelize(ParallelType.grid_y)
        smem_cache.axis(2).parallelize(ParallelType.grid_z)
        smem_cache.axis(3).parallelize(ParallelType.block_x)
        smem_cache.axis(5).parallelize(ParallelType.tma)
        smem_cache.axis(6).parallelize(ParallelType.tma)
        # [BIDx, BIDy, BIDz, TIDx, Serial, TMA, TMA]

        # Schedule the smem->gmem part
        output.split(2, 32)
        output.split(1, 32)
        output.split(3, 2)
        output.split(1, 2)
        output.reorder({3: -2, 2: -4})
        output.axis(0).parallelize(ParallelType.grid_x)
        output.axis(1).parallelize(ParallelType.grid_y)
        output.axis(2).parallelize(ParallelType.grid_z)
        output.merge(3, 4)
        output.axis(3).parallelize(ParallelType.block_x)

        if verbose_:
            print(fd.fusion.print_math())
            print(fd.fusion.print_kernel())
            # TMA will be generated like:
            # Note that the coordinate is in number of items, smem address is in
            # bytes. Also note that coordinate is in column major, so inner dims
            # goes first
            #
            # for (nvfuser_index_t i13 = 0; i13 < 2; ++i13) {
            #   if (threadIdx.x < 2) {
            #     Hopper::cpAsyncBulkTensorTileG2S(
            #         coordinate =
            #             {64 * blockIdx.z + 32 * i13,
            #              64 * blockIdx.y + 32 * threadIdx.x,
            #              blockIdx.x},
            #         smem_addr = toSmem(T2) + 8192 * threadIdx.x + 4096 * i13);
            #   }
            # }

    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(5, 3, 300, dtype=torch.float, device="cuda:0")
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0], compile_params=index32bit)
    outputs = ke.run([t0])
    assert outputs[0].equal(t0)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_basic_tma_example6(nvfuser_direct_test):
    """
    Example 6: Similar to example 5, but we are using TMA for store instead
    of load.
    """

    with FusionDefinition() as fd:
        input = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        output = fd.ops.set(input)
        fd.add_output(output)

        smem_cache = output.cache_before(LoadStoreOpType.tma)
        smem_cache.set_memory_type(MemoryType.shared)

        # For TMA store, the loop nest and parallelization is specified in the
        # consumer `output`, and the shared memory layout is specified in the
        # allocation dimain of `smem_cache`.

        # Step 1: define TMA domain
        # For this case, we want to treat all three dimensions separately.
        # TMA domain: [I0, I1, I2]
        # Note that the TMA domain only exist in people's mind, there is no need to
        # set anything here.

        # Step 2: define box
        output.split(2, 32)
        output.split(1, 32)
        # [I0, I1/32, 32, I2/32', 32']
        # Box dimensions defined by partitioning: I1 and I2
        #   partitioned IterDomain: I1, I2
        #   coordinate IterDomain: I1/32, I2/32'
        #   box IterDomain: 32, 32'
        # Box dimension defined by compositing: I0
        #   coordinate IterDomain: I0
        #   box IterDomain: no box IterDomain, so implicit size 1

        # Step 3: define tile
        # We use dense tile here, so tile == box. Nothing to do here.

        # Step 4: schedule the shared memory tensor
        # By default, the allocation domain is the logical domain. The default
        # value does not work for this case, because th tile will not be
        # contiguous in shared memory.
        # [I0, I1, I2]
        smem_cache.split(2, 32)
        smem_cache.split(1, 32)
        # [I0, I1/32, 32, I2/32', 32']
        smem_cache.split(3, 2)
        smem_cache.split(1, 2)
        # [I0, I1/32/2, 2, 32, I2/32'/2', 2', 32']
        smem_cache.reorder({3: -2, 2: -4})
        # [I0, I1/32/2, I2/32'/2', 2, 2', 32, 32']
        smem_cache.set_allocation_domain(
            smem_cache.get_loop_domain(), new_contiguity=True
        )

        # Step 5: schedule the consumer tensor
        # Because we are not inlining anything in this example, we do not care
        # about the order of IterDomains.
        # [I0, I1/32, 32, I2/32', 32']
        output.split(3, 2)
        output.split(1, 2)
        # [I0, I1/32/2, 2, 32, I2/32'/2', 2', 32']
        output.axis(0).parallelize(ParallelType.grid_x)
        output.axis(1).parallelize(ParallelType.grid_y)
        output.axis(2).parallelize(ParallelType.block_x)
        output.axis(3).parallelize(ParallelType.tma)
        output.axis(4).parallelize(ParallelType.grid_z)
        output.axis(6).parallelize(ParallelType.tma)
        # [BIDx, BIDy, TIDx, TMA, BIDz, Serial, TMA]

        # Schedule the gmem->smem part
        smem_cache.merge(-2, -1)
        smem_cache.axis(0).parallelize(ParallelType.grid_x)
        smem_cache.axis(1).parallelize(ParallelType.grid_y)
        smem_cache.axis(2).parallelize(ParallelType.grid_z)
        smem_cache.axis(-1).parallelize(ParallelType.block_x)

        if verbose_:
            print(fd.fusion.print_math())
            print(fd.fusion.print_kernel())
            # TMA will be generated like:
            # Note that the coordinate is in number of items, smem address is in
            # bytes.Also note that coordinate is in column major, so inner dims
            # goes first
            #
            # for (nvfuser_index_t i19 = 0; i19 < 2; ++i19) {
            #   if (threadIdx.x < 2) {
            #     Hopper::cpAsyncBulkTensorTileS2G(
            #         coordinate =
            #             {64 * blockIdx.z + 32 * i19,
            #              64 * blockIdx.y + 32 * threadIdx.x,
            #              blockIdx.x},
            #         smem_addr = toSmem(T2) + 8192 * threadIdx.x + 4096 * i19);
            #   }
            # }

    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(5, 3, 300, dtype=torch.float, device="cuda:0")
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0], compile_params=index32bit)
    outputs = ke.run([t0])
    assert outputs[0].equal(t0)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_vectorize_store_pointwise_tma(nvfuser_direct_test):
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True])
        tv1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True])
        tv2 = fd.ops.add(tv0, tv1)
        fd.add_output(tv2)

        # Create cache_tvs
        tv0a = tv0.cache_after(LoadStoreOpType.tma)
        tv1a = tv1.cache_after(LoadStoreOpType.tma)
        tv2b = tv2.cache_before()

        tv0a.set_memory_type(MemoryType.shared)
        tv1a.set_memory_type(MemoryType.shared)

        reference_tv = tv2

        # Step 1: Create tma domain
        # Use the root domain as TMA domain
        #   root domain: [I0, I1]

        num_threads = 128
        vectorization = 2
        tma_tile = num_threads * vectorization
        num_stages = 4
        num_ctas_for_hopper = 132

        # Step 2: Create Box
        # After TMA domain creation
        #         split: [I0, I3, 256]
        reference_tv.split(-1, tma_tile)
        #         split: [I2, 4, I3, 256]
        reference_tv.split(0, num_stages)

        # Step 3: Create Tile
        # Do nothing here because box == tile

        # Step 4: Schedule Shared Memory Tensor
        #         split: [I2, 4, I3, 128, 2]
        reference_tv.split(-1, vectorization)
        #         split: [I4, 132, 4, I3, 128, 2]
        reference_tv.split(0, num_ctas_for_hopper)
        #         reorder: [I4, 132, I3, 4, 128, 2]
        reference_tv.reorder({3: 2, 2: 3})

        # Transform Operations between cache operations and output reference
        fd.sched.transform_like(reference_tv)

        # Propagate common parallel dimensions
        reference_tv.axis(1).parallelize(ParallelType.grid_x)
        fd.sched.parallelize_like(reference_tv)

        tv2b.axis(-2).parallelize(ParallelType.block_x)

        # Vectorization for writing results to gmem
        reference_tv.axis(-3).parallelize(ParallelType.unroll)
        reference_tv.axis(-2).parallelize(ParallelType.block_x)
        reference_tv.axis(-1).parallelize(ParallelType.vectorize)

        # Apply bulk type to TMA tensors
        tv0a.axis(-1).parallelize(ParallelType.tma)
        tv0a.axis(-2).parallelize(ParallelType.tma)
        tv0a.axis(-3).parallelize(ParallelType.tma)

        tv1a.axis(-1).parallelize(ParallelType.tma)
        tv1a.axis(-2).parallelize(ParallelType.tma)
        tv1a.axis(-3).parallelize(ParallelType.tma)

        # ComputeAt
        fd.sched.inline_most()

        if verbose_:
            print(fd.fusion.print_math())
            print(fd.fusion.print_kernel())

    dim0 = 16384
    dim1 = 16384

    # Compile with KernelExecutor directly to avoid scheduling
    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(dim0, dim1, dtype=torch.float, device="cuda:0")
    t1 = torch.randn(dim0, dim1, dtype=torch.float, device="cuda:0")
    t2 = t0 + t1
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0, t1], compile_params=index32bit)
    outputs = ke.run([t0, t1])
    assert outputs[0].equal(t2)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_pointwise_broadcast_tma(nvfuser_direct_test):
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        tv1 = fd.define_tensor(
            shape=[-1, -1, -1, -1], contiguity=[True, False, True, True]
        )
        tv2 = fd.ops.broadcast(tv0, [True, False, False, False])
        tv3 = fd.ops.add(tv2, tv1)
        fd.add_output(tv3)

        # Create cache_tvs
        tv0a = tv0.cache_after(LoadStoreOpType.tma)
        tv1a = tv1.cache_after(LoadStoreOpType.tma)
        tv3b = tv3.cache_before(LoadStoreOpType.tma)

        tv0a.set_memory_type(MemoryType.shared)
        tv1a.set_memory_type(MemoryType.shared)
        tv3b.set_memory_type(MemoryType.shared)

        reference_tv = tv3

        # Step 1: Create tma domain
        #   root domain: [I0, I1, I2, I3]
        #    TMA domain: [I0, I1, I4]
        reference_tv.merge(-2, -1)

        # Step 2: Define TMA Box
        #         split: [I0, I1, I5, 256]
        reference_tv.split(-1, 256)

        # Step 3: Define Tile
        # Do nothing here because tile == box.

        # Step 4: Schedule Shared Memory Tensor
        #         merge: [I10, I5, 256]
        reference_tv.merge(0, 1)
        #         split: [I10, I7, 4, 256]
        reference_tv.split(-2, 4)
        #         merge: [I11, 4, 256]
        reference_tv.merge(0, 1)

        # Transform Operations between cache operations and output reference
        fd.sched.transform_like(reference_tv)

        # Define Parallelization Schema
        # Intermediate Tensors
        tv3b.axis(0).parallelize(ParallelType.grid_x)
        tv3b.axis(1).parallelize(ParallelType.unroll)
        tv3b.axis(2).parallelize(ParallelType.block_x)

        tv2.axis(0).parallelize(ParallelType.grid_x)
        tv2.axis(1).parallelize(ParallelType.unroll)
        tv2.axis(2).parallelize(ParallelType.block_x)

        # TMA Tensors
        tv1a.axis(0).parallelize(ParallelType.grid_x)
        tv1a.axis(1).parallelize(ParallelType.block_x)
        tv1a.axis(2).parallelize(ParallelType.tma)

        tv0a.axis(0).parallelize(ParallelType.grid_x)
        tv0a.axis(1).parallelize(ParallelType.block_x)
        tv0a.axis(2).parallelize(ParallelType.tma)

        tv3.axis(0).parallelize(ParallelType.grid_x)
        tv3.axis(1).parallelize(ParallelType.block_x)
        tv3.axis(2).parallelize(ParallelType.tma)

        # ComputeAt
        fd.sched.inline_most()

        if verbose_:
            print(fd.fusion.print_math())
            print(fd.fusion.print_kernel())

    dim0 = 32
    dim1 = 2
    dim2 = 4
    dim3 = 256

    # Compile with KernelExecutor directly to avoid scheduling
    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(dim1, dim2, dim3, dtype=torch.float, device="cuda:0")
    t1 = torch.randn(dim0, dim1, dim2, dim3, dtype=torch.float, device="cuda:0")
    t2 = t0 + t1
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0, t1], compile_params=index32bit)
    outputs = ke.run([t0, t1])
    assert outputs[0].equal(t2)


@pytest.mark.skipif(
    is_pre_hopper(), reason="Only supported on Hopper and newer devices."
)
def test_tutorial_tma_bank_conflict_free_transpose(nvfuser_direct_test):
    with FusionDefinition() as fd:
        input = fd.define_tensor(shape=[-1, -1], contiguity=[True, True])
        output = fd.ops.permute(input, [1, 0])
        fd.add_output(output)

        # Change the fusion to input->smem->register->smem->output where the
        # smem->register part does the transpose
        input_smem_cache = input.cache_after(LoadStoreOpType.tma)
        input_smem_cache.set_memory_type(MemoryType.shared)

        output_smem_cache = output.cache_before(LoadStoreOpType.tma)
        output_smem_cache.set_memory_type(MemoryType.shared)

        output_reg_cache = output_smem_cache.cache_before()

        # Create 32x32 tile. Each CTA has one tile, and the entire tile will be
        # loaded to shared memory by TMA, and stored back to global memory by TMA.

        # [I1, I0]
        output.split(1, 32)
        output.split(0, 32)
        # [I1, 32', I0, 32]
        output.reorder({0: 1, 1: 2, 2: 0})
        output.merge(0, 1)
        # [I0/32 * I1/32', 32', 32]
        output.axis(0).parallelize(ParallelType.grid_x)
        # [BIDx, 32', 32]

        fd.sched.bounded_transform_backward(
            output, -1, [input], propagate_parallel_type=True
        )

        # For fusion output, we just use TMA to store the entire tile back to global
        # memory. There is no need to further schedule the output tensor.
        output.axis(1).parallelize(ParallelType.tma)
        output.axis(2).parallelize(ParallelType.tma)
        # [BIDx, Bulk, Bulk]

        # output_smem_cache and output_reg_cache are scheduled in the same way.
        # We use each warp to load one column of input_smem_cache. We vectorize
        # the load to 16 bytes, and use 8 warps to load all these 8 columns. Then,
        # when we write to output_smem_cache, we unroll the write. Each warp writes
        # one row in output_smem_cache in each iteration, so there is no bank
        # conflict.

        # [BIDx, 32', 32]
        output_smem_cache.set_allocation_domain(
            output_smem_cache.get_loop_domain(), new_contiguity=True
        )
        output_smem_cache.split(1, 4)
        # [BIDx, 8', 4', 32]

        fd.sched.bounded_transform_backward(output_smem_cache, -1, [input])

        output_smem_cache.merge(1, 3)
        # [BIDx, 256, 4']
        output_smem_cache.axis(1).parallelize(ParallelType.block_x)

        fd.sched.bounded_transform_backward(
            output_smem_cache, -1, [input_smem_cache], propagate_parallel_type=True
        )

        output_smem_cache.axis(2).parallelize(ParallelType.unroll)
        output_reg_cache.axis(2).parallelize(ParallelType.vectorize)
        output_reg_cache.set_allocation_domain(
            output_reg_cache.get_loop_domain(), new_contiguity=True
        )

        # Schedule the memory format for 128 byte swizzle
        # [BIDx, 8', 4', 32]
        input_smem_cache.reorder({3: 1, 1: 2, 2: 3})
        # [BIDx, 32, 8', 4']
        input_smem_cache.split(1, 8)
        # [BIDx, 4, 8, 8', 4']
        input_smem_cache.swizzle(2, 3)
        # [BIDx, 4, 8, 8', 4']
        input_smem_cache.set_allocation_domain(
            input_smem_cache.get_loop_domain(), new_contiguity=True
        )

        input_smem_cache.axis(1).parallelize(ParallelType.tma)
        input_smem_cache.axis(2).parallelize(ParallelType.tma)
        input_smem_cache.axis(3).parallelize(ParallelType.tma)
        input_smem_cache.axis(4).parallelize(ParallelType.tma)
        # [BIDx, Bulk, Bulk, Bulk, Bulk]

        if verbose_:
            print(fd.fusion.print_math())
            print(fd.fusion.print_kernel())

    index32bit = CompileParams(
        index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
    )
    t0 = torch.randn(10000, 10000, dtype=torch.float, device="cuda:0")
    ke = KernelExecutor()
    ke.compile(fd.fusion, [t0], compile_params=index32bit)
    outputs = ke.run([t0])
    assert outputs[0].equal(t0.t())


def test_tutorial_compute_heuristics_and_schedule():
    """
    Demonstrate explicit scheduling: compute_heuristics, modify, then schedule.
    This shows how to customize automatically computed heuristics.
    """
    inputs = [
        torch.randn(4, 4, device="cuda"),
        torch.randn(4, 4, device="cuda"),
    ]

    with FusionDefinition() as fd:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.add(t0, t1)
        t3 = fd.ops.exp(t2)
        fd.add_output(t3)

        # Step 1: Compute heuristics for pointwise scheduler
        heuristic_params = schedule.compute_heuristics(
            fd.fusion, SchedulerType.pointwise, inputs
        )

        # Step 2: Modify the computed heuristics
        # Example: Adjust vectorization and unroll factors
        heuristic_params.vectorization_factor = 1
        heuristic_params.unroll_factor_inner = 2
        print(heuristic_params)

        # Step 3: Apply the schedule using modified heuristics
        schedule.schedule(fd.fusion, SchedulerType.pointwise, heuristic_params)

    # Execute with the modified heuristic params
    nvf_out = fd.manual_execute(inputs, heuristic_params)
    eager_out = torch.exp(inputs[0] + inputs[1])
    torch.testing.assert_close(eager_out, nvf_out[0])


def test_tutorial_pointwise_auto_scheduler():
    """
    Implement a simple pointwise kernel with automatic scheduling.
    Uses nvfuser's PointwiseScheduler.
    """
    inputs = [
        torch.randn(4, 4, device="cuda"),
        torch.randn(4, 4, device="cuda"),
    ]

    with FusionDefinition() as fd:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.add(t0, t1)
        t3 = fd.ops.exp(t2)
        fd.add_output(t3)

        # Apply selected scheduler
        heuristic_params = check_auto_schedule(schedule.schedule)(
            fd.fusion, SchedulerType.pointwise, inputs
        )

    nvf_out = fd.manual_execute(inputs, heuristic_params)
    eager_out = torch.exp(inputs[0] + inputs[1])
    torch.testing.assert_close(eager_out, nvf_out[0])


def test_tutorial_reduction_auto_scheduler():
    """
    Implement a simple reduction kernel with automatic scheduling.
    - Expects failure with PointwiseScheduler
    - Uses nvfuser's ReductionScheduler
    """
    inputs = [
        torch.randn(4, 4, device="cuda"),
    ]

    with FusionDefinition() as fd:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.ops.sum(t0, dims=[1])
        t2 = fd.ops.exp(t1)
        fd.add_output(t2)

        # Test error msg for can_schedule
        pointwise_status, error_msg = schedule.can_schedule(
            fd.fusion, SchedulerType.pointwise, inputs
        )
        assert not pointwise_status
        assert (
            error_msg.strip()
            == "Scheduler _pointwise_ ***rejected*** because : cannot find reference tensor"
        )

        # Apply selected scheduler
        heuristic_params = check_auto_schedule(schedule.schedule)(
            fd.fusion, SchedulerType.reduction, inputs
        )

    nvf_out = fd.manual_execute(inputs, heuristic_params)
    eager_out = torch.exp(inputs[0].sum(1))
    torch.testing.assert_close(eager_out, nvf_out[0])


def test_tutorial_inner_persistent_auto_scheduler():
    """
    Implement a simple normalization kernel with automatic scheduling.
    Uses nvfuser's InnerPersistentScheduler.
    """
    tensor_size = 4
    inputs = [torch.randn(tensor_size, tensor_size, device="cuda")]

    with FusionDefinition() as fd:
        t0 = fd.from_pytorch(inputs[0])
        s0 = fd.define_scalar(1e-6, dtype=DataType.Double)
        norm_const = fd.define_scalar(tensor_size, dtype=DataType.Int)

        bcast_sum0 = fd.ops.sum(t0, dims=[-1], keepdim=True)
        mean = fd.ops.div(bcast_sum0, norm_const)

        diff = fd.ops.sub(t0, mean)
        diff_sq = fd.ops.mul(diff, diff)
        bcast_sum1 = fd.ops.sum(diff_sq, dims=[-1], keepdim=True)
        var = fd.ops.div(bcast_sum1, norm_const)

        t0_diff = fd.ops.sub(t0, mean)
        var_eps = fd.ops.sqrt(fd.ops.add(var, s0))
        t0_norm = fd.ops.div(t0_diff, var_eps)
        fd.add_output(t0_norm)

        # Apply selected scheduler
        heuristic_params = check_auto_schedule(schedule.schedule)(
            fd.fusion, SchedulerType.inner_persistent, inputs
        )

    nvf_out = fd.manual_execute(inputs, heuristic_params)
    var, mean = torch.var_mean(inputs[0], dim=-1, correction=0, keepdim=True)
    eager_out = (inputs[0] - mean) / torch.sqrt(var + 1e-6)
    torch.testing.assert_close(eager_out, nvf_out[0])


def test_tutorial_scheduling_layer_norm_with_profiling():
    """Test and profile layer norm fusion with manual scheduling."""

    def _definition_func(fd: FusionDefinition, inputs, tensor_size):
        """Define the layer norm fusion operations."""
        t0 = fd.from_pytorch(inputs[0])
        s0 = fd.define_scalar(1e-6, dtype=DataType.Double)
        norm_const = fd.define_scalar(tensor_size, dtype=DataType.Int)

        mean_cast = fd.ops.cast(t0, dtype=DataType.Float)
        sum0 = fd.ops.sum(mean_cast, dims=[-1])
        # NOTE Manually broadcast because fusion definition cannot access
        # hidden reduction tensor view.
        bcast_sum0 = fd.ops.broadcast(sum0, [False, True])
        mean = fd.ops.div(bcast_sum0, norm_const)

        var_cast = fd.ops.cast(t0, dtype=DataType.Float)
        diff = fd.ops.sub(var_cast, mean)
        diff_sq = fd.ops.mul(diff, diff)
        sum1 = fd.ops.sum(diff_sq, dims=[-1])
        # NOTE Manually broadcast because fusion definition cannot access
        # hidden reduction tensor view.
        bcast_sum1 = fd.ops.broadcast(sum1, [False, True])
        var = fd.ops.div(bcast_sum1, norm_const)

        t0_cast = fd.ops.cast(t0, dtype=DataType.Float)
        t0_diff = fd.ops.sub(t0_cast, mean)
        var_eps = fd.ops.sqrt(fd.ops.add(var, s0))
        t0_norm = fd.ops.div(t0_diff, var_eps)

        t0_norm_cast = fd.ops.cast(t0_norm, dtype=DataType.BFloat16)
        fd.add_output(t0_norm_cast)

        return t0, mean, t0_norm, sum0, sum1

    def _schedule_func(fd: FusionDefinition, t0, mean, t0_norm, sum0, sum1):
        """Schedule the layer norm fusion."""
        # create cache tensors
        cache_after_t0 = t0.cache_after()
        cache_after_t0.set_memory_type(MemoryType.shared)

        cache_before_t0_norm = t0_norm.cache_before()
        cache_tvs = [cache_after_t0, cache_before_t0_norm]
        print("cache input:\t", cache_after_t0)
        print("cache output:\t", cache_before_t0_norm)

        # Schedule Reference Tensor
        reference_tv = mean
        reference_tv.split(-1, 256 * 4)
        reference_tv.split(-1, 4)
        fd.sched.transform_like(reference_tv)
        print("scheduled reference tensor:\n", reference_tv)

        # Add rfactor TensorViews
        reduction_tvs = [sum0, sum1]
        rfactor_tvs = [tv.rfactor([-1]) for tv in reduction_tvs]

        # Add common parallelization
        reference_tv.axis(0).parallelize(ParallelType.grid_x)
        reference_tv.axis(-2).parallelize(ParallelType.block_x)
        fd.sched.parallelize_like(reference_tv)
        print("parallelized reference tensor:\n", reference_tv)

        # Vectorize input load and output store
        cache_after_t0.axis(-1).parallelize(ParallelType.vectorize)
        t0_norm.axis(-1).parallelize(ParallelType.vectorize)
        print("vectorized input load:\n", cache_after_t0)
        print("vectorized output store:\n", t0_norm)

        # Add computeAt; inline_most automatically skips vectorized iterDomains
        fd.sched.inline_most()

    batch_size = 1024
    tensor_size = 4096
    inputs = [
        torch.randn(batch_size, tensor_size, dtype=torch.bfloat16, device="cuda"),
    ]

    print("\n\n===================== Schedule Layer Norm =========================")

    with FusionDefinition() as fd:
        t0, mean, t0_norm, sum0, sum1 = _definition_func(fd, inputs, tensor_size)
        _schedule_func(fd, t0, mean, t0_norm, sum0, sum1)

    with PythonProfiler(auto_scheduled=False) as prof:
        nvf_out = fd.manual_execute(inputs)

    torch_out = torch.nn.functional.layer_norm(
        inputs[0], normalized_shape=inputs[0].shape[1:]
    )
    print("==================================================================")

    # Change rtol and atol for fp16 dtype
    results_match = torch.allclose(nvf_out[0], torch_out, rtol=1e-2, atol=1e-2)
    assert results_match, "Nvfuser and PyTorch results do not match!"

    print("====================== Profile Kernel =============================")
    assert len(prof.profile.kernel_profiles) == 1
    kp = prof.profile.kernel_profiles[0]

    basic_information = f"Name: {kp.name}, Schedule: {kp.scheduler}, \
    Segment id: {kp.segment_id}, Device: {kp.device}, Stream: {kp.stream}"
    print(basic_information)

    kernel_information = f"Compile time: {kp.compile_time_ms:.2f} ms, \
    Grid: {kp.grid_str}, Block: {kp.block_str}, Registers: {kp.registers}"
    print(kernel_information)

    runtime_information = f"Input size: {kp.input_bytes} bytes, \
    Output size: {kp.output_bytes} bytes, Time: {kp.time_ms:2f} ms"
    print(runtime_information)

    bandwidth_information = f"Effective Bandwidth: {kp.effective_bandwidth_gbs:.2f} GB/s, \
    Peak Bandwidth: {kp.percentage_peak_bandwidth:2f}%"
    print(bandwidth_information)
    print("===================================================================")

    # Validate profiler captured kernel info
    assert prof.profile.fusion_id >= 0
    assert len(prof.profile.kernel_profiles) > 0
    assert prof.profile.kernel_profiles[0].scheduler == "user"


def test_tutorial_autotune_pointwise_mul():
    """
    Test autotuning with machine learning for pointwise multiplication fusion.
    Demonstrates the full workflow: data collection, ML training, and validation.
    """
    # Check if sklearn is available
    pytest.importorskip("sklearn")
    from sklearn import ensemble
    from nvfuser_direct.autotune import (
        ScriptConfiguration,
        collect_data,
        separate_data,
        test_model_rmse,
    )

    class AutotunePointwiseMul:
        """Autotuning configuration for pointwise multiplication fusion."""

        @dataclass(unsafe_hash=True)
        class PointwiseConfiguration:
            break_point: int
            bdim: [int]
            vectorize_factor: int
            outer_unroll: int
            inner_unroll: int

        def __repr__(self):
            return "pointwise_MUL"

        def generate_scheduler_configurations(self, input_shape):
            """Generate all possible scheduler configurations for the given input shape."""

            def _named_product(**items):
                return itertools.starmap(
                    self.PointwiseConfiguration, itertools.product(*items.values())
                )

            num_dimensions = len(input_shape)
            warp_size = 32
            warp_group = warp_size * 4
            # limited to a maximum of 128 threads because of pointwise scheduler
            max_threads_per_cta = 128
            threads_per_cta = list(
                range(warp_group, max_threads_per_cta + 1, warp_group)
            )

            scheduler_configs = []
            for bp in range(num_dimensions):
                for num_threads in threads_per_cta:
                    if bp == 0:
                        # 1D scheduler configurations
                        bdim_shapes = [(num_threads, 1)]
                        outer_unroll_range = [1]
                        # unroll_factor is between [1, 9]
                        inner_unroll_range = range(1, 10)
                    else:
                        # 2D scheduler configurations
                        max_bdimy = num_threads // warp_size
                        log2_max_bdimy = int(math.log2(max_bdimy))
                        bdimy_configs = [
                            2**log_bdimy for log_bdimy in range(1, log2_max_bdimy + 1)
                        ]

                        bdim_shapes = [
                            (max(warp_size, num_threads // bdimy), bdimy)
                            for bdimy in bdimy_configs
                        ]
                        # total_unroll_factor is between [1, 9] given that outer and
                        # inner unroll factors are between [1, 3].
                        outer_unroll_range = range(1, 4)
                        inner_unroll_range = range(1, 4)

                    scheduler_config = _named_product(
                        break_point=[bp],
                        bdim=bdim_shapes,
                        vectorize_factor=[1, 2, 4, 8],
                        outer_unroll=outer_unroll_range,
                        inner_unroll=inner_unroll_range,
                    )
                    scheduler_configs.append(scheduler_config)
            return itertools.chain(*scheduler_configs)

        def create_inputs(self, shape, tensor_datatype):
            """Create input tensors for the MUL fusion."""
            return [
                torch.randn(*shape, dtype=tensor_datatype, device="cuda"),
                torch.randn(*shape, dtype=tensor_datatype, device="cuda"),
            ]

        def create_fusion_func(self, inputs):
            """Create the fusion definition function for MUL."""

            def mul(fd: FusionDefinition) -> None:
                T0 = fd.define_tensor(
                    shape=[-1, -1],
                    contiguity=[True, True],
                    dtype=DataType.BFloat16,
                    is_cpu=False,
                    stride_order=[1, 0],
                )
                T1 = fd.define_tensor(
                    shape=[-1, -1],
                    contiguity=[True, True],
                    dtype=DataType.BFloat16,
                    is_cpu=False,
                    stride_order=[1, 0],
                )
                T2 = fd.ops.cast(T0, dtype=DataType.Float)
                T3 = fd.ops.cast(T1, dtype=DataType.Float)
                T4 = fd.ops.mul(T2, T3)
                T5 = fd.ops.cast(T4, dtype=DataType.BFloat16)
                fd.add_output(T5)

            return mul

        def eager_reference(self, inputs):
            """PyTorch eager mode reference for validation."""
            return inputs[0] * inputs[1]

        def custom_scheduler(self, fd, inputs, scheduler_config):
            """Get heuristic parameters with custom scheduler configuration."""
            # Check if compatible with pointwise scheduler and get default parameters
            pointwise_params = schedule.compute_heuristics(
                fd.fusion, SchedulerType.pointwise, inputs
            )

            # Modify original parameters
            if scheduler_config is not None:
                pointwise_params.break_point = scheduler_config.break_point
                pointwise_params.vectorization_factor = (
                    scheduler_config.vectorize_factor
                )
                pointwise_params.unroll_factor_outer = scheduler_config.outer_unroll
                pointwise_params.unroll_factor_inner = scheduler_config.inner_unroll
                pointwise_params.lparams.bdimx = scheduler_config.bdim[0]
                pointwise_params.lparams.bdimy = scheduler_config.bdim[1]

            # Get base heuristic parameters
            schedule.schedule(fd.fusion, SchedulerType.pointwise, pointwise_params)
            return pointwise_params

    # ====================== Setup Script Configuration  =======================
    script_config = ScriptConfiguration(
        num_dimensions=2,
        outer_shapes=[16384],
        inner_shapes=[128, 1024, 4096, 16384],
        tensor_datatype=torch.bfloat16,
        test_data_percentage=0.1,
        empirical_batch_size=16384,
        empirical_hidden_sizes=list(range(256, 32768, 256)),
    )

    autotune_config = AutotunePointwiseMul()

    # ============================ Run Experiments  ============================
    print("Collecting training data...")
    parameters, performance = collect_data(script_config, autotune_config)
    print(f"Collected {len(parameters)} data points")

    # ============================ Separate Data  ==============================
    train_data, test_data = separate_data(script_config, parameters, performance)
    train_inputs, train_perf = train_data
    test_inputs, test_perf, test_shapes, best_test_scheduler_config = test_data

    print(f"Training set size: {len(train_inputs)}")
    print(f"Test set size: {len(test_inputs)}")

    # ========================= Train Regression Models  =======================
    print("Training RandomForestRegressor...")
    clf = ensemble.RandomForestRegressor()
    clf = clf.fit(train_inputs, train_perf)

    # Verify the model trained successfully
    assert clf is not None
    assert hasattr(clf, "predict")

    # ========================= Test Regression Models  ========================
    print("Testing model performance...")
    test_model_rmse(clf, script_config, autotune_config, test_data)

    # Note: test_model() is commented out as it's very time-consuming
    # and generates plots. Uncomment if needed for detailed analysis:
    # test_model(clf, script_config, autotune_config)

    print("Autotune test completed successfully!")

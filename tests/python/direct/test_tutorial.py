# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
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
)
from nvfuser_direct import idm

from python.direct_utils import (
    is_pre_hopper,
)

verbose_ = True


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

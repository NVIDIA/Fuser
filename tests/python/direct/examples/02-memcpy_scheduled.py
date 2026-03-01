"""
Example: Memory Copy with Manual Scheduling

This example builds on the basic memcpy example by adding manual scheduling
to parallelize the operation across GPU threads and thread blocks.
"""

import torch
from nvfuser_direct import FusionDefinition, ParallelType

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
    print("=== After merge ===")
    print(tv1)

    # Next, we factor out a subdomain for threads in each thread block.
    tv1.split(0, 256)

    # In this case, the flattened domain is now 2D domain with an inner
    # domain of extent 256 and an outer domain of extent i0*i1/256, so the
    # tensor should now look like [i0*i1/256, 256]. Note that in reality we
    # do ceiling division as i0 * i1 may not be divisible by 256.
    print("\n=== After split ===")
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
print("\n=== Fusion Math (Scheduled) ===")
print(fd.fusion.print_math())
# Notice that the axes of tv1 are now printed with blockIdx.x and
# threadIdx.x, which shows they are parallelized by the
# respective parallel types.

# The CUDA kernel should look very differently as there should be no
# for-loops.
print("\n=== Generated Kernel (Scheduled) ===")
print(fd.fusion.print_kernel())

# This time, the kernel is launched with multiple threads and thread
# blocks. Note that the thread block and grid shapes are inferred from the
# given inputs. To see how many threads are used, run this script
# with NVFUSER_DUMP=launch_param
t0 = torch.randn(32, 32, dtype=torch.float, device="cuda:0")
outputs = fd.manual_execute([t0])
assert outputs[0].equal(t0)

print("\n=== Validation Passed ===")
print(f"Input shape: {t0.shape}")
print(f"Output shape: {outputs[0].shape}")
print("Output matches input!")
print("\nTip: Run with NVFUSER_DUMP=launch_param to see thread configuration")


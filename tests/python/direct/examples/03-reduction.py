"""
Example: Reduction Operations

This example demonstrates different ways to parallelize reduction operations
using thread blocks and grid parallelization.
"""

import torch
from nvfuser_direct import FusionDefinition, TensorView, ParallelType


def fusion_func(fd: FusionDefinition) -> TensorView:
    """Common fusion function for all reduction examples"""
    # Create a 2D tensor
    tv0 = fd.define_tensor(shape=[-1, -1])

    # Reduce the second dimension
    tv1 = fd.ops.sum(tv0, dims=[1])
    fd.add_output(tv1)

    return tv1


# Example 1: Block-parallel reduction
print("=" * 60)
print("Example 1: Block-parallel reduction")
print("=" * 60)

with FusionDefinition() as fd1:
    ref_tv = fusion_func(fd1)

    # At this point, nothing is parallelized. The reduction is done by
    # a single thread sequentially.

    # Block-parallel reduction
    ref_tv.axis(1).parallelize(ParallelType.block_x)

    print("\n=== Fusion Math ===")
    print(fd1.fusion.print_math())
    print("\n=== Generated Kernel ===")
    print(fd1.fusion.print_kernel())

t0 = torch.randn(10, 1024, dtype=torch.float, device="cuda:0")
ref = t0.sum(dim=1)

fd1.manual_validate([t0], [ref])
print("\n✓ Validation passed!")


# Example 2: Grid-parallel reduction
print("\n" + "=" * 60)
print("Example 2: Grid-parallel reduction")
print("=" * 60)

with FusionDefinition() as fd2:
    ref_tv = fusion_func(fd2)

    # Use thread blocks for the reduction
    ref_tv.axis(1).parallelize(ParallelType.grid_x)

    print("\n=== Fusion Math ===")
    print(fd2.fusion.print_math())
    print("\n=== Generated Kernel ===")
    print(fd2.fusion.print_kernel())

t0 = torch.randn(10, 1024, dtype=torch.float, device="cuda:0")
ref = t0.sum(dim=1)

fd2.manual_validate([t0], [ref])
print("\n✓ Validation passed!")


# Example 3: Mixed BIDx and TIDx reduction
print("\n" + "=" * 60)
print("Example 3: Mixed grid and block reduction")
print("=" * 60)

with FusionDefinition() as fd3:
    ref_tv = fusion_func(fd3)

    # Mix BIDx and TIDx
    ref_tv.axis(0).parallelize(ParallelType.grid_x)
    ref_tv.axis(1).parallelize(ParallelType.block_x)

    print("\n=== Fusion Math ===")
    print(fd3.fusion.print_math())
    print("\n=== Generated Kernel ===")
    print(fd3.fusion.print_kernel())

# The kernel will be launched with 10 thread blocks, each of which has 1024 threads.
# Run this script with NVFUSER_DUMP=launch_param to see the launch configuration
t0 = torch.randn(10, 1024, dtype=torch.float, device="cuda:0")
ref = t0.sum(dim=1)

fd3.manual_validate([t0], [ref])
print("\n✓ Validation passed!")
print("Tip: Run with NVFUSER_DUMP=launch_param to see launch configuration")

print("\n" + "=" * 60)
print("All reduction examples completed successfully!")
print("=" * 60)


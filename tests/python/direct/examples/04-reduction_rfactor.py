"""
Example: Reduction with Rfactor

This example demonstrates the rfactor transformation for multi-step reductions,
which is a common pattern in CUDA for efficient parallel reductions.
"""

import torch
from nvfuser_direct import FusionDefinition, TensorView, ParallelType


def fusion_func(fd: FusionDefinition) -> TensorView:
    """Simple 1D reduction"""
    tv0 = fd.define_tensor(shape=[-1])
    tv1 = fd.ops.sum(tv0, dims=[0])
    fd.add_output(tv1)
    return tv1


# Example 1: Two-step reduction with rfactor
print("=" * 60)
print("Example 1: Two-step reduction with rfactor")
print("=" * 60)
print("""
A common pattern of reductions in CUDA involves multiple steps of
reductions, where the first step is a per-thread local reduction,
followed by a block reduction of the per-thread partial results.

Conceptual CUDA code:
  // Step 1: Per-thread reduction
  float partial_result = 0;
  for (int i = threadIdx.x; i += blockDim.x; i < N) {
    partial_result += input[i];
  }
  
  // Step 2: Accumulation within each thread block
  __shared__ float shared_buf[blockDim.x];
  shared_buf[threadIdx.x] = partial_result;
  __syncthreads();
  float final_result = 0;
  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x; ++i) {
      final_result += shared_buf[i];
    }
  }
""")

with FusionDefinition() as fd0:
    tv1 = fusion_func(fd0)

    # To reproduce the multi-step reduction pattern in nvFuser, a fusion
    # transformation called reduction rfactor is used. The basic idea is to
    # split a reduction domain such that each of the output domains of the
    # split is separately reduced.

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
    print("\n=== Fusion Math (After rfactor) ===")
    print(fd0.fusion.print_math())

    # To realize the parallelization, we use TIDx for both tv1 and tv2:
    tv1.axis(0).parallelize(ParallelType.block_x)
    tv2.axis(1).parallelize(ParallelType.block_x)

    # At this point, tv2 is a TIDx-parallelized operation of multiple
    # independent reductions. There will be 1024 threads, each of which
    # reduces the first axis of size r1/1024. tv1 is also parallelized by
    # TIDx, but unlike tv2 the reduction domain is parallelized, so it
    # becomes a block-reduction operation.
    print("\n=== Fusion Math (After parallelization) ===")
    print(fd0.fusion.print_math())
    print("\n=== Generated Kernel ===")
    print(fd0.fusion.print_kernel())

# Let's run the scheduled fusion
t0 = torch.randn(10000, dtype=torch.float, device="cuda:0")
ref = t0.sum(dim=0)
fd0.manual_validate([t0], [ref])
print("\n✓ Validation passed!")


# Example 2: Three-step reduction with rfactor
print("\n" + "=" * 60)
print("Example 2: Three-step reduction with rfactor")
print("=" * 60)
print("Extending to three steps for larger data with grid reduction\n")

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

    print("\n=== Fusion Math ===")
    print(fd1.fusion.print_math())
    print("\n=== Generated Kernel ===")
    print(fd1.fusion.print_kernel())

t1 = torch.randn(10000000, dtype=torch.float, device="cuda:0")
ref1 = t1.sum(dim=0)
fd1.manual_validate([t1], [ref1])
print("\n✓ Validation passed!")

print("\n" + "=" * 60)
print("All rfactor reduction examples completed successfully!")
print("=" * 60)


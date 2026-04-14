"""
Example: TMA Bank-Conflict-Free Transpose

This example demonstrates a bank-conflict-free transpose using TMA
with swizzled shared memory layout.

⚠️ Requires: NVIDIA Hopper (SM90) or newer GPU
"""

import torch
from nvfuser_direct import (
    FusionDefinition,
    ParallelType,
    LoadStoreOpType,
    MemoryType,
    DataType,
    CompileParams,
    KernelExecutor,
)

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

print("=== Fusion Math ===")
print(fd.fusion.print_math())
print("\n=== Generated Kernel ===")
print(fd.fusion.print_kernel())

index32bit = CompileParams(
    index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
)
t0 = torch.randn(10000, 10000, dtype=torch.float, device="cuda:0")
ke = KernelExecutor()
ke.compile(fd.fusion, [t0], compile_params=index32bit)
outputs = ke.run([t0])

assert outputs[0].equal(t0.t())
print("\n✓ Validation passed!")
print(f"Input shape: {t0.shape}")
print(f"Output shape (transposed): {outputs[0].shape}")


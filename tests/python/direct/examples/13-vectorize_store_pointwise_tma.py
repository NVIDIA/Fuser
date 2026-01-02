"""
Example: Vectorized Store Pointwise with TMA

This example demonstrates a pointwise operation with TMA loads and
vectorized stores.

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

print("=== Fusion Math ===")
print(fd.fusion.print_math())
print("\n=== Generated Kernel ===")
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
print("\n✓ Validation passed!")
print(f"Input shape: {t0.shape}, {t1.shape}")
print(f"Output shape: {outputs[0].shape}")


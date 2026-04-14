"""
Example: Pointwise with Broadcast and TMA

This example demonstrates a pointwise operation with broadcasting
using TMA for memory operations.

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

print("=== Fusion Math ===")
print(fd.fusion.print_math())
print("\n=== Generated Kernel ===")
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
print("\n✓ Validation passed!")
print(f"Input shape: {t0.shape}, {t1.shape}")
print(f"Output shape: {outputs[0].shape}")


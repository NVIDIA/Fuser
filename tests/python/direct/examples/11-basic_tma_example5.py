"""
Example: Basic TMA - 2D Tiling

Still the same copy kernel of 3D tensor, but this time, we want to do tiling
on the inner two dimensions. The first dimension is treated as a "batch"
dimension. We use CTA tile (64, 64), and TMA tile (32, 32), so we need 4 TMA
instructions to load the entire CTA tile. We want to use two threads, and each
thread issues two TMA instructions.

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

print("=== Fusion Math ===")
print(fd.fusion.print_math())
print("\n=== Generated Kernel ===")
print(fd.fusion.print_kernel())
print("""
TMA will be generated like:
Note that the coordinate is in number of items, smem address is in bytes.
Also note that coordinate is in column major, so inner dims goes first

for (nvfuser_index_t i13 = 0; i13 < 2; ++i13) {
  if (threadIdx.x < 2) {
    Hopper::cpAsyncBulkTensorTileG2S(
        coordinate =
            {64 * blockIdx.z + 32 * i13,
             64 * blockIdx.y + 32 * threadIdx.x,
             blockIdx.x},
        smem_addr = toSmem(T2) + 8192 * threadIdx.x + 4096 * i13);
  }
}
""")

index32bit = CompileParams(
    index_type=DataType.Int32, maxrregcount=255, enable_magic_zero=False
)
t0 = torch.randn(5, 3, 300, dtype=torch.float, device="cuda:0")
ke = KernelExecutor()
ke.compile(fd.fusion, [t0], compile_params=index32bit)
outputs = ke.run([t0])

assert outputs[0].equal(t0)
print("\n✓ Validation passed!")
print(f"Input shape: {t0.shape}")
print(f"Output shape: {outputs[0].shape}")


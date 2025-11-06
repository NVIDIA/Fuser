"""
Example: Basic TMA - 2D TMA Store

Similar to example 5, but we are using TMA for store instead of load.

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

print("=== Fusion Math ===")
print(fd.fusion.print_math())
print("\n=== Generated Kernel ===")
print(fd.fusion.print_kernel())
print("""
TMA will be generated like:
Note that the coordinate is in number of items, smem address is in bytes.
Also note that coordinate is in column major, so inner dims goes first

for (nvfuser_index_t i19 = 0; i19 < 2; ++i19) {
  if (threadIdx.x < 2) {
    Hopper::cpAsyncBulkTensorTileS2G(
        coordinate =
            {64 * blockIdx.z + 32 * i19,
             64 * blockIdx.y + 32 * threadIdx.x,
             blockIdx.x},
        smem_addr = toSmem(T2) + 8192 * threadIdx.x + 4096 * i19);
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


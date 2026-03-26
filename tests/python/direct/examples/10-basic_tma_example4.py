"""
Example: Basic TMA - Store Operations

Similar to example 3, except that we are using TMA for store
instead of load.

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

print("=== Fusion Math ===")
print(fd.fusion.print_math())
print("\n=== Generated Kernel ===")
print(fd.fusion.print_kernel())
print("""
TMA will be generated like:
Note that the coordinate is in number of items, smem address is in bytes

if (threadIdx.x < 4) {
  Hopper::cpAsyncBulkTensorTileS2G(
      coordinate = {1024 * blockIdx.x + 256 * threadIdx.x},
      smem_addr = (toSmem(T2) + 1024 * threadIdx.x));
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


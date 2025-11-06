"""
Example: Basic TMA - Multiple 1D Instructions (For Loop)

Similar to example 1, we treat the fusion as 1D and use 1D TMA to load
data to shared memory. But this time, instead of using 1 TMA instruction
to load the entire CTA tile, we use 4 TMA instructions. We use a for loop
to launch these 4 instructions.

CTA tile size = 4 * TMA tile size = 1024

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

print("=== Fusion Math ===")
print(fd.fusion.print_math())
print("\n=== Generated Kernel ===")
print(fd.fusion.print_kernel())
print("""
TMA will be generated like:
Note that the coordinate is in number of items, smem address is in bytes

for (nvfuser_index_t i8 = 0; i8 < 4; ++i8) {
  if (threadIdx.x == 0) {
    Hopper::cpAsyncBulkTensorTileG2S(
        coordinate = {1024 * blockIdx.x + 256 * i8},
        smem_addr = (toSmem(T2) + 1024 * i8));
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


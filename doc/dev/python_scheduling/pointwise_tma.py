# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

# Description: Schedule pointwise multiplication with TMA loads using nvfuser scheduling primitives
# Converted from C++ test: PointwiseTest.PointwiseMulMultiWaveTMA

import torch
from nvfuser import (
    FusionDefinition,
    DataType,
    ParallelType,
    MemoryType,
    LoadStoreOpType,
)

dim0, dim1 = 8192, 8192
dtype = DataType.BFloat16

inputs = [
    torch.randn(dim0, dim1, dtype=torch.bfloat16, device="cuda"),
    torch.randn(dim0, dim1, dtype=torch.bfloat16, device="cuda"),
]


class PointwiseMulDefault(FusionDefinition):
    """Pointwise multiplication using default auto-scheduler"""

    def definition(self):
        self.t0 = self.from_pytorch(inputs[0])
        self.t1 = self.from_pytorch(inputs[1])

        # Data type conversion and computation
        self.t0_float = self.ops.cast(self.t0, dtype=DataType.Float)
        self.t1_float = self.ops.cast(self.t1, dtype=DataType.Float)
        self.t2 = self.ops.mul(self.t0_float, self.t1_float)
        self.t3 = self.ops.cast(self.t2, dtype=dtype)

        # Output
        self.add_output(self.t3)

    # No schedule() method - nvfuser will use auto-scheduler


class PointwiseMulTMA(FusionDefinition):
    """Pointwise multiplication with manual TMA schedule"""

    def __init__(self, tma_m=128, tma_n=128, blk_m=8, blk_n=16, tid_m=2):
        super().__init__()
        self.tma_tile_m = tma_m
        self.tma_tile_n = tma_n
        self.blk_tile_m = blk_m
        self.blk_tile_n = blk_n
        self.tid_tile_m = tid_m

    def definition(self):
        self.t0 = self.from_pytorch(inputs[0])
        self.t1 = self.from_pytorch(inputs[1])

        # Data type conversion and computation
        self.t0_float = self.ops.cast(self.t0, dtype=DataType.Float)
        self.t1_float = self.ops.cast(self.t1, dtype=DataType.Float)
        self.t2 = self.ops.mul(self.t0_float, self.t1_float)
        self.t3 = self.ops.cast(self.t2, dtype=dtype)

        # Output
        self.add_output(self.t3)

    def schedule(self):
        # Tile sizes
        vect_factor = 128 // 16  # 128 bits / 16 bits per bfloat16 = 8
        tma_tile_m, tma_tile_n = self.tma_tile_m, self.tma_tile_n
        blk_tile_m, blk_tile_n = self.blk_tile_m, self.blk_tile_n
        tid_tile_m, tid_tile_n = self.tid_tile_m, vect_factor  # [Unroll, Vectorize]

        # Create TMA loads from inputs to shared memory
        self.t0_smem = self.sched.cache_after(self.t0, LoadStoreOpType.tma)
        self.sched.set_memory_type(self.t0_smem, MemoryType.shared)

        self.t1_smem = self.sched.cache_after(self.t1, LoadStoreOpType.tma)
        self.sched.set_memory_type(self.t1_smem, MemoryType.shared)

        # Cache loads from shared memory to registers (vectorized)
        self.t0_reg = self.sched.cache_after(self.t0_smem)
        self.t1_reg = self.sched.cache_after(self.t1_smem)

        # Cache output for vectorized store
        self.t3_cache = self.sched.cache_before(self.t3)

        # Schedule TMA load tensors
        tma_tvs = [self.t0_smem, self.t1_smem]
        for tv in tma_tvs:
            # [I0, I1] -> [I0/m, m, I1/n, n]
            self.sched.split(tv, dim=0, factor=tma_tile_m)
            self.sched.split(tv, dim=-1, factor=tma_tile_n)
            # [I0/m, m, I1/n, n] --> [I0/m, I1/n, m, n]
            self.sched.reorder(tv, {1: 2})
            # Parallelize
            self.sched.parallelize(tv, axis=0, parallel_type=ParallelType.grid_y)
            self.sched.parallelize(tv, axis=1, parallel_type=ParallelType.grid_x)
            self.sched.parallelize(tv, axis=2, parallel_type=ParallelType.tma)
            self.sched.parallelize(tv, axis=3, parallel_type=ParallelType.tma)

        # Schedule all compute tensors (registers and output)
        compute_tvs = [
            self.t0_reg,
            self.t1_reg,
            self.t0_float,
            self.t1_float,
            self.t2,
            self.t3_cache,
            self.t3,
        ]

        for tv in compute_tvs:
            # [I0, I1] -> [I0/m, m, I1/n, n]
            self.sched.split(tv, dim=0, factor=tma_tile_m)
            self.sched.split(tv, dim=-1, factor=tma_tile_n)
            # [I0/m, m, I1/n, n] -> [I0/m, m/u, u, I1/n, n/v, v]
            self.sched.split(tv, dim=1, factor=tid_tile_m)
            self.sched.split(tv, dim=-1, factor=tid_tile_n)
            # [I0/m, m/u, u, I1/n, n/v, v] -> [I0/m, m/u/y, y, u, I1/n, n/v/x, x, v]
            self.sched.split(tv, dim=1, factor=blk_tile_m)
            self.sched.split(tv, dim=-2, factor=blk_tile_n)
            # [I0/m, m/u/y, y, u, I1/n, n/v/x, x, v] -> [I0/m, I1/n, m/u/y, n/v/x, y, x, u, v]
            self.sched.reorder(tv, {1: 2, 2: 4, 3: 6, 4: 1, 5: 3, 6: 5})

            # Parallelize
            self.sched.parallelize(tv, axis=0, parallel_type=ParallelType.grid_y)
            self.sched.parallelize(tv, axis=1, parallel_type=ParallelType.grid_x)
            self.sched.parallelize(tv, axis=4, parallel_type=ParallelType.block_y)
            self.sched.parallelize(tv, axis=5, parallel_type=ParallelType.block_x)
            self.sched.parallelize(tv, axis=6, parallel_type=ParallelType.unroll)
            # Only vectorize the cache tensors (loads and stores)
            if tv == self.t0_reg or tv == self.t1_reg or tv == self.t3:
                self.sched.parallelize(tv, axis=7, parallel_type=ParallelType.vectorize)

        # Inline most tensors
        self.sched.inline_most()


def print_kernel_profile(kp):
    basic_information = f"name: {kp.name}, schedule: {kp.scheduler}, segment_id: {kp.segment_id}, device: {kp.device}, stream: {kp.stream}"
    print(basic_information)

    kernel_information = f"compile time: {kp.compile_time_ms:.2f} ms, grid: {kp.grid_str}, block: {kp.block_str}, registers: {kp.registers}"
    print(kernel_information)

    runtime_information = f"input size: {kp.input_bytes} bytes, output size: {kp.output_bytes} bytes, time: {kp.time_ms:.2f} ms"
    print(runtime_information)

    bandwidth_information = f"Effective Bandwidth: {kp.effective_bandwidth_gbs:.2f} GB/s, Peak Bandwidth: {kp.percentage_peak_bandwidth:.2f}%"
    print(bandwidth_information)
    print()


# Compute PyTorch reference
torch_out = inputs[0].float() * inputs[1].float()
torch_out = torch_out.to(torch.bfloat16)


# ============================================================================================================
# Run with Default Scheduler
# ============================================================================================================
print("\n\n" + "=" * 110)
print("DEFAULT SCHEDULER (Auto)")
print("=" * 110)

fn_default = PointwiseMulDefault()
nvf_out_default = fn_default.execute(inputs, profile=True)

print(
    f"Results match PyTorch: {torch.allclose(nvf_out_default[0], torch_out, rtol=1e-2, atol=1e-2)}"
)

print("\n--- Kernel Profile ---")
kps_default = fn_default.profile().kernel_profiles
for kp in kps_default:
    print_kernel_profile(kp)


# ============================================================================================================
# Auto-tune: Test Different Tile Configurations
# ============================================================================================================
print("\n" + "=" * 110)
print("AUTO-TUNING TMA SCHEDULER")
print("=" * 110)

# Define search space
tma_tiles = [64, 128, 256]
blk_tiles = [(8, 16), (8, 32), (16, 16)]
tid_tiles_m = [1, 2, 4]

results = []
best_time = float("inf")
best_config = None

print(
    f"\nTesting {len(tma_tiles) * len(tma_tiles) * len(blk_tiles) * len(tid_tiles_m)} configurations..."
)
print(f"{'Config':<40} {'Time (ms)':<12} {'Bandwidth (GB/s)':<20} {'Status'}")
print("-" * 110)

config_count = 0
for tma_m in tma_tiles:
    for tma_n in tma_tiles:
        for blk_m, blk_n in blk_tiles:
            for tid_m in tid_tiles_m:
                config_count += 1
                config_str = f"TMA={tma_m}x{tma_n}, BLK={blk_m}x{blk_n}, TID={tid_m}"

                try:
                    # Create fusion with specific tile configuration
                    fn_tma = PointwiseMulTMA(
                        tma_m=tma_m, tma_n=tma_n, blk_m=blk_m, blk_n=blk_n, tid_m=tid_m
                    )
                    nvf_out_tma = fn_tma.execute(inputs, profile=True)

                    # Verify correctness
                    if not torch.allclose(
                        nvf_out_tma[0], torch_out, rtol=1e-2, atol=1e-2
                    ):
                        print(f"{config_str:<40} {'N/A':<12} {'N/A':<20} INCORRECT")
                        continue

                    # Get performance metrics
                    kps_tma = fn_tma.profile().kernel_profiles
                    tma_time = sum(kp.time_ms for kp in kps_tma)
                    tma_bw = (
                        sum(kp.effective_bandwidth_gbs for kp in kps_tma) / len(kps_tma)
                        if kps_tma
                        else 0
                    )

                    results.append(
                        {
                            "config": config_str,
                            "tma_m": tma_m,
                            "tma_n": tma_n,
                            "blk_m": blk_m,
                            "blk_n": blk_n,
                            "tid_m": tid_m,
                            "time": tma_time,
                            "bandwidth": tma_bw,
                            "kps": kps_tma,
                        }
                    )

                    # Track best configuration
                    status = ""
                    if tma_time < best_time:
                        best_time = tma_time
                        best_config = results[-1]
                        status = "← BEST"

                    print(
                        f"{config_str:<40} {tma_time:<12.4f} {tma_bw:<20.2f} {status}"
                    )

                except Exception as e:
                    print(f"{config_str:<40} {'FAILED':<12} {'N/A':<20} {str(e)[:30]}")
                    continue

print("-" * 110)
print(f"\nTested {len(results)}/{config_count} configurations successfully\n")


# ============================================================================================================
# Best Configuration Results
# ============================================================================================================
if best_config:
    print("\n" + "=" * 110)
    print("BEST MANUAL TMA CONFIGURATION")
    print("=" * 110)
    print(f"Configuration: {best_config['config']}")
    print(f"Time: {best_config['time']:.4f} ms")
    print(f"Bandwidth: {best_config['bandwidth']:.2f} GB/s")
    print("Results match PyTorch: True")

    print("\n--- Kernel Profile ---")
    for kp in best_config["kps"]:
        print_kernel_profile(kp)


# ============================================================================================================
# Performance Comparison
# ============================================================================================================
print("\n" + "=" * 110)
print("PERFORMANCE COMPARISON")
print("=" * 110)

default_time = sum(kp.time_ms for kp in kps_default)
default_bw = (
    sum(kp.effective_bandwidth_gbs for kp in kps_default) / len(kps_default)
    if kps_default
    else 0
)

print("Default Auto-Scheduler:")
print(f"  Time:      {default_time:.4f} ms")
print(f"  Bandwidth: {default_bw:.2f} GB/s")

if best_config:
    print(f"\nBest Manual TMA ({best_config['config']}):")
    print(f"  Time:      {best_config['time']:.4f} ms")
    print(f"  Bandwidth: {best_config['bandwidth']:.2f} GB/s")
    print(f"\nSpeedup: {default_time / best_config['time']:.2f}x")

    # Show top 5 configurations
    print("\n--- Top 5 Configurations (single run) ---")
    sorted_results = sorted(results, key=lambda x: x["time"])[:5]
    for i, result in enumerate(sorted_results, 1):
        print(
            f"{i}. {result['config']}: {result['time']:.4f} ms ({result['bandwidth']:.2f} GB/s)"
        )

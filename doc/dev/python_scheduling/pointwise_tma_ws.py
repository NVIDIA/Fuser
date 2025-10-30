# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

# Description: Schedule pointwise multiplication with TMA loads using nvfuser scheduling primitives
# Converted from C++ test: PointwiseTest.PointwiseMulMultiWaveTMA

import os
import torch


from nvfuser import (
    FusionDefinition,
    DataType,
    ParallelType,
    MemoryType,
    LoadStoreOpType,
    WarpSpecialized,
)

dim0, dim1 = 8192, 8192
dtype = DataType.BFloat16

# clear l2 cache

# Number of streaming multiprocessors (always use actual device SM count)
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
l2_cache_size = torch.cuda.get_device_properties(0).L2_cache_size
device_smem_bytes = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
print(f"L2 cache size: {l2_cache_size} bytes")
print(f"Device SMEM bytes: {device_smem_bytes} bytes")
# run with NVFUSER_DISALBE=kernel_reuse
os.environ["NVFUSER_DISABLE"] = "kernel_reuse"


def clear_l2_cache():
    cache_clear_size = l2_cache_size // 4
    dummy = torch.empty(cache_clear_size, dtype=torch.float32, device="cuda")
    dummy.fill_(0.0)
    torch.cuda.synchronize()


inputs = [
    torch.randn(dim0, dim1, dtype=torch.bfloat16, device="cuda"),
    torch.randn(dim0, dim1, dtype=torch.bfloat16, device="cuda"),
]

has_tanh = False

explicit_unroll = False


class PointwiseMulDefault(FusionDefinition):
    """Pointwise multiplication using default auto-scheduler"""

    def definition(self):
        # self.t0 = self.from_pytorch(inputs[0])
        # self.t1 = self.from_pytorch(inputs[1])
        self.t0 = self.define_tensor(
            sizes=[dim0, dim1], strides=[dim1, 1], dtype=dtype, static_sizes=True
        )
        self.t1 = self.define_tensor(
            sizes=[dim0, dim1], strides=[dim1, 1], dtype=dtype, static_sizes=True
        )
        # Data type conversion and computation
        self.t0_float = self.ops.cast(self.t0, dtype=DataType.Float)
        self.t1_float = self.ops.cast(self.t1, dtype=DataType.Float)
        self.t2_mul = self.ops.mul(self.t0_float, self.t1_float)
        if has_tanh:
            self.t2 = self.ops.tanh(self.t2_mul)
            self.t3 = self.ops.cast(self.t2, dtype=dtype)
        else:
            self.t3 = self.ops.cast(self.t2_mul, dtype=dtype)

        # Output
        self.add_output(self.t3)

    # No schedule() method - nvfuser will use auto-scheduler


class PointwiseMulTMA(FusionDefinition):
    """Pointwise multiplication with manual TMA schedule following TMP3"""

    def __init__(
        self,
        tma_m=32,
        tma_n=256,
        tid_m=4,
        tid_n=8,
        blk_n=256,
        fully_reg_cached=True,
        number_of_stages=4,
        use_tma_store=False,
    ):
        super().__init__()
        self.tma_tile_m = tma_m
        self.tma_tile_n = tma_n
        self.tid_tile_m = tid_m  # Unroll
        self.tid_tile_n = tid_n  # Vectorize
        self.blk_tile_n = blk_n  # TIDx
        self.fully_reg_cached = fully_reg_cached
        self.number_of_stages = number_of_stages
        self.use_tma_store = use_tma_store

    def definition(self):
        # Define tensors with concrete static shapes instead of symbolic
        self.t0 = self.define_tensor(
            sizes=[dim0, dim1], strides=[dim1, 1], dtype=dtype, static_sizes=True
        )
        self.t1 = self.define_tensor(
            sizes=[dim0, dim1], strides=[dim1, 1], dtype=dtype, static_sizes=True
        )

        # Data type conversion and computation
        self.t0_float = self.ops.cast(self.t0, dtype=DataType.Float)
        self.t1_float = self.ops.cast(self.t1, dtype=DataType.Float)
        self.t2_mul = self.ops.mul(self.t0_float, self.t1_float)
        if has_tanh:
            self.t2 = self.ops.tanh(self.t2_mul)
            self.t3 = self.ops.cast(self.t2, dtype=dtype)
        else:
            self.t3 = self.ops.cast(self.t2_mul, dtype=dtype)

        # Output
        self.add_output(self.t3)

    def schedule(self):
        # Tile sizes following TMP3
        tma_tile_m, tma_tile_n = self.tma_tile_m, self.tma_tile_n
        tid_tile_m, tid_tile_n = self.tid_tile_m, self.tid_tile_n  # [Unroll, Vectorize]
        blk_tile_n = self.blk_tile_n  # TIDx

        # Create TMA loads from inputs to shared memory
        self.t0_smem = self.sched.cache_after(self.t0, LoadStoreOpType.tma)
        self.sched.set_memory_type(self.t0_smem, MemoryType.shared)

        self.t1_smem = self.sched.cache_after(self.t1, LoadStoreOpType.tma)
        self.sched.set_memory_type(self.t1_smem, MemoryType.shared)

        # Cache loads from shared memory to registers (vectorized)
        self.t0_reg = self.sched.cache_after(self.t0_smem)
        self.t1_reg = self.sched.cache_after(self.t1_smem)

        # Output caching: regs -> [smem ->] global memory
        if self.use_tma_store:
            # TMA store path: regs -> smem -> global memory (via TMA)
            self.t3_smem = self.sched.cache_before(self.t3, LoadStoreOpType.tma)
            self.sched.set_memory_type(self.t3_smem, MemoryType.shared)
            self.t3_regs = self.sched.cache_before(self.t3_smem)
        else:
            # Regular store path: regs -> global memory (no TMA)
            self.t3_regs = self.sched.cache_before(self.t3)
            self.t3_smem = None

        # Use t3 as reference for scheduling (following TMP3)
        reference = self.t3

        # [I0, I1] -> [I0/m, m, I1/n, n]
        self.sched.split(reference, dim=0, factor=tma_tile_m)
        self.sched.split(reference, dim=-1, factor=tma_tile_n)
        # [I0/m, m, I1/n, n] -> [I0/m, I1/n, m, n]
        self.sched.reorder(reference, {1: 2})
        # [I0/m, I1/n, m, n] -> [I/sm, sm, m, n]
        self.sched.merge(reference, dim=0)  # Merge I0/m and I1/n
        self.sched.split(reference, dim=0, factor=num_sms)
        # [I/sm, sm, m, n] -> [sm, I/sm, m, n]
        self.sched.reorder(reference, {0: 1})

        # Propagate transform to all tensors
        self.sched.transform_like(reference)

        # Apply inlineAt for TMA cache (pos=2 corresponds to after [sm, I/sm])
        self.sched.inline_at(reference, pos=2)

        # Schedule all compute tensors (registers and output)
        compute_tvs = [
            self.t0_reg,
            self.t1_reg,
            self.t0_float,
            self.t1_float,
            self.t2_mul,
            self.t3_regs,
        ]
        if has_tanh:
            compute_tvs.insert(-1, self.t2)  # Insert t2 (tanh result) before t3_regs

        # Add t3_smem if using TMA store, otherwise add t3 (output tensor)
        if self.use_tma_store:
            compute_tvs.append(self.t3_smem)
        else:
            # For regular store, schedule the output tensor
            compute_tvs.append(self.t3)

        for tv in compute_tvs:
            # [..., m, n] -> [..., mn]
            self.sched.merge(tv, dim=-2)
            # [..., mn] -> [..., mn/v, v]
            self.sched.split(tv, dim=-1, factor=tid_tile_n)
            # [..., mn/v, v] -> [..., mn/v/x, x, v]
            self.sched.split(tv, dim=-2, factor=blk_tile_n)
            # [..., mn/v/x, x, v] -> [..., mn/v/x/u, u, x, v]
            self.sched.split(tv, dim=-3, factor=tid_tile_m)

            # Parallelize
            # [sm, I/sm, ...]
            self.sched.parallelize(
                tv, axis=0, parallel_type=ParallelType.grid_x
            )  # BIDx
            self.sched.parallelize(
                tv, axis=-2, parallel_type=ParallelType.block_x
            )  # TIDx
            if explicit_unroll:
                self.sched.parallelize(tv, axis=-3, parallel_type=ParallelType.unroll)
            # Vectorize: register cache tensors for loads from smem, and smem tensor for TMA store
            vectorize_condition = (
                tv == self.t0_reg
                or tv == self.t1_reg
                or (self.use_tma_store and tv == self.t3_smem)
                or (not self.use_tma_store and tv == self.t3)
            )
            if vectorize_condition:
                self.sched.parallelize(
                    tv, axis=-1, parallel_type=ParallelType.vectorize
                )

        # Inline selected tensors for register caching
        if not self.fully_reg_cached:
            self.sched.inline_most(compute_tvs)
        else:
            # Inline register caches at position 2
            self.sched.inline_at(
                self.t0_reg, pos=2, selected_tensors=[self.t0_reg, self.t1_reg]
            )
            inline_most_tvs = [
                self.t0_float,
                self.t1_float,
                self.t2_mul,
                self.t3_regs,
            ]
            if has_tanh:
                inline_most_tvs.insert(-1, self.t2)  # Insert t2 (tanh) before t3_regs
            self.sched.inline_most(inline_most_tvs)

        # Circular Buffer with TMA loads (warp specialization)
        # register sharing strategy:
        # (1) blk_tile_n == 128 -> no register sharing
        # (2) blk_tile_n == 256 -> 168 registers -> register sharing: [40, 232]
        # (3) blk_tile_n == 384 -> 128 registers -> register sharing: [32, 160]
        # (4) blk_tile_n == 512 -> 96  registers -> register sharing: [32, 112]
        # (5) blk_tile_n == 640 -> 80  registers -> register sharing: [40, 88]
        # (6) blk_tile_n == 768 -> 72  registers -> register sharing: [24, 80]
        # (7) blk_tile_n == 896 -> 64  registers -> register sharing: [64, 64]
        if blk_tile_n == 128:
            circular_buffer_type = WarpSpecialized(
                ParallelType.block_x
            )  # No register sharing
        elif blk_tile_n == 256:
            circular_buffer_type = WarpSpecialized(
                ParallelType.block_x, (40, 232)
            )  # TIDx with register sharing
        elif blk_tile_n == 384:
            circular_buffer_type = WarpSpecialized(
                ParallelType.block_x, (32, 160)
            )  # TIDx with register sharing
        elif blk_tile_n == 512:
            circular_buffer_type = WarpSpecialized(
                ParallelType.block_x, (32, 112)
            )  # TIDx with register sharing
        elif blk_tile_n == 640:
            circular_buffer_type = WarpSpecialized(
                ParallelType.block_x, (40, 88)
            )  # TIDx with register sharing
        elif blk_tile_n == 768:
            circular_buffer_type = WarpSpecialized(
                ParallelType.block_x, (24, 80)
            )  # TIDx with register sharing
        elif blk_tile_n == 896:
            circular_buffer_type = WarpSpecialized(
                ParallelType.block_x
            )  # TIDx with register sharing

        number_of_stages = self.number_of_stages
        prefetch_distance = number_of_stages - 1

        # TMA loads - add to circular buffer
        tma_load_tvs = [self.t0_smem, self.t1_smem]
        for tv in tma_load_tvs:
            self.sched.parallelize(
                tv, axis=0, parallel_type=ParallelType.grid_x
            )  # BIDx
            self.sched.parallelize(tv, axis=-1, parallel_type=ParallelType.tma)  # Bulk
            self.sched.parallelize(tv, axis=-2, parallel_type=ParallelType.tma)  # Bulk
            self.sched.circular_buffer(
                tv, number_of_stages, prefetch_distance, circular_buffer_type
            )

        # TMA store - parallelize output tensor (t3) for gmem (only if use_tma_store)
        if self.use_tma_store:
            self.sched.parallelize(
                self.t3, axis=0, parallel_type=ParallelType.grid_x
            )  # BIDx
            self.sched.parallelize(
                self.t3, axis=-1, parallel_type=ParallelType.tma
            )  # Bulk
            self.sched.parallelize(
                self.t3, axis=-2, parallel_type=ParallelType.tma
            )  # Bulk


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
if has_tanh:
    torch_out = torch.tanh(torch_out)
torch_out = torch_out.to(torch.bfloat16)


# ============================================================================================================
# Run with Default Scheduler
# ============================================================================================================
if False:
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

# if False:
if True:
    # debug run
    print("=" * 110)
    print("DEBUG RUN - Testing with and without TMA Store")
    print("=" * 110)

    for use_tma_store in [False]:
        # clear_l2_cache()
        store_mode = (
            "WITH TMA Store" if use_tma_store else "WITHOUT TMA Store (Regular Store)"
        )
        print(f"\n{store_mode}")
        print("-" * 110)

        fn_tma = PointwiseMulTMA(
            tma_m=64,
            tma_n=256,
            tid_m=2,
            tid_n=8,
            blk_n=512,
            fully_reg_cached=False,
            number_of_stages=3,
            use_tma_store=use_tma_store,
        )
        nvf_out_tma = fn_tma.execute(inputs, profile=True)
        assert torch.allclose(nvf_out_tma[0], torch_out, rtol=1e-2, atol=1e-2)

        kps_tma = fn_tma.profile().kernel_profiles
        for kp in kps_tma:
            print_kernel_profile(kp)

    exit()

# ============================================================================================================
# Auto-tune: Test Different Tile Configurations
# ============================================================================================================
print("\n" + "=" * 110)
print("AUTO-TUNING TMA SCHEDULER")
print("=" * 110)

# On Blackwell, needs 64KB of data in flight to saturate memory bandwidth.
# For this case, we have 2 inputs of bfloat16, they are 2 x 2 = 4 Bytes.
# Needs to load 64KB / 4B = 16K elements.
# smem = 232 KB / 64KB = 3.625 --> 3 loads --> 232 / 3 = 77.33 KB -> 77.33 KB / 4B = 19332.5 --> 19332 elements
# 16384 = 64 x 256 = 128 * 128
tma_tiles = [(64, 256), (128, 128), (64, 128), (64, 64), (32, 64), (32, 32)]
tid_tiles = [(1, 8), (2, 8), (4, 8)]
blk_tiles_n = [256, 512]
fully_reg_cached_options = [True, False]
number_of_stages_options = [3]  # use max allowed by SMEM
use_tma_store_options = [True]  # Test with and without TMA store


def not_enough_smem(tma_m, tma_n, n_stages):
    n_input = 2
    bytes_per_element = 2
    return tma_m * tma_n * n_stages * n_input * bytes_per_element > device_smem_bytes


results = []
best_time = float("inf")
best_config = None

total_configs = (
    len(tma_tiles)
    * len(tid_tiles)
    * len(blk_tiles_n)
    * len(fully_reg_cached_options)
    * len(number_of_stages_options)
    * len(use_tma_store_options)
)
print(f"\nTesting {total_configs} configurations...")
print(f"{'Config':<70} {'Time (ms)':<12} {'Bandwidth (GB/s)':<20} {'Status'}")
print("-" * 110)

config_count = 0
for tma_m, tma_n in tma_tiles:
    for tid_m, tid_n in tid_tiles:
        for blk_n in blk_tiles_n:
            if blk_n * tid_m * tid_n > tma_m * tma_n:
                continue
            for fully_reg_cached in fully_reg_cached_options:
                for n_stages in number_of_stages_options:
                    for use_tma_store in use_tma_store_options:
                        if use_tma_store:
                            smem_for_load = device_smem_bytes - tma_m * tma_n * 2
                        else:
                            smem_for_load = device_smem_bytes
                        smem_for_load = (
                            smem_for_load - 4096
                        )  # overhhead for mbarriers, kernel launch, etc.
                        n_stages = smem_for_load // (tma_m * tma_n * 2 * 2)
                        if not_enough_smem(tma_m, tma_n, n_stages):
                            print(f"Not enough SMEM for {tma_m}x{tma_n}x{n_stages}")
                            continue
                        config_count += 1
                        reg_mode = (
                            "FullyRegCached"
                            if fully_reg_cached
                            else "NotFullyRegCached"
                        )
                        store_mode = "TMAStore" if use_tma_store else "RegStore"
                        config_str = f"TMA={tma_m}x{tma_n}, TID={tid_m}x{tid_n}, BLK={blk_n}, Stages={n_stages}, {reg_mode}, {store_mode}"

                        try:
                            clear_l2_cache()
                            # Create fusion with specific tile configuration
                            fn_tma = PointwiseMulTMA(
                                tma_m=tma_m,
                                tma_n=tma_n,
                                tid_m=tid_m,
                                tid_n=tid_n,
                                blk_n=blk_n,
                                fully_reg_cached=fully_reg_cached,
                                number_of_stages=n_stages,
                                use_tma_store=use_tma_store,
                            )
                            nvf_out_tma = fn_tma.execute(inputs, profile=True)

                            # Verify correctness
                            if not torch.allclose(
                                nvf_out_tma[0], torch_out, rtol=1e-2, atol=1e-2
                            ):
                                print(
                                    f"{config_str:<70} {'N/A':<12} {'N/A':<20} INCORRECT"
                                )
                                continue

                            # Get performance metrics
                            kps_tma = fn_tma.profile().kernel_profiles
                            tma_time = sum(kp.time_ms for kp in kps_tma)
                            tma_bw = (
                                sum(kp.effective_bandwidth_gbs for kp in kps_tma)
                                / len(kps_tma)
                                if kps_tma
                                else 0
                            )

                            results.append(
                                {
                                    "config": config_str,
                                    "tma_m": tma_m,
                                    "tma_n": tma_n,
                                    "tid_m": tid_m,
                                    "tid_n": tid_n,
                                    "blk_n": blk_n,
                                    "number_of_stages": n_stages,
                                    "fully_reg_cached": fully_reg_cached,
                                    "use_tma_store": use_tma_store,
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
                                f"{config_str:<70} {tma_time:<12.4f} {tma_bw:<20.2f} {status}"
                            )

                        except Exception as e:
                            print(
                                f"{config_str:<70} {'FAILED':<12} {'N/A':<20} {str(e)[:30]}"
                            )
                            exit()

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

    # Show top 10 configurations
    print("\n--- Top 10 Configurations (single run) ---")
    sorted_results = sorted(results, key=lambda x: x["time"])[:10]
    for i, result in enumerate(sorted_results, 1):
        print(
            f"{i}. {result['config']}: {result['time']:.4f} ms ({result['bandwidth']:.2f} GB/s)"
        )

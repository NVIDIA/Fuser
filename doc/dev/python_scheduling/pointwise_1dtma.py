# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

# Description: Schedule pointwise multiplication with 1D TMA loads using nvfuser scheduling primitives
# Converted from C++ test: PointwiseTest.PointwiseMulMultiWave1dTMA
import os
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

has_tanh = False

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


class PointwiseMulDefault(FusionDefinition):
    """Pointwise multiplication using default auto-scheduler"""

    def definition(self):
        # self.t0 = self.from_pytorch(inputs[0])
        # self.t1 = self.from_pytorch(inputs[1])
        self.t0 = self.define_tensor(
            sizes=[dim0, dim1], strides=[dim1, 1], dtype=DataType.BFloat16
        )
        self.t1 = self.define_tensor(
            sizes=[dim0, dim1], strides=[dim1, 1], dtype=DataType.BFloat16
        )

        # Data type conversion and computation
        self.t0_float = self.ops.cast(self.t0, dtype=DataType.Float)
        self.t1_float = self.ops.cast(self.t1, dtype=DataType.Float)
        self.t2 = self.ops.mul(self.t0_float, self.t1_float)
        if has_tanh:
            self.t2 = self.ops.tanh(self.t2)
        self.t3 = self.ops.cast(self.t2, dtype=dtype)

        # Output
        self.add_output(self.t3)

    # No schedule() method - nvfuser will use auto-scheduler


class PointwiseMulTMA(FusionDefinition):
    """Pointwise multiplication with manual 1D TMA schedule"""

    def __init__(
        self,
        tidx=128,
        unroll_factor=2,
        use_tma_store=False,
        explicit_unroll=True,
    ):
        super().__init__()
        self.tidx = tidx
        self.unroll_factor = unroll_factor
        self.vect_factor = 8  # 128 bits / 16 bits (bfloat16) = 8
        self.tma_tile = self.vect_factor * tidx * unroll_factor
        self.use_tma_store = use_tma_store
        self.explicit_unroll = explicit_unroll

    def definition(self):
        # self.t0 = self.from_pytorch(inputs[0])
        # self.t1 = self.from_pytorch(inputs[1])
        self.t0 = self.define_tensor(
            sizes=[dim0, dim1], strides=[dim1, 1], dtype=DataType.BFloat16
        )
        self.t1 = self.define_tensor(
            sizes=[dim0, dim1], strides=[dim1, 1], dtype=DataType.BFloat16
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
        # Parameters for 1D tiling
        vect_factor = self.vect_factor
        tidx = self.tidx
        unroll_factor = self.unroll_factor
        tma_tile = self.tma_tile

        # Create TMA loads from inputs to shared memory
        self.t0_smem = self.sched.cache_after(self.t0, LoadStoreOpType.tma_1d)
        self.sched.set_memory_type(self.t0_smem, MemoryType.shared)

        self.t1_smem = self.sched.cache_after(self.t1, LoadStoreOpType.tma_1d)
        self.sched.set_memory_type(self.t1_smem, MemoryType.shared)

        # Cache loads from shared memory to registers (vectorized)
        self.t0_reg = self.sched.cache_after(self.t0_smem)
        self.t1_reg = self.sched.cache_after(self.t1_smem)

        # Output caching: regs -> [smem ->] global memory
        if self.use_tma_store:
            # TMA store path: regs -> smem -> global memory (via TMA)
            self.t3_smem = self.sched.cache_before(self.t3, LoadStoreOpType.tma_1d)
            self.sched.set_memory_type(self.t3_smem, MemoryType.shared)
            self.t3_regs = self.sched.cache_before(self.t3_smem)
        else:
            # Regular store path: regs -> global memory (no TMA)
            self.t3_regs = self.sched.cache_before(self.t3)
            self.t3_smem = None

        # Pick reference tensor for scheduling (following C++ test pattern)
        reference = self.t2_mul

        # Schedule the TMA tile (1D approach)
        # [I0, I1] -> [I0*I1] -> [I0*I1/tma, tma]
        self.sched.merge(reference, dim=0)
        self.sched.split(reference, dim=0, factor=tma_tile)

        # Propagate TMA tiles to all tensors
        self.sched.transform_like(reference)

        # Schedule block tile and thread tile
        # [I, tma] -> [I, tma/v, v]
        self.sched.split(reference, dim=1, factor=vect_factor)
        # [I, tma/v, v] -> [I, tma/v/x, x, v]
        self.sched.split(reference, dim=1, factor=tidx)
        # [I, tma/v/x, x, v] -> [I, unswitch, tma/v/x, x, v]
        self.sched.split(reference, dim=0, factor=1)
        # [I, unswitch, tma/v/x, x, v] -> [I, unswitch, x, tma/v/x, v]
        self.sched.reorder(reference, {2: 3})

        # Parallelize TMA tensors
        tma_tvs = [self.t0_smem, self.t1_smem]
        if self.use_tma_store:
            tma_tvs.append(self.t3)

        for tv in tma_tvs:
            self.sched.parallelize(
                tv, axis=0, parallel_type=ParallelType.grid_x
            )  # BIDx
            self.sched.parallelize(tv, axis=1, parallel_type=ParallelType.tma)  # Bulk

        # Compute tensors (non-TMA)
        compute_tvs = [
            self.t0_reg,
            self.t1_reg,
            self.t0_float,
            self.t1_float,
            self.t2_mul,
            self.t3_regs,
        ]
        if has_tanh:
            compute_tvs.insert(-1, self.t2)

        # Add t3_smem if using TMA store, otherwise add t3 (output tensor)
        if self.use_tma_store:
            compute_tvs.append(self.t3_smem)
        else:
            compute_tvs.append(self.t3)

        # Propagate transformation to non-TMA tensors
        self.sched.transform_like(reference, selected_tensors=compute_tvs)

        # Parallelize non-TMA tensors
        # [I, unswitch, x, tma/v/x, v]
        for tv in compute_tvs:
            self.sched.parallelize(
                tv, axis=0, parallel_type=ParallelType.grid_x
            )  # BIDx
            self.sched.parallelize(
                tv, axis=1, parallel_type=ParallelType.unswitch
            )  # Unswitch
            self.sched.parallelize(
                tv, axis=2, parallel_type=ParallelType.block_x
            )  # TIDx
            if self.explicit_unroll:
                self.sched.parallelize(tv, axis=3, parallel_type=ParallelType.unroll)

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

    exit()

# if False:
if True:
    # debug run
    print("=" * 110)
    print("DEBUG RUN - Testing 1D TMA configurations")
    print("=" * 110)

    for use_tma_store in [False]:
        for explicit_unroll in [False]:
            clear_l2_cache()
            store_mode = (
                "WITH TMA Store"
                if use_tma_store
                else "WITHOUT TMA Store (Regular Store)"
            )
            unroll_mode = "WITH Unroll" if explicit_unroll else "WITHOUT Unroll"
            print(f"\n{store_mode}, {unroll_mode}")
            print("-" * 110)

            fn_tma = PointwiseMulTMA(
                tidx=128,
                unroll_factor=8,
                use_tma_store=use_tma_store,
                explicit_unroll=explicit_unroll,
            )
            nvf_out_tma = fn_tma.execute(inputs, profile=True)
            assert torch.allclose(nvf_out_tma[0], torch_out, rtol=1e-2, atol=1e-2)
            kps_tma = fn_tma.profile().kernel_profiles
            for kp in kps_tma:
                print_kernel_profile(kp)

    exit()
# ============================================================================================================
# Auto-tune: Test Different Tile Configurations (1D)
# ============================================================================================================
print("\n" + "=" * 110)
print("AUTO-TUNING 1D TMA SCHEDULER")
print("=" * 110)

# Define search space for 1D TMA
# vect_factor = 8 for bfloat16 (128 bits / 16 bits)
# tma_tile = vect_factor * tidx * unroll_factor
tidx_options = [128, 256, 512]
unroll_factor_options = [1, 2, 4, 8, 16, 32]
use_tma_store_options = [False, True]
explicit_unroll_options = [True, False]

results = []
best_time = float("inf")
best_config = None

total_configs = (
    len(tidx_options)
    * len(unroll_factor_options)
    * len(use_tma_store_options)
    * len(explicit_unroll_options)
)
print(f"\nTesting {total_configs} configurations...")
print(f"{'Config':<70} {'Time (ms)':<12} {'Bandwidth (GB/s)':<20} {'Status'}")
print("-" * 110)

config_count = 0
vect_factor = 8  # Fixed for bfloat16
for tidx in tidx_options:
    for unroll_factor in unroll_factor_options:
        tma_tile = vect_factor * tidx * unroll_factor
        for use_tma_store in use_tma_store_options:
            for explicit_unroll_opt in explicit_unroll_options:
                # Check SMEM requirements
                required_smem = tma_tile * 2 * 2  # 2 inputs, 2 bytes per bfloat16
                if use_tma_store:
                    required_smem += tma_tile * 2  # 1 output, 2 bytes per bfloat16
                if required_smem > device_smem_bytes:
                    print(
                        f"Not enough SMEM for tidx={tidx}, unroll={unroll_factor}, tma_tile={tma_tile}"
                    )
                    continue

                config_count += 1
                store_mode = "TMAStore" if use_tma_store else "RegStore"
                unroll_mode = "Unroll" if explicit_unroll_opt else "NoUnroll"
                config_str = f"TIDx={tidx}, Unroll={unroll_factor}, TMA={tma_tile}, {store_mode}, {unroll_mode}"

                try:
                    # Create fusion with specific tile configuration
                    clear_l2_cache()
                    fn_tma = PointwiseMulTMA(
                        tidx=tidx,
                        unroll_factor=unroll_factor,
                        use_tma_store=use_tma_store,
                        explicit_unroll=explicit_unroll_opt,
                    )
                    nvf_out_tma = fn_tma.execute(inputs, profile=True)

                    # Verify correctness
                    if not torch.allclose(
                        nvf_out_tma[0], torch_out, rtol=1e-2, atol=1e-2
                    ):
                        print(f"{config_str:<70} {'N/A':<12} {'N/A':<20} INCORRECT")
                        exit()

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
                            "tidx": tidx,
                            "unroll_factor": unroll_factor,
                            "tma_tile": tma_tile,
                            "use_tma_store": use_tma_store,
                            "explicit_unroll": explicit_unroll_opt,
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
                    print(f"{config_str:<70} {'FAILED':<12} {'N/A':<20} {str(e)[:30]}")
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

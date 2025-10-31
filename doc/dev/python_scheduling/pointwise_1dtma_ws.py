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
    """Pointwise multiplication with manual 1D TMA schedule following C++ test"""

    def __init__(
        self,
        tidx=128,
        unroll_factor=2,
        number_of_stages=4,
        use_tma_store=False,
    ):
        super().__init__()
        self.tidx = tidx
        self.unroll_factor = unroll_factor
        self.vect_factor = 8  # 128 bits / 16 bits (bfloat16) = 8
        self.tma_tile = self.vect_factor * tidx * unroll_factor
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
        # [I0*I1/tma, tma] -> [I0*I1/tma/sm, sm, tma]
        self.sched.split(reference, dim=0, factor=num_sms)
        # [I0*I1/tma/sm, sm, tma] -> [sm, I0*I1/tma/sm, tma]
        self.sched.reorder(reference, {0: 1})

        # Propagate TMA tiles to all tensors
        self.sched.transform_like(reference)

        # Schedule block tile and thread tile
        # [sm, I0*I1/tma/sm, tma] -> [sm, I0*I1/tma/sm, tma/v, v]
        self.sched.split(reference, dim=2, factor=vect_factor)
        # [sm, I0*I1/tma/sm, tma/v, v] -> [sm, I0*I1/tma/sm, tma/v/x, x, v]
        self.sched.split(reference, dim=2, factor=tidx)
        # [sm, I0*I1/tma/sm, tma/v/x, x, v] -> [sm, I0*I1/tma/sm, x, tma/v/x, v]
        self.sched.reorder(reference, {2: 3})

        # Parallelize TMA tensors
        tma_tvs = [self.t0_smem, self.t1_smem]
        if self.use_tma_store:
            tma_tvs.append(self.t3)

        for tv in tma_tvs:
            # [sm, I0*I1/tma/sm, tma]
            self.sched.parallelize(
                tv, axis=0, parallel_type=ParallelType.grid_x
            )  # BIDx
            self.sched.parallelize(tv, axis=2, parallel_type=ParallelType.tma)  # Bulk

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
        # [sm, I0*I1/tma/sm, x, tma/v/x, v]
        for tv in compute_tvs:
            self.sched.parallelize(
                tv, axis=0, parallel_type=ParallelType.grid_x
            )  # BIDx
            self.sched.parallelize(
                tv, axis=2, parallel_type=ParallelType.block_x
            )  # TIDx
            if explicit_unroll:
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

        # Circular Buffer with TMA loads (warp specialization)
        # Register sharing strategy based on tidx
        if tidx == 128:
            circular_buffer_type = WarpSpecialized(ParallelType.block_x)
        elif tidx == 256:
            circular_buffer_type = WarpSpecialized(ParallelType.block_x, (40, 232))
        elif tidx == 384:
            circular_buffer_type = WarpSpecialized(ParallelType.block_x, (32, 160))
        elif tidx == 512:
            circular_buffer_type = WarpSpecialized(ParallelType.block_x, (32, 112))
        else:
            circular_buffer_type = WarpSpecialized(ParallelType.block_x)

        number_of_stages = self.number_of_stages
        prefetch_distance = number_of_stages - 1

        # TMA loads - add to circular buffer
        tma_load_tvs = [self.t0_smem, self.t1_smem]
        for tv in tma_load_tvs:
            self.sched.circular_buffer(
                tv, number_of_stages, prefetch_distance, circular_buffer_type
            )


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

    # exit()


if True:
    # if True:
    # debug run
    print("=" * 110)
    print("DEBUG RUN - Testing 1D TMA with and without TMA Store")
    print("=" * 110)

    for use_tma_store in [False]:
        # clear_l2_cache()
        store_mode = (
            "WITH TMA Store" if use_tma_store else "WITHOUT TMA Store (Regular Store)"
        )
        print(f"\n{store_mode}")
        print("-" * 110)

        fn_tma = PointwiseMulTMA(
            tidx=512,
            unroll_factor=4,
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
# Auto-tune: Test Different Tile Configurations (1D)
# ============================================================================================================
print("\n" + "=" * 110)
print("AUTO-TUNING 1D TMA SCHEDULER")
print("=" * 110)

# On Blackwell, needs 64KB of data in flight to saturate memory bandwidth.
# For this case, we have 2 inputs of bfloat16, they are 2 x 2 = 4 Bytes.
# vect_factor = 8 for bfloat16 (128 bits / 16 bits)
# tma_tile = vect_factor * tidx * unroll_factor
tidx_options = [128, 256, 512]
unroll_factor_options = [1, 2, 4, 8, 16, 32]
number_of_stages_options = [3]  # use max allowed by SMEM
use_tma_store_options = [False, True]  # Test with and without TMA store


def not_enough_smem(tma_tile, n_stages, use_tma_store):
    n_input = 2
    bytes_per_element = 2
    smem_for_loads = tma_tile * n_stages * n_input * bytes_per_element
    smem_for_store = tma_tile * bytes_per_element if use_tma_store else 0
    return (smem_for_loads + smem_for_store) > device_smem_bytes


results = []
best_time = float("inf")
best_config = None

total_configs = (
    len(tidx_options)
    * len(unroll_factor_options)
    * len(number_of_stages_options)
    * len(use_tma_store_options)
)
print(f"\nTesting {total_configs} configurations...")
print(f"{'Config':<60} {'Time (ms)':<12} {'Bandwidth (GB/s)':<20} {'Status'}")
print("-" * 110)

config_count = 0
vect_factor = 8  # Fixed for bfloat16
for tidx in tidx_options:
    for unroll_factor in unroll_factor_options:
        tma_tile = vect_factor * tidx * unroll_factor
        for use_tma_store in use_tma_store_options:
            # Calculate actual stages based on available SMEM
            if use_tma_store:
                smem_for_load = device_smem_bytes - tma_tile * 2
            else:
                smem_for_load = device_smem_bytes
            smem_for_load = smem_for_load - 4096  # overhead for mbarriers, etc.
            n_stages = smem_for_load // (tma_tile * 2 * 2)
            if n_stages <= 1:
                print(
                    f"Not enough SMEM for tidx={tidx}, unroll={unroll_factor}, stages={n_stages}"
                )
                continue
            if not_enough_smem(tma_tile, n_stages, use_tma_store):
                print(
                    f"Not enough SMEM for tidx={tidx}, unroll={unroll_factor}, stages={n_stages}"
                )
                continue

            config_count += 1
            store_mode = "TMAStore" if use_tma_store else "RegStore"
            config_str = f"TIDx={tidx}, Unroll={unroll_factor}, TMA={tma_tile}, Stages={n_stages}, {store_mode}"

            try:
                clear_l2_cache()
                # Create fusion with specific tile configuration
                fn_tma = PointwiseMulTMA(
                    tidx=tidx,
                    unroll_factor=unroll_factor,
                    number_of_stages=n_stages,
                    use_tma_store=use_tma_store,
                )
                nvf_out_tma = fn_tma.execute(inputs, profile=True)

                # Verify correctness
                if not torch.allclose(nvf_out_tma[0], torch_out, rtol=1e-2, atol=1e-2):
                    print(f"{config_str:<60} {'N/A':<12} {'N/A':<20} INCORRECT")
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
                        "tidx": tidx,
                        "unroll_factor": unroll_factor,
                        "tma_tile": tma_tile,
                        "number_of_stages": n_stages,
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

                print(f"{config_str:<60} {tma_time:<12.4f} {tma_bw:<20.2f} {status}")

            except Exception as e:
                print(f"{config_str:<60} {'FAILED':<12} {'N/A':<20} {str(e)[:30]}")
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

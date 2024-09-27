# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

# Description: Schedule layer-norm fusion with nvfuser scheduling primitives

import torch
from nvfuser import (
    FusionDefinition,
    DataType,
    ParallelType,
    MemoryType,
)


batch_size = 1024
tensor_size = 4096
inputs = [
    torch.randn(batch_size, tensor_size, dtype=torch.bfloat16, device="cuda"),
]


class LayerNorm(FusionDefinition):
    def definition(self):
        self.t0 = self.from_pytorch(inputs[0])
        self.s0 = self.define_scalar(1e-6, dtype=DataType.Double)
        self.norm_const = self.define_scalar(tensor_size, dtype=DataType.Int)

        self.mean_cast = self.ops.cast(self.t0, dtype=DataType.Float)
        self.sum0 = self.ops.sum(self.mean_cast, dims=[-1])
        # NOTE Manually broadcast because fusion definition cannot access
        # hidden reduction tensor view.
        self.bcast_sum0 = self.ops.broadcast(self.sum0, [False, True])
        self.mean = self.ops.div(self.bcast_sum0, self.norm_const)

        self.var_cast = self.ops.cast(self.t0, dtype=DataType.Float)
        self.diff = self.ops.sub(self.var_cast, self.mean)
        self.diff_sq = self.ops.mul(self.diff, self.diff)
        self.sum1 = self.ops.sum(self.diff_sq, dims=[-1])
        # NOTE Manually broadcast because fusion definition cannot access
        # hidden reduction tensor view.
        self.bcast_sum1 = self.ops.broadcast(self.sum1, [False, True])
        self.var = self.ops.div(self.bcast_sum1, self.norm_const)

        self.t0_cast = self.ops.cast(self.t0, dtype=DataType.Float)
        self.t0_diff = self.ops.sub(self.t0_cast, self.mean)
        self.var_eps = self.ops.sqrt(self.ops.add(self.var, self.s0))
        self.t0_norm = self.ops.div(self.t0_diff, self.var_eps)

        self.t0_norm_cast = self.ops.cast(self.t0_norm, dtype=DataType.BFloat16)
        self.add_output(self.t0_norm_cast)

    def schedule(self):
        # create cache tensors
        cache_after_t0 = fn.sched.cache_after(fn.t0)
        fn.sched.set_memory_type(cache_after_t0, MemoryType.shared)

        cache_before_t0_norm = fn.sched.cache_before(fn.t0_norm)
        cache_tvs = [cache_after_t0, cache_before_t0_norm]
        print("cache input:\t", fn.sched.to_string(cache_after_t0))
        print("cache output:\t", fn.sched.to_string(cache_before_t0_norm))

        # Schedule Reference Tensor
        reference_tv = fn.mean
        fn.sched.split(reference_tv, dim=-1, factor=256 * 4)
        fn.sched.split(reference_tv, dim=-1, factor=4)
        fn.sched.transform_like(reference_tv)
        print("scheduled reference tensor:\n", fn.sched.to_string(reference_tv))

        # Add rfactor TensorViews
        reduction_tvs = list(filter(fn.sched.is_reduction, fn.sched.tensors()))
        assert len(reduction_tvs) == 2
        rfactor_tvs = [fn.sched.rfactor(tv, dims=[-1]) for tv in reduction_tvs]

        # Add common parallelization
        fn.sched.parallelize(reference_tv, axis := 0, ParallelType.grid_x)
        fn.sched.parallelize(reference_tv, axis := -2, ParallelType.block_x)
        fn.sched.parallelize_like(reference_tv)
        print("parallelized reference tensor:\n", fn.sched.to_string(reference_tv))

        # Vectorize input load and output store
        fn.sched.parallelize(cache_after_t0, axis := -1, ParallelType.vectorize)
        fn.sched.parallelize(fn.t0_norm, axis := -1, ParallelType.vectorize)
        print("vectorized input load:\n", fn.sched.to_string(cache_after_t0))
        print("vectorized output store:\n", fn.sched.to_string(fn.t0_norm))

        # Add computeAt; inline_most automatically skips vectorized iterDomains
        fn.sched.inline_most()


print(
    "\n\n============================================ Schedule Layer Norm ============================================"
)
fn = LayerNorm()
nvf_out = fn.execute(inputs, profile=True)
torch_out = torch.nn.functional.layer_norm(
    inputs[0], normalized_shape=inputs[0].shape[1:]
)
print(
    "==============================================================================================================\n\n"
)

# Change rtol and atol for fp16 dtype
print(
    "Nvfuser and PyTorch results match:\t",
    torch.allclose(nvf_out[0], torch_out, rtol=1e-2, atol=1e-2),
)


def print_kernel_profile(kp):
    basic_information = f"name: {kp.name}, schedule: {kp.scheduler}, segment_id: {kp.segment_id}, device: {kp.device}, stream: {kp.stream}"
    print(basic_information)

    kernel_information = f"compile time: {kp.compile_time_ms:.2f} ms, grid: {kp.grid_str}, block: {kp.block_str}, registers: {kp.registers}"
    print(kernel_information)

    runtime_information = f"input size: {kp.input_bytes} bytes, output size: {kp.output_bytes} bytes, time: {kp.time_ms:2f} ms"
    print(runtime_information)

    bandwidth_information = f"Effective Bandwidth: {kp.effective_bandwidth_gbs:.2f} GB/s, Peak Bandwidth: {kp.percentage_peak_bandwidth:2f}%"
    print(bandwidth_information)


print(
    "\n\n============================================= Profile Kernel ================================================"
)
kps = fn.profile().kernel_profiles
for kp in kps:
    print_kernel_profile(kp)
print(
    "=============================================================================================================="
)

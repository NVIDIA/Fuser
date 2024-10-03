# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

# Description: Schedule pointwise fusion without single reference tensor using scheduling primitives.

import torch
from nvfuser import (
    FusionDefinition,
    ParallelType,
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


inputs = [
    torch.randn(1024, device="cuda"),
    torch.randn(15000, 1024, device="cuda"),
    torch.randn(1024, 20000, device="cuda"),
]

# 1D - grid, block, vectorize
# 1D tiling is incompatible with both broadcast structures
# Horizontal fusion is SPMD
# a[b, 1024] + b[15000, 1024]
# a[1024, b] + c[1024, 20000]


# Apply schedule with decorator pattern.
def schedule_fn(fd):
    def schedule(fd):
        cache_after_t1 = fd.sched.cache_after(fd.t1)
        cache_after_t2 = fd.sched.cache_after(fd.t2)
        cache_before_t5 = fd.sched.cache_before(fd.t5)
        cache_before_t6 = fd.sched.cache_before(fd.t6)

        tensors_2d = list(filter(lambda t: t.ndim == 2, fd.sched.tensors()))

        # merge iterdomains and apply [grid, block, vectorize]
        for t in tensors_2d:
            # (I0 * I1) / 4, 4
            fd.sched.merge(t, dim=0)
            # (I0 * I1) / 4, 4
            fd.sched.split(t, dim=0, factor=4)
            # (I0 * I1) / 4 / 128, 128, 4
            fd.sched.split(t, dim=0, factor=128)
            # (I0 * I1) / 4 / 128, 128, 4
            fd.sched.parallelize(t, axis := 0, ParallelType.grid_x)
            fd.sched.parallelize(t, axis := -2, ParallelType.block_x)

        # vectorize 2d tensors
        fd.sched.parallelize(cache_after_t1, axis := -1, ParallelType.vectorize)
        fd.sched.parallelize(cache_after_t2, axis := -1, ParallelType.vectorize)
        fd.sched.parallelize(fd.t5, axis := -1, ParallelType.vectorize)
        fd.sched.parallelize(fd.t6, axis := -1, ParallelType.vectorize)

        # computeAt - automatically handles vectorize paralleltype
        fd.sched.inline_most()

    fd.schedule = schedule
    return fd


@schedule_fn
class Pointwise(FusionDefinition):
    def definition(self):
        self.t0 = self.from_pytorch(inputs[0])
        self.t1 = self.from_pytorch(inputs[1])
        self.t2 = self.from_pytorch(inputs[2])

        self.t3 = self.ops.broadcast(self.t0, [True, False])
        self.t4 = self.ops.broadcast(self.t0, [False, True])

        self.t5 = self.ops.add(self.t3, self.t1)
        self.t6 = self.ops.add(self.t4, self.t2)
        self.add_output(self.t5)
        self.add_output(self.t6)


print(
    "\n\n============================================= Profile 1 segment Kernel ================================================"
)
fn = Pointwise()
nvf_out = fn.execute(inputs, profile=True)

kps = fn.profile().kernel_profiles
for kp in kps:
    print_kernel_profile(kp)
print(
    "=============================================================================================================="
)

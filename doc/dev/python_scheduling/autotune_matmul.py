# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
from nvfuser import FusionDefinition, SchedulerType

# Description of the problem
M = 512
N = 512
K = 4096
dtype = torch.bfloat16
# TODO: layout

# These are the parameters we'll optimize
parameter_configurations = [
    splitk_factors := list(range(1, 8)),
    load_stages := list(range(1, 4)),
]


# A decorator to create a pointwise fusion given some input arguments.
def create_fusion_func(inputs):
    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.matmul(t0, t1)
        fd.add_output(t2)

    return fusion_func


# The pytorch eager mode reference used to validating nvfuser kernel.
def eager_reference(inputs):
    return torch.matmul(inputs[0], inputs[1])


# Apply scheduler with custom parameters using decorator
def custom_matmul_scheduler(fd, config):
    def inner_fn():
        # Check if compatible with matmul scheduler
        status, error = fd.sched.can_schedule(SchedulerType.matmul)
        assert status, error

        schedule_params = fd.sched.compute_matmul_heuristics()

        # Modify original parameters
        if config is not None:
            splitk_factor, stages = config
            schedule_params.circular_buffer_options.circular_buffer_smem_write = (
                stages > 1
            )
            schedule_params.circular_buffer_options.smem_circular_buffer_stage = stages
            schedule_params.splitk_factor = splitk_factor

        # Schedule fusion
        fd.sched.schedule()

    fd.schedule = inner_fn
    return fd


# Apply schedule decorator, run fusion, and profile performance
def run_profile(presched_fd, inputs, config=None, num_iterations=10):
    scheduled_fd = custom_matmul_scheduler(presched_fd, config)

    mean_bw = 0
    mean_time = 0.0
    for iteration in range(num_iterations):
        nvf_outputs = scheduled_fd.execute(inputs, profile=True)

        prof = scheduled_fd.profile()
        bandwidth = prof.kernel_profiles[0].effective_bandwidth_gbs
        time = prof.kernel_profiles[0].time_ms
        mean_bw += (bandwidth - mean_bw) / (iteration + 1)
        mean_time += (time - mean_time) / (iteration + 1)

    # validate correctness
    # assert torch.allclose(nvf_outputs[0], eager_reference(inputs))

    return mean_bw, mean_time


# exhaustively search for best time
inputs = [
    torch.randn((M, K), dtype=dtype, device="cuda"),
    torch.randn((K, N), dtype=dtype, device="cuda"),
]

with FusionDefinition() as presched_fd:
    create_fusion_func(inputs)(presched_fd)

optimal_config = None
optimal_perf = None
for config in itertools.product(splitk_factors, load_stages):
    splitk_factor, stages = config
    # TODO: introduce a utility to check if this config is valid.
    # For example, it should check on smem reuse
    bw, kernel_time = run_profile(presched_fd, inputs, config)
    perf_metric = kernel_time

    print(f"  sk={splitk_factor} st={stages} bw={bw} kernel_time={kernel_time}")

    if optimal_config is None or perf_metric < optimal_perf:
        optimal_config = config
        optimal_perf = perf_metric

print(
    f"M={M} N={N} K={K} optimal splitk={optimal_config[0]} stages={optimal_config[1]}"
)

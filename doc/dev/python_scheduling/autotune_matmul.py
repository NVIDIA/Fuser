# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
from nvfuser import FusionDefinition, SchedulerType

# ============================ Description ============================

# 1. Define a nvfuser fusion and its pytorch eager mode reference.
#
# 2. Profile the CUDA kernel performance by iterating over a set of input
# arguments and scheduler configurations.
#
# 3. Train a regression model to predict the desired performance metric given
# some input arguments and a scheduler configuration.
#
# 4. Measure the performance of the regression model.
#  - Calculate RMSE of predicted and actual performance on test set.
#  - Find the configuration with the best performance using regression model.
#    Then, compare against the heuristic configuration selected by nvfuser.
#  - For a specific batch size, gather performance across a range of hidden
#    sizes. Calculate performance for best predicted and nvfuser
#    configurations. Plot a chart comparing performance using matplotlib.

# The selected performance metric is effective_bandwidth_gbs. The empirical
# scheduler selects the configuration that has the highest predicted
# effective_bandwidth_gbs.

# ============================ Configurations ============================

# Settings for input tensor generation
num_dimensions = 3
Ms = [512]
Ns = [512]
Ks = [4096]

# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors.
parameter_configurations = [
    splitk_factors := list(range(1, 8)),
    load_stages := list(range(1, 4)),
]

# We profile a range of input shapes with various configurations.
# This argument determines how much of the profiled data to keep as a test set.
test_data_percentage = 0.1

# The selected batch size for empirical and nvfuser comparison.
empirical_batch_size = 512

# The range of hidden sizes for empirical and nvfuser comparision.
empirical_hidden_sizes = list(range(256, 28672, 256))


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


# ============================ Function Definitions ============================


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
            schedule_params.circular_buffer_smem_write = stages > 1
            schedule_params.splitk_factor = splitk_factor
            schedule_params.smem_circular_buffer_stage = stages

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


def argmax(map_config_to_perf):
    best_perf = -1
    best_config = None
    for config, perf in map_config_to_perf.items():
        if perf > best_perf:
            best_perf = perf
            best_config = config
    return best_config


# Given a prediction model, input_shape, and set of parameter configurations,
# find the best parameters
def find_best_parameters(predictor, input_shape, parameter_configurations):
    map_config_to_performance = {
        config: predictor.predict([[*input_shape, *config]])
        for config in itertools.product(*parameter_configurations)
    }
    return argmax(map_config_to_performance)


# ============================ Run Experiments  ================================

# Collect data for decision tree
parameters = []
performance = []

dtype = torch.bfloat16

for M, N, K in itertools.product(Ms, Ns, Ks):
    # print(M, N, K)
    inputs = [
        torch.randn((M, K), dtype=dtype, device="cuda"),
        torch.randn((K, N), dtype=dtype, device="cuda"),
    ]

    with FusionDefinition() as presched_fd:
        create_fusion_func(inputs)(presched_fd)

    optimal_config = None
    optimal_perf = None
    # exhaustively search for best time
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

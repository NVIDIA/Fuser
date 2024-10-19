# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
import random
import numpy as np
import itertools
from nvfuser import FusionCache, FusionDefinition, SchedulerType

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

# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors.
parameter_configurations = [
    vectorize_range := [1, 2, 4],
    unroll_range := list(range(1, 10)),
]

# We profile a range of input shapes with various configurations.
# This argument determines how much of the profiled data to keep as a test set.
test_data_percentage = 0.1

# The selected batch size for empirical and nvfuser comparison.
empirical_batch_size = 512

# maximum number of operation in fusion
max_number_operations = 3

# ============================ Function Definitions ============================


# Apply scheduler with custom parameters using decorator
def custom_pointwise_scheduler(fd, config):
    def inner_fn():
        # Check if compatible with pointwise scheduler
        status, _ = fd.sched.can_schedule(SchedulerType.pointwise)
        assert status

        schedule_params = fd.sched.compute_pointwise_heuristics()

        # Modify original parameters
        if config is not None:
            vectorization_factor, unroll_factor = config
            schedule_params.vectorization_factor = vectorization_factor
            schedule_params.unroll_factor = unroll_factor

        # Schedule fusion
        fd.sched.schedule()

    fd.schedule = inner_fn
    return fd


# Apply schedule decorator, run fusion, and profile performance
def run_profile(presched_fd, inputs, config=None):
    scheduled_fd = custom_pointwise_scheduler(presched_fd, config)
    nvf_outputs = scheduled_fd.execute(inputs, profile=True)

    # validate correctness
    # TODO use thunder to compile torch eager to nvfuser fusion
    # assert torch.allclose(nvf_outputs[0], eager_reference(inputs))

    prof = scheduled_fd.profile()
    bandwidth = prof.kernel_profiles[0].effective_bandwidth_gbs
    time = prof.kernel_profiles[0].time_ms
    return bandwidth, time


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


def generate_shapes_with_bcast(tensor_shape):
    num_dims = len(tensor_shape)
    for num_bcast_dims in range(num_dims):
        # Get all combinations of broadcast dimensions
        for bcast_indices in itertools.combinations(range(num_dims), num_bcast_dims):
            yield [
                dim_size if idx not in bcast_indices else 1
                for idx, dim_size in enumerate(tensor_shape)
            ]


def create_fusion_state(full_tensor_shape, maximum_number_operations):
    # vertical composition: number of operations, order of mufu operations
    # horizontal composition: number and size of tensor dimensions, order of
    # broadcast dimensions.

    for num_ops in range(1, maximum_number_operations + 1):
        for num_mufu in range(num_ops + 1):
            # Get a random subset of num_mufu indices from num_ops
            mufu_indices = np.random.choice(num_ops, num_mufu, replace=False)

            num_tensors = 1 + num_ops - num_mufu
            tensor_shapes_with_bcast = list(
                generate_shapes_with_bcast(full_tensor_shape)
            )
            shapes_for_all_tensors = [tensor_shapes_with_bcast] * num_tensors
            for input_shapes in itertools.product(*shapes_for_all_tensors):
                yield (num_ops, mufu_indices, input_shapes)


def create_fusion_definition(num_operations, mufu_indices, input_shapes):
    input_tensors = [torch.randn(shape, device="cuda") for shape in input_shapes]

    with FusionDefinition() as fd:
        output = None
        output_tensor = None
        input_tensor_idx = 0
        for idx in range(num_operations):
            if output is None:
                output = fd.from_pytorch(input_tensors[input_tensor_idx])
                input_tensor_idx += 1

            if idx in mufu_indices:
                output = fd.ops.exp(output)
            else:
                other = fd.from_pytorch(input_tensors[input_tensor_idx])
                input_tensor_idx += 1
                output = fd.ops.add(output, other)
        fd.add_output(output)

    return fd, input_tensors


# ============================ Run Experiments  ================================

# Collect data for decision tree
parameters = []
performance = []

full_tensor_shape = [10, 256]
for fusion_config in create_fusion_state(full_tensor_shape, max_number_operations):
    num_ops, mufu_indices, input_shapes = fusion_config
    presched_fd, input_tensors = create_fusion_definition(*fusion_config)

    print(fusion_config)
    # unroll and vectorization configurations
    for config in itertools.product(vectorize_range, unroll_range):
        perf_metric, _ = run_profile(presched_fd, input_tensors, config)
        parameters.append((*input_shapes, num_ops, len(mufu_indices), *config))
        performance.append(perf_metric)

# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
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

# Run through a shmoo of pointwise fusion configurations.
# Apply different vectorization and unroll factor settings.
# Profile several metrics for each fusion and scheduler combination.
# Create Pandas DataFrame
# Save results to csv file.

# Fusion Configurations:
# Binary Math Operation is add.
# MUFU Operation is exp.
#
# Vertical composition: Number of math operations, order of mufu operations
#
# Horizontal composition: Number and size of tensor dimensions, order of
# broadcast dimensions.

# Performance Metrics:
# 1. Number of Registers
# 2. Shared memory usage - static and dynamic
# 3. Grid shape
# 4. Block shape
# 5. Effective Bandwidth (GB/s)
# 6. Kernel Runtime (ms)

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
    outputs = scheduled_fd.execute(inputs, profile=True)

    # validate correctness
    # TODO use thunder to compile torch eager to nvfuser fusion
    # assert torch.allclose(nvf_outputs[0], eager_reference(inputs))

    prof = scheduled_fd.profile()

    num_registers = prof.kernel_profiles[0].registers
    smem = prof.kernel_profiles[0].shared_mem_str
    grid_shape = prof.kernel_profiles[0].grid_str
    block_shape = prof.kernel_profiles[0].block_str
    bandwidth = prof.kernel_profiles[0].effective_bandwidth_gbs
    time = prof.kernel_profiles[0].time_ms

    return outputs, (grid_shape, block_shape, num_registers, smem, bandwidth, time)


# ============================ Create Fusion  ================================


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


# ============================ Metrics  ============================


# Broadcast multiples is a matrix of size [ndims, 2]. Each entry [i] is the
# number of inputs and output tensors that have a non-broadcast dimension
# mapped to the same dimension. Broadcast multiples is multiplied by data type
# size of each tensor.
def get_broadcast_multiple(input_tensors, output_tensors, breakpoint_dim):
    lhs = 0
    rhs = 0
    from itertools import chain

    for t in chain(input_tensors, output_tensors):
        for idx, dim_size in enumerate(t.shape):
            value = t.dtype.itemsize if dim_size > 1 else 0
            if idx < breakpoint_dim:
                lhs += value
            else:
                rhs += value
    return lhs, rhs


# ============================ Configurations ============================

# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors.
parameter_configurations = [
    vectorize_range := [1, 2, 4],
    unroll_range := list(range(1, 10)),
]

# maximum number of operation in fusion
max_number_operations = 3

# Settings for input tensor generation
num_dimensions = 2
outer_shapes = [512]
inner_shapes = [2**i for i in range(5, 15)]

# ============================ Run Experiments  ================================

# Collect data for decision tree
data = []

for full_tensor_shape in itertools.product(outer_shapes, inner_shapes):
    print(full_tensor_shape)
    for fusion_config in create_fusion_state(full_tensor_shape, max_number_operations):
        num_ops, mufu_indices, input_shapes = fusion_config
        presched_fd, input_tensors = create_fusion_definition(*fusion_config)

        print(fusion_config)
        # unroll and vectorization configurations
        for config in itertools.product(vectorize_range, unroll_range):
            try:
                output_tensors, metrics = run_profile(
                    presched_fd, input_tensors, config
                )
            except KeyboardInterrupt:
                import sys

                sys.exit()
            except:
                print(
                    f"Warning: failed to run fusion given {input_tensors} and configuration {config}"
                )
                continue

            # collect extra metrics
            broadcast_multiples = [
                get_broadcast_multiple(input_tensors, output_tensors, idx)
                for idx in range(len(full_tensor_shape) + 1)
            ]
            grid, block, registers, smem, bandwidth, time = metrics

            # create data entry
            entry = [
                input_shapes,
                [list(t.shape) for t in output_tensors],
                num_ops,
                len(mufu_indices),
                broadcast_multiples,
                *config,
                grid,
                block,
                registers,
                smem,
                bandwidth,
                time,
            ]
            data.append(entry)

# ============================ Save Pandas DataFrame ============================

import pandas as pd

df = pd.DataFrame(
    data,
    columns=[
        "input_shapes",
        "output_shapes",
        "number_of_operations",
        "number_of_mufu_operations",
        "broadcast_multiples",
        "vectorization",
        "unroll_factor",
        "grid",
        "block",
        "number_of_registers",
        "shared_memory_usage",
        "effective_bandwidth",
        "kernel_time_ms",
    ],
)

major, minor = torch.cuda.get_device_capability()
df.to_csv("pointwise_data_device_{major}_{minor}.csv", index=True)
print(f"Finished creating {len(data)} entries")

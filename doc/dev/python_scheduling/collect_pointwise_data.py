# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
import numpy as np
from nvfuser import FusionDefinition, SchedulerType

# ============================ Description ============================

# Run through a shmoo of pointwise fusion configurations.
# Apply different vectorization and unroll factor settings.
# Profile several metrics for each fusion and scheduler combination.
# Create Pandas DataFrame
# Save results to csv file.

# Fusion Configurations:
# Tensor DataType
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


# A tensor can have a number of broadcast dimensions at any axis of tensor.
# Given a number of dimension in a tensor, iterate through range of broadcast
# axes. Given a number of broadcast axes, iterate through all combinations of
# their placement in the tensor.
def generate_shapes_with_bcast(tensor_shape):
    num_dims = len(tensor_shape)
    for num_bcast_dims in range(num_dims):
        # Get all combinations of broadcast dimensions
        for bcast_indices in itertools.combinations(range(num_dims), num_bcast_dims):
            yield [
                dim_size if idx not in bcast_indices else 1
                for idx, dim_size in enumerate(tensor_shape)
            ]


def create_fusion_state(full_tensor_shape, maximum_number_operations, tensor_datatypes):
    # vertical composition:
    # * number of operations
    # * order of mufu operations
    #
    # horizontal composition:
    # * number, dtype, and size of tensor dimensions
    # * order of broadcast dimensions.
    for num_ops in range(1, maximum_number_operations + 1):
        for num_mufu in range(num_ops + 1):
            # Get a random subset of num_mufu indices from num_ops
            mufu_indices = np.random.choice(num_ops, num_mufu, replace=False)

            num_tensors = 1 + num_ops - num_mufu
            # Get all combinations of broadcast dimensions for a tensor
            tensor_shapes_with_bcast = list(
                generate_shapes_with_bcast(full_tensor_shape)
            )

            # Get all combinations of tensor shapes and tensor data types
            tensor_shapes_with_bcast_and_dtype = itertools.product(
                tensor_shapes_with_bcast, tensor_datatypes
            )

            # Get all combinations for fusion's input tensors
            shapes_for_all_tensors = itertools.tee(
                tensor_shapes_with_bcast_and_dtype, num_tensors
            )
            for input_shapes in itertools.product(*shapes_for_all_tensors):
                yield (num_ops, mufu_indices, input_shapes)


def create_fusion_definition(num_operations, mufu_indices, input_shapes):
    input_tensors = [
        torch.randn(shape, dtype=dtype, device="cuda") for shape, dtype in input_shapes
    ]

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


# Get all tensor shape, fusion, and scheduler configurations
def create_data_config():
    for full_tensor_shape in itertools.product(
        data_gen_config.outer_shapes, data_gen_config.inner_shapes
    ):
        for fusion_config in create_fusion_state(
            full_tensor_shape,
            data_gen_config.max_number_operations,
            data_gen_config.tensor_datatypes,
        ):
            for scheduler_config in itertools.product(
                data_gen_config.vectorize_range, data_gen_config.unroll_range
            ):
                yield (full_tensor_shape, fusion_config, scheduler_config)


# ============================ Utilities  ============================


# Find largest datatype in input tensors.
# Calculate largest factor given 128B vectorize memory operation.
# Return if current vectorize factor <= max_vectorize_factor
def valid_vectorize_factor(input_tensors, vectorize_factor):
    max_dtype_size = max([t.dtype.itemsize for t in input_tensors])
    max_vectorize_factor = 16 // max_dtype_size
    return vectorize_factor <= max_vectorize_factor


# Create pandas dataframe then save it as a csv file to specified location
def save(directory_path, data, interval=None):
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

    from pathlib import Path

    base_directory = Path(directory_path)
    major, minor = torch.cuda.get_device_capability()

    if interval is None:
        file_path_str = f"pointwise_data_device_{major}_{minor}.csv"
    else:
        file_path_str = f"pointwise_data_pt{interval}_device_{major}_{minor}.csv"

    file_path = base_directory.joinpath(file_path_str)
    df.to_csv(file_path, index=True)
    print(f"Finished creating {len(data)} entries. Saved to {file_path_str}.")


# ============================ Metrics  ============================


# Broadcast multiples is a matrix of size [ndims, 2]. Each entry [i] is the
# number of inputs and output tensors that have a non-broadcast dimension
# mapped to the same dimension. Broadcast multiples is multiplied by data type
# size of each tensor.
def get_broadcast_multiple(input_tensors, output_tensors, breakpoint_dim):
    lhs = 0
    rhs = 0

    for t in itertools.chain(input_tensors, output_tensors):
        for idx, dim_size in enumerate(t.shape):
            value = t.dtype.itemsize if dim_size > 1 else 0
            if idx < breakpoint_dim:
                lhs += value
            else:
                rhs += value
    return lhs, rhs


# ============================ Configurations ============================

from dataclasses import dataclass


@dataclass
class DataGenerationConfiguration:
    # tensor configuration
    num_dimensions: int
    outer_shapes: [int]
    inner_shapes: [int]
    tensor_datatypes: [torch.dtype]

    # maximum number of operation in fusion
    max_number_operations: int

    # pointwise scheduler parameters
    vectorize_range: [int]
    unroll_range: [int]


# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors. Limit vectorization factor to 4 instead of 8 because pointwise
# configurations cast float16 and bfloat16 to float32.
data_gen_config = DataGenerationConfiguration(
    num_dimensions=2,
    outer_shapes=[512],
    inner_shapes=[128, 512, 2048, 8192, 16384, 32768],
    tensor_datatypes=[torch.float32, torch.bfloat16],
    max_number_operations=3,
    vectorize_range=[1, 2, 4],
    unroll_range=list(range(1, 10)),
)


# Run profiling on series of fusions to collect data.
def run(args):
    data = []
    interval = args.start_interval

    for idx, (full_tensor_shape, fusion_config, scheduler_config) in enumerate(
        create_data_config()
    ):
        # short-circuit: skip configurations based on fast-forward argument.
        # When resuming data collection, skip configurations we have already tested.
        if idx < args.fast_forward:
            continue

        # Save data based on interval
        if len(data) >= args.save_interval:
            save(args.save_path, data, interval)
            interval += 1
            data.clear()

        num_ops, mufu_indices, input_shapes = fusion_config
        vectorize_factor, unroll_factor = scheduler_config

        # create prescheduled fusion and input arguments
        presched_fd, input_tensors = create_fusion_definition(*fusion_config)

        # short-circuit: skip if vectorization factor is incompatible with input tensors
        if not valid_vectorize_factor(input_tensors, vectorize_factor):
            continue

        # collect profiling data given fusion, input arguments, and scheduler configuration
        try:
            output_tensors, metrics = run_profile(
                presched_fd, input_tensors, scheduler_config
            )
        except KeyboardInterrupt:
            import sys

            sys.exit()
        except (AssertionError, RuntimeError):
            print(
                f"Warning: failed to run fusion given {input_tensors} and configuration {scheduler_config}"
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
            [[shape, dtype.itemsize] for shape, dtype in input_shapes],
            [list(t.shape) + [t.itemsize] for t in output_tensors],
            num_ops,
            len(mufu_indices),
            broadcast_multiples,
            vectorize_factor,
            unroll_factor,
            grid,
            block,
            registers,
            smem,
            bandwidth,
            time,
        ]
        data.append(entry)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect Data for Pointwise Scheduler."
    )
    parser.add_argument(
        "--save_path",
        default="~/workspace/autotune/pointwise",
        help="The path to save data",
    )
    parser.add_argument(
        "--save_interval",
        default=1000,
        type=int,
        help="The number of entries to collect before saving results.",
    )
    parser.add_argument(
        "--start_interval",
        default=0,
        type=int,
        help="The start interval for data parts.",
    )
    parser.add_argument(
        "--fast_forward",
        default=0,
        type=int,
        help="The number of entries to skip before starting data collection.",
    )
    args = parser.parse_args()

    run(args)

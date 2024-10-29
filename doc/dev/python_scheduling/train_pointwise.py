# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
import numpy as np
import random
import math
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

# TODO


# ============================ Utilities  ============================


# Find largest datatype in input tensors.
# Calculate largest factor given 128B vectorize memory operation.
# Return if current vectorize factor <= max_vectorize_factor
def valid_vectorize_factor(input_tensors, vectorize_factor):
    max_dtype_size = max([t.dtype.itemsize for t in input_tensors])
    max_vectorize_factor = 16 // max_dtype_size
    return vectorize_factor <= max_vectorize_factor


# Create pandas dataframe then save it as a csv file to specified location
def load(directory_path):
    import pandas as pd
    import os

    all_data_frames = []
    total = 0
    for f in os.listdir(directory_path):
        full_path = os.path.join(directory_path, f)
        # short-circuit: skip if not a file
        if not os.path.isfile(full_path):
            continue

        # short-circuit: skip non-csv files
        if not f.endswith(".csv"):
            continue

        data_frame = pd.read_csv(full_path, index_col=0, header=0)
        # The data frame index is numbered from 0 to N-1. Increment the indices
        # by the cummulative total rows in all data frames, so all indicies are
        # unique.
        data_frame.index += total
        total += data_frame.shape[0]
        all_data_frames.append(data_frame)

    return pd.concat(all_data_frames, axis=0)


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
    number_operations: [int]

    # pointwise scheduler parameters
    vectorize_range: [int]
    unroll_range: [int]


# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors. Limit vectorization factor to 4 instead of 8 because pointwise
# configurations cast float16 and bfloat16 to float32.
data_gen_config = DataGenerationConfiguration(
    num_dimensions=2,
    outer_shapes=[512, 8192, 32768],
    inner_shapes=[512, 8192, 32768],
    tensor_datatypes=[torch.float32, torch.bfloat16],
    number_operations=[2, 5],
    vectorize_range=[1, 2, 4],
    unroll_range=[1, 3, 6, 9],
)


# Run profiling on series of fusions to collect data.
def run(args):
    # Step 1: load data from directory
    data_df = load(args.save_path)
    print("Loaded training data frame", data_df.shape)
    print(data_df.columns)

    # Step 2: Train decision tree using all data.
    input_data = data_df[
        [
            "input_shapes",
            "output_shapes",
            "number_of_operations",
            "vectorization",
            "unroll_factor",
        ]
    ]
    calculate_total_bytes = lambda string: math.prod(
        [int(char) for char in string if char.isdigit()]
    )
    rows = []
    for index, r in input_data.iterrows():
        input_bytes = calculate_total_bytes(r["input_shapes"])
        output_bytes = calculate_total_bytes(r["output_shapes"])
        entry = [
            input_bytes,
            output_bytes,
            r["number_of_operations"],
            r["vectorization"],
            r["unroll_factor"],
        ]
        rows.append(entry)
    input_data = np.array(rows)
    output_data = data_df[["effective_bandwidth"]].to_numpy()

    # Apply decision tree regressor
    # Given input shapes, output shapes, and scheduler parameters, predict performance metric.
    from sklearn import tree

    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(input_data, output_data)

    # Step 3: Test decision tree by comparing against nvfuser.


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
    args = parser.parse_args()

    run(args)

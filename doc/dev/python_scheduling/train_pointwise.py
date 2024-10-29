# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import thunder
import torch
import itertools
import numpy as np
import random
import math
from nvfuser import FusionCache, FusionDefinition, SchedulerType, clone

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
def run_profile(eager_reference, presched_fd, inputs, config=None):
    scheduled_fd = custom_pointwise_scheduler(presched_fd, config)
    outputs = scheduled_fd.execute(inputs, profile=True)

    # validate correctness
    assert torch.allclose(outputs[0], eager_reference(*inputs), atol=1e-2, rtol=1e-2)

    prof = scheduled_fd.profile()
    return prof.kernel_profiles[0].time_ms


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
def find_best_parameters(
    predictor, input_shape, output_shape, parameter_configurations
):
    map_config_to_performance = {
        config: predictor.predict([[input_shape, output_shape, *config]])
        for config in itertools.product(*parameter_configurations)
    }
    return argmax(map_config_to_performance)


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


def get_numpy_training_data(data_frame):
    input_data = data_frame[
        [
            "input_shapes",
            "output_shapes",
            "vectorization",
            "unroll_factor",
        ]
    ]
    calculate_total_bytes = lambda string: math.prod(
        [float(char) for char in string if char.isdigit()]
    )
    rows = []
    for index, r in input_data.iterrows():
        input_bytes = calculate_total_bytes(r["input_shapes"]) / 1e6
        output_bytes = calculate_total_bytes(r["output_shapes"]) / 1e6
        entry = [
            input_bytes,
            output_bytes,
            r["vectorization"],
            r["unroll_factor"],
        ]
        rows.append(entry)
    input_data = np.array(rows)
    output_data = data_frame[["effective_bandwidth"]].to_numpy()
    return input_data, output_data


def matplotlib_test(clf, reference_inputs, eager_reference, fd):
    #  For a specific batch size, gather performance across a range of hidden sizes.
    #  Calculate performance for best predicted and nvfuser configurations. Plot a
    #  chart comparing performance using matplotlib.

    # NOTE: The matplotlib experiment plots the kernel runtime, which could be
    # different than the selected performance metric. Currently, the performance
    # metric is effective_bandwidth_gbs.

    import matplotlib.pyplot as plt
    import numpy as np

    # The selected batch size for empirical and nvfuser comparison.
    empirical_batch_size = 512

    # The range of hidden sizes for empirical and nvfuser comparision.
    empirical_hidden_sizes = list(range(256, 28672, 256))

    # For pointwise scheduler, we test the cartesian product of vectorization and
    # unroll factors.
    parameter_configurations = [
        vectorize_range := [1, 2, 4],
        unroll_range := list(range(1, 10)),
    ]

    est_perfs = []
    for hidden_shape in empirical_hidden_sizes:
        inputs = [
            torch.randn(
                empirical_batch_size, hidden_shape, dtype=ref.dtype, device="cuda"
            )
            for ref in reference_inputs
        ]
        input_bytes = math.prod([a.numel() * a.dtype.itemsize for a in inputs]) / 1e6
        output_bytes = inputs[0].numel() * inputs[0].dtype.itemsize
        estimate_config = find_best_parameters(
            clf, input_bytes, output_bytes, parameter_configurations
        )

        # clone reference fusion definition
        presched_fd = FusionDefinition()
        clone(fd, presched_fd)

        est_time_ms = run_profile(eager_reference, presched_fd, inputs, estimate_config)
        est_perfs.append(est_time_ms)
        print(
            f"{empirical_batch_size}, {hidden_shape}, {estimate_config}, {est_time_ms:.3f}"
        )

    nvf_perfs = []
    for hidden_shape in empirical_hidden_sizes:
        inputs = [
            torch.randn(
                empirical_batch_size, hidden_shape, dtype=ref.dtype, device="cuda"
            )
            for ref in reference_inputs
        ]

        # clone reference fusion definition
        presched_fd = FusionDefinition()
        clone(fd, presched_fd)

        nvf_time_ms = run_profile(eager_reference, presched_fd, inputs)
        nvf_perfs.append(nvf_time_ms)
        print(f"{empirical_batch_size}, {hidden_shape}, {nvf_time_ms:.3f}")

    # Get mean speed-up from nvfuser to empirical configurations across all input shapes.
    # Negative value mean empirical configurations are slower than nvfuser.
    print("Mean speed-up", np.mean(np.array(nvf_perfs) - np.array(est_perfs)))

    np_hidden_size = np.array(empirical_hidden_sizes)
    plt.plot(np_hidden_size, np.array(est_perfs))
    plt.plot(np_hidden_size, np.array(nvf_perfs))

    plt.xlabel("Hidden Size")
    plt.ylabel("Time(ms)")
    plt.title(
        f"Batch Size = {empirical_batch_size}, Compare Decision Tree Heuristic vs NvFuser"
    )
    plt.legend(["decision_tree", "nvfuser"], loc="lower right")
    plt.savefig(f"pointwise_empirical_batchsize{empirical_batch_size}.png")


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


# The pytorch eager mode reference used to validating nvfuser kernel.
def eager_reference(a, b):
    return torch.nn.functional.gelu(a + b, approximate="tanh")


# Run profiling on series of fusions to collect data.
def run(args):
    # Step 1: load data from directory
    data_df = load(args.save_path)
    print("Loaded training data frame", data_df.shape)
    print(data_df.columns)

    # Step 2: Train decision tree using all data.
    input_data, output_data = get_numpy_training_data(data_df)

    # Apply decision tree regressor
    # Given input shapes, output shapes, and scheduler parameters, predict performance metric.
    from sklearn import tree

    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(input_data, output_data)

    # Step 3: Test decision tree by comparing against nvfuser.
    # Use thunder to jit an eager reference.
    # Get nvfuser fusion definition from thunder
    a = torch.randn(512, 10016, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(512, 10016, dtype=torch.bfloat16, device="cuda")
    nvf_model = thunder.jit(eager_reference)
    result = nvf_model(a, b)
    fd = thunder.last_traces(nvf_model)[-1].python_ctx()["nvFusion0"].last_used

    # Run decision tree and nvfuser heuristics
    # Create Graph with matplotlib
    matplotlib_test(clf, reference_inputs := (a, b), eager_reference, fd)


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

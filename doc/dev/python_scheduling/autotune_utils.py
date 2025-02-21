# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import math
import itertools
from nvfuser import FusionCache, FusionDefinition
from dataclasses import dataclass, astuple

# ================================ Description ================================
# This file contains the utility function for autotuning scripts.
# =============================================================================


# Returns the result of a/b rounded to the nearest integer in the direction of
# positive infinity.
def ceil_div(a, b):
    return int(math.ceil(a / b))


# Returns the result of a/b rounded to the nearest integer in the direction of
# negative infinity.
def floor_div(a, b):
    return int(math.floor(a / b))


@dataclass
class ScriptConfiguration:
    # Settings for input tensor generation
    # number of dimensions in the tensor argument
    num_dimensions: int

    # the data type for the tensor argument
    tensor_datatype: torch.dtype

    # During training, the cartesian product of outer_shapes and inner_shapes
    # is used to define the shape of the input tensor arguments.
    outer_shapes: [int]
    inner_shapes: [int]

    # We profile a range of input shapes with various configurations.
    # This argument determines how much of the profiled data to keep as a test
    # set.
    test_data_percentage: [float]

    # The selected batch size for empirical and nvfuser comparison.
    empirical_batch_size: [int]

    # The range of hidden sizes for empirical and nvfuser comparision.
    empirical_hidden_sizes: [int]


# Converted DataClass to a Tuple. It flattens nested tuples. The function is
# used for compatibility with machine learning model.
def flatten_configuration(scheduler_config):
    new_scheduler_config = []
    for item in astuple(scheduler_config):
        if type(item) is tuple:
            new_scheduler_config.extend(item)
        else:
            new_scheduler_config.append(item)
    return tuple(new_scheduler_config)


# Collect data for machine learning
def collect_data(script_config, autotune_config):
    parameters = []
    performance = []

    for shape in itertools.product(
        script_config.outer_shapes, script_config.inner_shapes
    ):
        print(shape)
        inputs = autotune_config.create_inputs(shape, script_config.tensor_datatype)

        with FusionDefinition() as presched_fd:
            autotune_config.create_fusion_func(inputs)(presched_fd)

        # unroll and vectorization configurations
        for parameter_config in autotune_config.generate_scheduler_configurations(
            shape
        ):
            perf_metric, _ = run_profile(
                autotune_config, presched_fd, inputs, parameter_config
            )
            parameters.append((*shape, *flatten_configuration(parameter_config)))
            performance.append(perf_metric)
    return parameters, performance


# Separate collected data into training and test sets
def separate_data(script_config, parameters, performance):
    import random

    train_inputs = []
    test_inputs = []
    train_perf = []
    test_perf = []
    test_shapes = set()
    all_test_scheduler_config = {}  # key: input_shape, value: (scheduler_config, perf)

    for data, perf in zip(parameters, performance):
        shape = data[: script_config.num_dimensions]
        scheduler_config = data[script_config.num_dimensions :]

        if shape in all_test_scheduler_config:
            all_test_scheduler_config[shape][scheduler_config] = perf
        else:
            all_test_scheduler_config[shape] = {scheduler_config: perf}

        if (
            script_config.test_data_percentage > 0
            and random.random() < script_config.test_data_percentage
        ):
            test_shapes.add(shape)
            test_inputs.append(data)
            test_perf.append(perf)
        else:
            train_inputs.append(data)
            train_perf.append(perf)

    # key: input_shape, value: best_scheduler_config
    best_test_scheduler_config = {
        shape: argmax(all_test_scheduler_config[shape]) for shape in test_shapes
    }

    return (train_inputs, train_perf), (
        test_inputs,
        test_perf,
        test_shapes,
        best_test_scheduler_config,
    )


# Apply schedule decorator, run fusion, and profile performance
def run_profile(autotune_config, presched_fd, inputs, scheduler_config=None):
    scheduled_fd = autotune_config.custom_scheduler(presched_fd, scheduler_config)
    nvf_outputs = scheduled_fd.execute(inputs, profile=True)

    # validate correctness
    assert torch.allclose(
        nvf_outputs[0], autotune_config.eager_reference(inputs), atol=1e-2, rtol=1e-2
    )

    prof = scheduled_fd.profile()
    bandwidth = prof.kernel_profiles[0].effective_bandwidth_gbs
    time = prof.kernel_profiles[0].time_ms
    return bandwidth, time


# Given a map from scheduler configuration to predicted performance, find the
# configuration with the maximum predicted performance
def argmax(map_scheduler_config_to_perf):
    best_perf = -1
    best_scheduler_config = None
    for scheduler_config, perf in map_scheduler_config_to_perf.items():
        if perf > best_perf:
            best_perf = perf
            best_scheduler_config = scheduler_config
    return best_scheduler_config


# Given a prediction model, input_shape, and set of parameter configurations,
# find the best parameters
def find_best_parameters(clf, input_shape, scheduler_configurations):
    map_scheduler_config_to_performance = {
        scheduler_config: clf.predict(
            [[*input_shape, *flatten_configuration(scheduler_config)]]
        )
        for scheduler_config in scheduler_configurations
    }
    return argmax(map_scheduler_config_to_performance)


# Measure model performance with RMSE
def test_model_rmse(clf, script_config, autotune_config, test_data):
    test_inputs, test_perf, test_shapes, best_test_scheduler_config = test_data
    test_pred = clf.predict(test_inputs)

    # Estimate prediction error with RMSE
    import numpy as np

    test_perf = np.array(test_perf)
    print(
        "Test prediction error (RMSE)",
        np.sqrt(np.mean(np.power(test_perf - test_pred, 2))),
    )
    print("Test performance", test_perf)
    print("Test prediction", test_pred)

    print("======================= compare configurations  =======================")
    # Find best configuration for test_shapes
    print("input shape, estimate_config, actual_config, correct")
    correctness_count = 0
    mismatch_configs = []
    for shape in test_shapes:
        estimate_config = find_best_parameters(
            clf, shape, autotune_config.generate_scheduler_configurations(shape)
        )

        match_config = (
            flatten_configuration(estimate_config) == best_test_scheduler_config[shape]
        )
        if not match_config:
            mismatch_configs.append((shape, estimate_config))

        correctness_count += int(match_config)
        print(
            f"{shape}, {estimate_config}, {best_test_scheduler_config[shape]}, {match_config}"
        )
    print(
        "% of predictions match nvfuser parameters",
        correctness_count / len(test_shapes),
    )
    print(correctness_count, "out of", len(test_shapes))

    print("======================= compare performance =========================")

    for shape, estimate_config in mismatch_configs:
        inputs = autotune_config.create_inputs(shape, script_config.tensor_datatype)

        with FusionDefinition() as presched_fd:
            autotune_config.create_fusion_func(inputs)(presched_fd)

        _, est_perf = run_profile(autotune_config, presched_fd, inputs, estimate_config)
        _, nvf_perf = run_profile(autotune_config, presched_fd, inputs)
        est_perf_faster = est_perf < nvf_perf
        print(
            f"{shape} \t estimate_perf: {est_perf: .5f} \t nvfuser_perf: {nvf_perf: .5f} \t is_estimated_config_faster: {est_perf_faster}"
        )
    print("=====================================================================")


# Given a machine learning model, compare the performance of its predicted configuration
# against nvfuser on a given fusion
def test_model(clf, script_config, autotune_config):
    #  For a specific batch size, gather performance across a range of hidden sizes.
    #  Calculate performance for best predicted and nvfuser configurations. Plot a
    #  chart comparing performance using matplotlib.

    # NOTE: The matplotlib experiment plots the kernel runtime, which could be
    # different than the selected performance metric. Currently, the performance
    # metric is effective_bandwidth_gbs.

    import matplotlib.pyplot as plt
    import numpy as np

    FusionCache.reset()
    est_perfs = []
    for hidden_shape in script_config.empirical_hidden_sizes:
        inputs = autotune_config.create_inputs(
            (script_config.empirical_batch_size, hidden_shape),
            script_config.tensor_datatype,
        )

        estimate_config = find_best_parameters(
            clf,
            (script_config.empirical_batch_size, hidden_shape),
            autotune_config.generate_scheduler_configurations(
                (script_config.empirical_batch_size, hidden_shape)
            ),
        )

        with FusionDefinition() as presched_fd:
            autotune_config.create_fusion_func(inputs)(presched_fd)

        _, est_time_ms = run_profile(
            autotune_config, presched_fd, inputs, estimate_config
        )
        est_perfs.append(est_time_ms)
        print(
            f"{script_config.empirical_batch_size}, {hidden_shape}, {estimate_config}, {est_time_ms: .3f}"
        )

    FusionCache.reset()
    nvf_perfs = []
    for hidden_shape in script_config.empirical_hidden_sizes:
        inputs = autotune_config.create_inputs(
            (script_config.empirical_batch_size, hidden_shape),
            script_config.tensor_datatype,
        )

        with FusionDefinition() as presched_fd:
            autotune_config.create_fusion_func(inputs)(presched_fd)

        _, nvf_time_ms = run_profile(autotune_config, presched_fd, inputs)
        nvf_perfs.append(nvf_time_ms)
        print(
            f"{script_config.empirical_batch_size}, {hidden_shape}, {nvf_time_ms: .3f}"
        )

    # Get mean speed-up from nvfuser to empirical configurations across all input shapes.
    # Negative value mean empirical configurations are slower than nvfuser.
    print("Mean speed-up", np.mean(np.array(nvf_perfs) - np.array(est_perfs)))

    np_hidden_size = np.array(script_config.empirical_hidden_sizes)
    plt.plot(np_hidden_size, np.array(est_perfs))
    plt.plot(np_hidden_size, np.array(nvf_perfs))

    plt.xlabel("Hidden Size")
    plt.ylabel("Time(ms)")
    plt.title(
        f"Batch Size = {script_config.empirical_batch_size}, Compare Machine Learning Heuristic vs NvFuser"
    )
    plt.legend(["random_forest", "nvfuser"], loc="lower right")
    plt.savefig(
        f"{autotune_config}_empirical_batch_size_{script_config.empirical_batch_size}.png"
    )
    plt.close("all")

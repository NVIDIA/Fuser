# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import thunder
import torch
import itertools
import random
from nvfuser import FusionCache, FusionDefinition, SchedulerType, DataType
from dataclasses import dataclass


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
num_dimensions = 2
outer_shapes = [256]
inner_shapes = [2**i for i in range(5, 15)]

# We profile a range of input shapes with various configurations.
# This argument determines how much of the profiled data to keep as a test set.
test_data_percentage = 0.1

# The selected batch size for empirical and nvfuser comparison.
empirical_batch_size = 16384

# The range of hidden sizes for empirical and nvfuser comparision.
empirical_hidden_sizes = list(range(256, 28672, 256))


# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors.
def generate_parameter_configurations(num_dimensions):
    breakpoints = range(num_dimensions)
    vectorize_range = [1, 2, 4, 8]
    unroll_ranges_1d = [[1], list(range(1, 10))]
    unroll_ranges_2d = [list(range(1, 4)), list(range(1, 4))]

    configs = []
    for bp in breakpoints:
        unroll_range = unroll_ranges_1d if bp == 0 else unroll_ranges_2d
        outer_unroll_range, inner_unroll_range = unroll_range
        config = itertools.product(
            [bp], vectorize_range, outer_unroll_range, inner_unroll_range
        )
        configs.append(config)
    return itertools.chain(*configs)


# A decorator to create a pointwise fusion given some input arguments.
def create_fusion_func(inputs):
    def fusion_func(fd: FusionDefinition):
        T0 = fd.define_tensor(
            shape=[1, -1],
            contiguity=[None, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T6 = fd.ops.cast(T1, dtype=DataType.Float)
        T7 = fd.ops.cast(T0, dtype=DataType.Float)
        T8 = fd.ops.add(T6, T7)
        T9 = fd.ops.mul(T8, T8)
        T10 = fd.ops.mul(T9, T8)
        S11 = fd.define_scalar(0.500000, dtype=DataType.Double)
        T12 = fd.ops.mul(S11, T8)
        S13 = fd.define_scalar(0.0447150, dtype=DataType.Double)
        T14 = fd.ops.mul(S13, T10)
        T15 = fd.ops.add(T8, T14)
        S16 = fd.define_scalar(0.797885, dtype=DataType.Double)
        T17 = fd.ops.mul(S16, T15)
        T18 = fd.ops.tanh(T17)
        S19 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T20 = fd.ops.add(S19, T18)
        T21 = fd.ops.mul(T12, T20)
        T22 = fd.ops.cast(T21, dtype=DataType.BFloat16)
        fd.add_output(T22)

    return fusion_func


# The pytorch eager mode reference used to validating nvfuser kernel.
def eager_reference(inputs):
    return torch.nn.functional.gelu(
        inputs[0] + inputs[1].unsqueeze(0), approximate="tanh"
    )


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
            break_point, vectorize_factor, outer_unroll, inner_unroll = config
            schedule_params.break_point = break_point
            schedule_params.vectorization_factor = vectorize_factor
            schedule_params.unroll_factor_outer = outer_unroll
            schedule_params.unroll_factor_inner = inner_unroll
            schedule_params.lparams.bdimx = 128

        # Schedule fusion
        fd.sched.schedule()

    fd.schedule = inner_fn
    return fd


# Apply schedule decorator, run fusion, and profile performance
def run_profile(presched_fd, inputs, config=None):
    scheduled_fd = custom_pointwise_scheduler(presched_fd, config)
    nvf_outputs = scheduled_fd.execute(inputs, profile=True)

    # validate correctness
    assert torch.allclose(nvf_outputs[0], eager_reference(inputs), atol=1e-2, rtol=1e-2)

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
        for config in parameter_configurations
    }
    return argmax(map_config_to_performance)


# ============================ Run Experiments  ================================

# Collect data for decision tree
parameters = []
performance = []

for shape in itertools.product(outer_shapes, inner_shapes):
    print(shape)
    inputs = [
        torch.randn(1, shape[-1], dtype=torch.bfloat16, device="cuda"),
        torch.randn(*shape, dtype=torch.bfloat16, device="cuda"),
    ]

    with FusionDefinition() as presched_fd:
        create_fusion_func(inputs)(presched_fd)

    # unroll and vectorization configurations
    for config in generate_parameter_configurations(num_dimensions):
        perf_metric, _ = run_profile(presched_fd, inputs, config)
        parameters.append((*shape, *config))
        performance.append(perf_metric)

# ============================ Separate Data  ==================================

# Separate collected data into training and test sets
train_data = []
test_data = []
train_perf = []
test_perf = []
test_shapes = set()
all_test_config = {}  # key: input_shape, value: (config, perf)

for data, perf in zip(parameters, performance):
    shape = data[:num_dimensions]
    config = data[num_dimensions:]

    if shape in all_test_config:
        all_test_config[shape][config] = perf
    else:
        all_test_config[shape] = {config: perf}

    if random.random() < test_data_percentage:
        test_data.append(data)
        test_perf.append(perf)
    else:
        test_shapes.add(shape)
        train_data.append(data)
        train_perf.append(perf)

# key: input_shape, value: best_config
best_test_config = {shape: argmax(all_test_config[shape]) for shape in test_shapes}

# ========================= Train Regression Models  ===========================

# Apply decision tree regressor
# Given input shapes and scheduler parameters, predict performance metric.
from sklearn import tree

clf = tree.DecisionTreeRegressor()
clf = clf.fit(train_data, train_perf)
test_pred = clf.predict(test_data)

print("===================== measure performance rmse ========================")

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
print(
    "input shape, estimate_config:(vectorization, unroll), actual_config:(vectorization, unroll), correct"
)
correctness_count = 0
mismatch_configs = []
for shape in test_shapes:
    estimate_config = find_best_parameters(
        clf, shape, generate_parameter_configurations(num_dimensions)
    )

    match_config = estimate_config == best_test_config[shape]
    if not match_config:
        mismatch_configs.append((shape, estimate_config))

    correctness_count += int(match_config)
    print(f"{shape}, {estimate_config}, {best_test_config[shape]}, {match_config}")
print("% of predictions match nvfuser parameters", correctness_count / len(test_shapes))
print(correctness_count, "out of", len(test_shapes))

print("======================= compare performance =========================")

for shape, estimate_config in mismatch_configs:
    inputs = [
        torch.randn(1, shape[-1], dtype=torch.bfloat16, device="cuda"),
        torch.randn(*shape, dtype=torch.bfloat16, device="cuda"),
    ]

    with FusionDefinition() as presched_fd:
        create_fusion_func(inputs)(presched_fd)

    _, est_perf = run_profile(presched_fd, inputs, estimate_config)
    _, nvf_perf = run_profile(presched_fd, inputs)
    est_perf_faster = est_perf < nvf_perf
    print(
        f"{shape} \t estimate_perf:{est_perf:.5f} \t nvfuser_perf:{nvf_perf:.5f} \t is_estimated_config_faster:\t{est_perf_faster}"
    )

print("=====================================================================")

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
for hidden_shape in empirical_hidden_sizes:
    inputs = [
        torch.randn(1, hidden_shape, dtype=torch.bfloat16, device="cuda"),
        torch.randn(
            empirical_batch_size, hidden_shape, dtype=torch.bfloat16, device="cuda"
        ),
    ]
    estimate_config = find_best_parameters(
        clf,
        (empirical_batch_size, hidden_shape),
        generate_parameter_configurations(num_dimensions),
    )

    with FusionDefinition() as presched_fd:
        create_fusion_func(inputs)(presched_fd)

    _, est_time_ms = run_profile(presched_fd, inputs, estimate_config)
    est_perfs.append(est_time_ms)
    print(
        f"{empirical_batch_size}, {hidden_shape}, {estimate_config}, {est_time_ms:.3f}"
    )

FusionCache.reset()
nvf_perfs = []
for hidden_shape in empirical_hidden_sizes:
    inputs = [
        torch.randn(1, hidden_shape, dtype=torch.bfloat16, device="cuda"),
        torch.randn(
            empirical_batch_size, hidden_shape, dtype=torch.bfloat16, device="cuda"
        ),
    ]

    with FusionDefinition() as presched_fd:
        create_fusion_func(inputs)(presched_fd)

    _, nvf_time_ms = run_profile(presched_fd, inputs)
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

# =============================================================================

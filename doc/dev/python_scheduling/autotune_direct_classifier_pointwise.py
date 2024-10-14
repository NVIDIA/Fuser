# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
import random
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

# Settings for input tensor generation
num_dimensions = 2
outer_shapes = [512, 1024, 16384]
inner_shapes = [2**i for i in range(5, 15)]

vectorize_range = [1, 2, 4]
map_vectorize_factor_to_class = {
    vectorize_range[output_class]: output_class
    for output_class in range(len(vectorize_range))
}
unroll_range = list(range(1, 10))

# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors.
parameter_configurations = [vectorize_range, unroll_range]

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
        c0 = fd.define_scalar(3.0)
        t2 = fd.ops.add(t0, t1)
        t3 = fd.ops.mul(t2, c0)
        fd.add_output(t3)

    return fusion_func


# The pytorch eager mode reference used to validating nvfuser kernel.
def eager_reference(inputs):
    return (inputs[0] + inputs[1]) * 3


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
    assert torch.allclose(nvf_outputs[0], eager_reference(inputs))

    prof = scheduled_fd.profile()
    bandwidth = prof.kernel_profiles[0].effective_bandwidth_gbs
    time = prof.kernel_profiles[0].time_ms
    return bandwidth, time


def find_max_config(map_config_to_perf):
    best_perf = -1
    best_config = None
    for config, perf in map_config_to_perf.items():
        if perf > best_perf:
            best_perf = perf
            best_config = config
    return best_config, best_perf


def argmax(map_config_to_perf):
    return find_max_config(map_config_to_perf)[0]


# outputs = (vectorize_factor, unroll_factor, performance_metric)
def map_regression_outputs_to_classes(outputs):
    vectorize_factor, unroll_factor, performance_metric = outputs
    # map vectorization factor to output class via dictionary
    # map performance metric by using first two significant digits
    return (
        map_vectorize_factor_to_class[vectorize_factor],
        unroll_factor,
        round(performance_metric / 100),
    )


# Given a prediction model, input_shape, and set of parameter configurations,
# find the best parameters
def predict_best_parameters(predictor, input_shape):
    result = predictor.predict([input_shape])
    vectorize_factor_class, unroll_factor, perf_metric = result[0, :]
    return vectorize_range[vectorize_factor_class], unroll_factor


# ============================ Run Experiments  ================================

# Collect data for decision tree
parameters = []
performance = []

for shape in itertools.product(outer_shapes, inner_shapes):
    print(shape)
    inputs = [
        torch.randn(*shape, device="cuda"),
        torch.randn(*shape, device="cuda"),
    ]

    with FusionDefinition() as presched_fd:
        create_fusion_func(inputs)(presched_fd)

    # unroll and vectorization configurations
    for config in itertools.product(vectorize_range, unroll_range):
        perf_metric, _ = run_profile(presched_fd, inputs, config)
        parameters.append((*shape, *config))
        performance.append(perf_metric)

# ============================ Separate Data  ==================================

# Separate collected data into training and test sets
train_data = []
test_data = []
train_output = []
test_config = []
test_perf = []
test_shapes = set()
all_config = {}  # key: input_shape, value: (config, perf)

for data, perf in zip(parameters, performance):
    shape = data[:num_dimensions]
    config = data[num_dimensions:]

    if shape in all_config:
        all_config[shape][config] = perf
    else:
        all_config[shape] = {config: perf}

    best_config, best_perf = find_max_config(all_config[shape])

    if random.random() < test_data_percentage:
        test_shapes.add(shape)
        test_data.append(shape)
        test_config.append(best_config)
        test_perf.append(round(best_perf / 100))
    else:
        train_data.append(shape)
        train_output.append(
            map_regression_outputs_to_classes([*best_config, best_perf])
        )

# key: input_shape, value: best_config
best_test_config = {shape: argmax(all_config[shape]) for shape in test_shapes}

# ========================= Train Regression Models  ===========================

# Apply decision tree regressor
# Given input shapes and scheduler parameters, predict performance metric.
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_output)
test_pred = clf.predict(test_data)

print("===================== measure performance rmse ========================")

# Estimate prediction error with RMSE
import numpy as np

# last output is performance metric
test_pred_perf = test_pred[:, -1]
test_perf = np.array(test_perf)
print(
    "Test prediction error (RMSE)",
    np.sqrt(np.mean(np.power(test_perf - test_pred_perf, 2))),
)
print("Test performance", test_perf)
print("Test prediction", test_pred_perf)

print("======================= compare configurations  =======================")
# Find best configuration for test_shapes
print(
    "input shape, estimate_config:(vectorization, unroll), actual_config:(vectorization, unroll), correct"
)
correctness_count = 0
mismatch_configs = []
for shape in test_shapes:
    estimate_config = predict_best_parameters(clf, shape)

    match_config = estimate_config == best_test_config[shape]
    if not match_config:
        mismatch_configs.append((shape, estimate_config))

    correctness_count += int(match_config)
    print(f"{shape}, {estimate_config}, {best_test_config[shape]}, {match_config}")
print("% of predictions match nvfuser parameters", correctness_count / len(test_shapes))
print(correctness_count, "out of", len(test_shapes))

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
        torch.randn(empirical_batch_size, hidden_shape, device="cuda"),
        torch.randn(empirical_batch_size, hidden_shape, device="cuda"),
    ]
    estimate_config = predict_best_parameters(clf, (empirical_batch_size, hidden_shape))

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
        torch.randn(empirical_batch_size, hidden_shape, device="cuda"),
        torch.randn(empirical_batch_size, hidden_shape, device="cuda"),
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
plt.savefig(f"pointwise_direct_batchsize{empirical_batch_size}.png")

# =============================================================================

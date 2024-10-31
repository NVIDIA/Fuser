# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
import random
from nvfuser import FusionCache, FusionDefinition, SchedulerType, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from copy import deepcopy

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
outer_shapes = [256, 1024, 4096, 16384]
inner_shapes = [2**i for i in range(10, 15)]

# For pointwise scheduler, we test the cartesian product of vectorization and
# cta_size factors.
parameter_configurations = [
    vectorize_range := [1, 2, 4, 8],
    threads_per_cta_range := list(range(128, 288, 32)),
]

# We profile a range of input shapes with various configurations.
# This argument determines how much of the profiled data to keep as a test set.
test_data_percentage = 0.1

# The selected batch size for empirical and nvfuser comparison.
empirical_batch_size = 512

# The range of hidden sizes for empirical and nvfuser comparision.
empirical_hidden_sizes = list(range(1024, 28672, 256))

# NOTE For 24gb memory limit
# empirical_hidden_sizes = list(range(256, 22784, 256))


def create_inputs(shape):
    """Create input arguments for nvfuser fusion and eager mode"""
    a = torch.randn(*shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    grads = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(
        shape[1], dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    bias = torch.randn(
        shape[1], dtype=torch.bfloat16, device="cuda", requires_grad=True
    )

    eps = 1e-5
    mean = a.to(torch.float).mean(dim=-1)
    variance = a.to(torch.float).var(dim=-1, unbiased=False)
    invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)

    nvf_inputs = [a, grads, mean, invstd, weights]
    eager_inputs = [a, weights, bias, grads]
    return nvf_inputs, eager_inputs


# A decorator to create a pointwise fusion given some input arguments.
def create_fusion_func(inputs):
    PROMOTE_DTYPES = [DataType.BFloat16, DataType.Half]
    dtype = torch_dtype_to_nvfuser_dtype(inputs[0].dtype)

    def fusion_func(fd: FusionDefinition):
        T0 = fd.define_tensor(
            shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
        )
        T1 = fd.define_tensor(
            shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
        )

        T2 = fd.define_tensor(
            shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
        )
        T3 = fd.define_tensor(
            shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
        )

        T4 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

        if dtype in PROMOTE_DTYPES:
            T0 = fd.ops.cast(T0, dtype=DataType.Float)
            T1 = fd.ops.cast(T1, dtype=DataType.Float)
            T4 = fd.ops.cast(T4, dtype=DataType.Float)

        V8 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
        T9 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
        V12 = T0.shape()
        T13 = fd.ops.broadcast_in_dim(T9, shape=V12, broadcast_dims=[0, 1])
        T14 = fd.ops.sub(T0, T13)

        T18 = fd.ops.broadcast_in_dim(T3, shape=V12, broadcast_dims=[0, 1])
        T19 = fd.ops.mul(T14, T18)

        T23 = fd.ops.broadcast_in_dim(T4, shape=V12, broadcast_dims=[1])
        T28 = fd.ops.sum(T1, dims=[0], keepdim=False, dtype=DataType.Null)

        T30 = fd.ops.mul(T1, T23)
        T31 = fd.ops.mul(T1, T19)
        T32 = fd.ops.sum(T31, dims=[0], keepdim=False, dtype=DataType.Null)

        T34 = fd.ops.mul(T30, T18)
        T35 = fd.ops.mul(T30, T14)
        T36 = fd.ops.sum(T35, dims=[1], keepdim=False, dtype=DataType.Null)

        T40 = fd.ops.broadcast_in_dim(T36, shape=V8, broadcast_dims=[0])
        T41 = fd.ops.neg(T34)
        T42 = fd.ops.sum(T41, dims=[1], keepdim=False, dtype=DataType.Null)
        T46 = fd.ops.broadcast_in_dim(T42, shape=V8, broadcast_dims=[0])
        S47 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T48 = fd.ops.mul(S47, T40)
        S49 = fd.define_scalar(3.00000, dtype=DataType.Double)
        T50 = fd.ops.pow(T3, S49)
        T51 = fd.ops.mul(T48, T50)
        T54 = fd.ops.sum(T46, dims=[1], keepdim=False, dtype=DataType.Null)
        T55 = fd.ops.sum(T51, dims=[1], keepdim=False, dtype=DataType.Null)

        T59 = fd.ops.broadcast_in_dim(T55, shape=V8, broadcast_dims=[0])
        T63 = fd.ops.broadcast_in_dim(T59, shape=V12, broadcast_dims=[0, 1])
        T67 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
        T71 = fd.ops.broadcast_in_dim(T67, shape=V12, broadcast_dims=[0, 1])

        S72 = fd.define_scalar(2.00000, dtype=DataType.Double)
        T73 = fd.ops.mul(S72, T63)
        T74 = fd.ops.sub(T0, T71)
        T75 = fd.ops.mul(T73, T74)

        S77 = fd.ops.reciprocal(T0.size(1))
        T78 = fd.ops.mul(T75, S77)
        T82 = fd.ops.broadcast_in_dim(T54, shape=V8, broadcast_dims=[0])
        T86 = fd.ops.broadcast_in_dim(T82, shape=V12, broadcast_dims=[0, 1])
        T88 = fd.ops.mul(S77, T86)
        T89 = fd.ops.add(T78, T88)
        T90 = fd.ops.add(T34, T89)

        if dtype in PROMOTE_DTYPES:
            T28 = fd.ops.cast(T28, dtype=dtype)
            T90 = fd.ops.cast(T90, dtype=dtype)
            T32 = fd.ops.cast(T32, dtype=dtype)

        fd.add_output(T90)
        fd.add_output(T32)
        fd.add_output(T28)

    return fusion_func


# The pytorch eager mode reference used to validating nvfuser kernel.
def eager_reference(inputs):
    inputs_cloned = deepcopy(inputs)
    a, weights, bias, grad_output = inputs_cloned
    eager_output = torch.nn.functional.layer_norm(
        a.to(torch.double),
        a.shape[1:],
        weight=weights.to(torch.double),
        bias=bias.to(torch.double),
    )
    grad_output = grad_output.to(torch.double)
    eager_output.backward(grad_output)
    return [a.grad, weights.grad, bias.grad]


# ============================ Function Definitions ============================


# Apply scheduler with custom parameters using decorator
def custom_persistent_scheduler(fd, config):
    def inner_fn():
        # Check if compatible with persistent scheduler
        status, _ = fd.sched.can_schedule(SchedulerType.inner_outer_persistent)
        assert status

        # Modify original parameters
        if config is not None:
            hyperparameters = fd.sched.schedule_hyperparameters()
            vectorize_factor, threads_per_block = config
            hyperparameters.vectorize_factor = vectorize_factor
            hyperparameters.threads_per_block_min = threads_per_block
            hyperparameters.threads_per_block_max = threads_per_block

        # Schedule fusion
        fd.sched.schedule(SchedulerType.inner_outer_persistent)

    fd.schedule = inner_fn
    return fd


# Apply schedule decorator, run fusion, and profile performance
def run_profile(presched_fd, nvf_inputs, eager_inputs, config=None):
    scheduled_fd = custom_persistent_scheduler(presched_fd, config)
    nvf_outputs = scheduled_fd.execute(nvf_inputs, profile=True)

    # validate correctness
    ref_outputs = eager_reference(eager_inputs)
    for nvf_out, ref_out in zip(nvf_outputs, ref_outputs):
        assert torch.allclose(nvf_out, ref_out, atol=1e-1, rtol=1e-1)

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

# Collect data for decision tree
parameters = []
performance = []

for shape in itertools.product(outer_shapes, inner_shapes):
    print(shape)
    nvf_inputs, eager_inputs = create_inputs(shape)

    with FusionDefinition() as presched_fd:
        create_fusion_func(nvf_inputs)(presched_fd)

    # vectorization and threads_per_cta configurations
    for config in itertools.product(*parameter_configurations):
        perf_metric, _ = run_profile(presched_fd, nvf_inputs, eager_inputs, config)
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
    "input shape, estimate_config:(vectorization, cta_size), actual_config:(vectorization, cta_size), correct"
)
correctness_count = 0
mismatch_configs = []
for shape in test_shapes:
    estimate_config = find_best_parameters(clf, shape, parameter_configurations)

    match_config = estimate_config == best_test_config[shape]
    if not match_config:
        mismatch_configs.append((shape, estimate_config))

    correctness_count += int(match_config)
    print(f"{shape}, {estimate_config}, {best_test_config[shape]}, {match_config}")
print("% of predictions match nvfuser parameters", correctness_count / len(test_shapes))
print(correctness_count, "out of", len(test_shapes))

print("======================= compare performance =========================")

for shape, estimate_config in mismatch_configs:
    nvf_inputs, eager_inputs = create_inputs(shape)

    with FusionDefinition() as presched_fd:
        create_fusion_func(nvf_inputs)(presched_fd)

    _, est_perf = run_profile(presched_fd, nvf_inputs, eager_inputs, estimate_config)
    _, nvf_perf = run_profile(presched_fd, nvf_inputs, eager_inputs)
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

# Avoid reusing any cached, user-scheduled fusions to have a clean run.
FusionCache.reset()
est_perfs = []
for hidden_shape in empirical_hidden_sizes:
    nvf_inputs, eager_inputs = create_inputs((empirical_batch_size, hidden_shape))
    estimate_config = find_best_parameters(
        clf, (empirical_batch_size, hidden_shape), parameter_configurations
    )

    with FusionDefinition() as presched_fd:
        create_fusion_func(nvf_inputs)(presched_fd)

    _, est_time_ms = run_profile(presched_fd, nvf_inputs, eager_inputs, estimate_config)
    est_perfs.append(est_time_ms)
    print(
        f"decision tree: {empirical_batch_size}, {hidden_shape}, {estimate_config}, {est_time_ms:.3f}"
    )

FusionCache.reset()
nvf_perfs = []
for hidden_shape in empirical_hidden_sizes:
    nvf_inputs, eager_inputs = create_inputs((empirical_batch_size, hidden_shape))
    estimate_config = find_best_parameters(
        clf, (empirical_batch_size, hidden_shape), parameter_configurations
    )

    with FusionDefinition() as presched_fd:
        create_fusion_func(nvf_inputs)(presched_fd)

    _, nvf_time_ms = run_profile(presched_fd, nvf_inputs, eager_inputs)
    nvf_perfs.append(nvf_time_ms)
    print(f"nvfuser: {empirical_batch_size}, {hidden_shape}, {nvf_time_ms:.3f}")

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
plt.savefig(f"persistent_inner_outer_empirical_batchsize{empirical_batch_size}.png")

# =============================================================================

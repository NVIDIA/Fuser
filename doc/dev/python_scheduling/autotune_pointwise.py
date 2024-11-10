# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
import math
import collections
import random
from nvfuser import FusionCache, FusionDefinition, SchedulerType, DataType
from dataclasses import dataclass
from enum import Enum


# ============================ Description ============================

# This script defines four pointwise fusions:
#
# 1. GELU with Outer-Broadcast Bias Addition
#    y = gelu(x + bias[broadcast, i], approximate='tanh')
#
# 2. SILU with Pointwise Multiplication
#    z = silu(x) * y
#
# 3. Inner-Broadcast Addition
#    y = x + y[i, broadcast]
#
# 4. Pointwise Multiplication
#    z = x + y
#
# Script Sequence:
#
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
#
# The selected performance metric is effective_bandwidth_gbs. The empirical
# scheduler selects the configuration that has the highest predicted
# effective_bandwidth_gbs.

# ============================ Configurations ============================


class FUSION(Enum):
    GELU_BIAS = 1
    SILU_MUL = 2
    BCAST_ADD = 3
    MUL = 4


@dataclass
class ScriptConfiguration:
    # Which Fusion to profile?
    selected_fusion: FUSION

    # Settings for input tensor generation
    num_dimensions: int
    outer_shapes: [int]
    inner_shapes: [int]
    tensor_datatype: torch.dtype

    # We profile a range of input shapes with various configurations.
    # This argument determines how much of the profiled data to keep as a test set.
    test_data_percentage: [float]

    # The selected batch size for empirical and nvfuser comparison.
    empirical_batch_size: [int]

    # The range of hidden sizes for empirical and nvfuser comparision.
    empirical_hidden_sizes: [int]


# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors.
def generate_scheduler_configurations(num_dimensions):
    def _named_product(**items):
        scheduler_config = collections.namedtuple("Configuration", items.keys())
        return itertools.starmap(scheduler_config, itertools.product(*items.values()))

    warp_size = 32
    warp_group = warp_size * 4
    # limited to a maximum of 128 threads because of pointwise scheduler
    max_threads_per_cta = 128
    threads_per_cta = list(range(warp_group, max_threads_per_cta + 1, warp_group))

    scheduler_configs = []
    for bp in range(num_dimensions):
        for num_threads in threads_per_cta:
            if bp == 0:
                # 1D scheduler configurations
                bdim_shapes = [(num_threads, 1)]
                outer_unroll_range = [1]
                # unroll_factor is between [1, 9]
                inner_unroll_range = range(1, 10)
            else:
                # 2D scheduler configurations
                max_bdimy = num_threads // warp_size
                log2_max_bdimy = int(math.log2(max_bdimy))
                bdimy_configs = [
                    2**log_bdimy for log_bdimy in range(1, log2_max_bdimy + 1)
                ]

                bdim_shapes = [
                    (max(warp_size, num_threads // bdimy), bdimy)
                    for bdimy in bdimy_configs
                ]
                # total_unroll_factor is between [1, 25] given that outer and
                # inner unroll factors are between [1, 5].
                outer_unroll_range = range(1, 6)
                inner_unroll_range = range(1, 6)

            scheduler_config = _named_product(
                break_point=[bp],
                bdim=bdim_shapes,
                vectorize_factor=[1, 2, 4, 8],
                outer_unroll=outer_unroll_range,
                inner_unroll=inner_unroll_range,
            )
            scheduler_configs.append(scheduler_config)
    return itertools.chain(*scheduler_configs)


def create_inputs(which_fusion, shape, tensor_datatype):
    def outer_bcast():
        return [
            torch.randn(1, shape[-1], dtype=tensor_datatype, device="cuda"),
            torch.randn(*shape, dtype=tensor_datatype, device="cuda"),
        ]

    def inner_bcast():
        return [
            torch.randn(shape[0], 1, dtype=tensor_datatype, device="cuda"),
            torch.randn(*shape, dtype=tensor_datatype, device="cuda"),
        ]

    def full():
        return [
            torch.randn(*shape, dtype=tensor_datatype, device="cuda"),
            torch.randn(*shape, dtype=tensor_datatype, device="cuda"),
        ]

    if which_fusion == FUSION.GELU_BIAS:
        return outer_bcast()
    elif which_fusion == FUSION.SILU_MUL or which_fusion == FUSION.MUL:
        return full()
    elif which_fusion == FUSION.BCAST_ADD:
        return inner_bcast()
    else:
        assert False


# A decorator to create a pointwise fusion given some input arguments.
def create_fusion_func(which_fusion, inputs):
    def gelu_bias(fd: FusionDefinition):
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

    def silu_mul(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
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
        T2 = fd.ops.cast(T0, dtype=DataType.Float)
        T3 = fd.ops.neg(T2)
        T4 = fd.ops.exp(T3)
        S5 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T6 = fd.ops.add(S5, T4)
        T7 = fd.ops.reciprocal(T6)
        T8 = fd.ops.mul(T2, T7)
        T9 = fd.ops.cast(T1, dtype=DataType.Float)
        T10 = fd.ops.mul(T8, T9)
        T11 = fd.ops.cast(T10, dtype=DataType.BFloat16)
        fd.add_output(T11)

    def bcast_add(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, 1],
            contiguity=[True, None],
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
        T2 = fd.ops.cast(T0, dtype=DataType.Float)
        T3 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.add(T2, T3)
        T5 = fd.ops.cast(T4, dtype=DataType.BFloat16)
        fd.add_output(T5)

    def mul(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
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
        T2 = fd.ops.cast(T0, dtype=DataType.Float)
        T3 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.mul(T2, T3)
        T5 = fd.ops.cast(T4, dtype=DataType.BFloat16)
        fd.add_output(T5)

    if which_fusion == FUSION.GELU_BIAS:
        return gelu_bias
    elif which_fusion == FUSION.SILU_MUL:
        return silu_mul
    elif which_fusion == FUSION.BCAST_ADD:
        return bcast_add
    elif which_fusion == FUSION.MUL:
        return mul
    else:
        assert False


# The pytorch eager mode reference used to validating nvfuser kernel.
def eager_reference(which_fusion, inputs):
    def gelu_bias(inputs):
        return torch.nn.functional.gelu(
            inputs[0] + inputs[1].unsqueeze(0), approximate="tanh"
        )

    def silu_mul(inputs):
        return torch.nn.functional.silu(inputs[0]) * inputs[1]

    def bcast_add(inputs):
        return inputs[0] + inputs[1]

    def mul(inputs):
        return inputs[0] * inputs[1]

    if which_fusion == FUSION.GELU_BIAS:
        return gelu_bias(inputs)
    elif which_fusion == FUSION.SILU_MUL:
        return silu_mul(inputs)
    elif which_fusion == FUSION.BCAST_ADD:
        return bcast_add(inputs)
    elif which_fusion == FUSION.MUL:
        return mul(inputs)
    else:
        assert False


# ============================ Function Definitions ============================


# Apply scheduler with custom parameters using decorator
def custom_pointwise_scheduler(fd, scheduler_config):
    def inner_fn():
        # Check if compatible with pointwise scheduler
        status, _ = fd.sched.can_schedule(SchedulerType.pointwise)
        assert status

        schedule_params = fd.sched.compute_pointwise_heuristics()

        # Modify original parameters
        if scheduler_config is not None:
            schedule_params.break_point = scheduler_config.break_point
            schedule_params.vectorization_factor = scheduler_config.vectorize_factor
            schedule_params.unroll_factor_outer = scheduler_config.outer_unroll
            schedule_params.unroll_factor_inner = scheduler_config.inner_unroll
            schedule_params.lparams.bdimx = scheduler_config.bdim[0]
            schedule_params.lparams.bdimy = scheduler_config.bdim[1]

        # Schedule fusion
        fd.sched.schedule()

    fd.schedule = inner_fn
    return fd


# Apply schedule decorator, run fusion, and profile performance
def run_profile(which_fusion, presched_fd, inputs, scheduler_config=None):
    scheduled_fd = custom_pointwise_scheduler(presched_fd, scheduler_config)
    nvf_outputs = scheduled_fd.execute(inputs, profile=True)

    # validate correctness
    assert torch.allclose(
        nvf_outputs[0], eager_reference(which_fusion, inputs), atol=1e-2, rtol=1e-2
    )

    prof = scheduled_fd.profile()
    bandwidth = prof.kernel_profiles[0].effective_bandwidth_gbs
    time = prof.kernel_profiles[0].time_ms
    return bandwidth, time


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
def find_best_parameters(predictor, input_shape, scheduler_configurations):
    map_scheduler_config_to_performance = {
        scheduler_config: predictor.predict(
            [[*input_shape, *flatten_configuration(scheduler_config)]]
        )
        for scheduler_config in scheduler_configurations
    }
    return argmax(map_scheduler_config_to_performance)


# Converted NamedTuple to a Tuple. It flattens nested tuples. The function is
# used for compatibility with machine learning model.
def flatten_configuration(scheduler_config):
    new_scheduler_config = []
    for item in scheduler_config:
        if type(item) is tuple:
            new_scheduler_config.extend(item)
        else:
            new_scheduler_config.append(item)
    return tuple(new_scheduler_config)


# Collect data for decision tree
def collect_data(script_config):
    # Collect data for decision tree
    parameters = []
    performance = []

    for shape in itertools.product(
        script_config.outer_shapes, script_config.inner_shapes
    ):
        print(shape)
        inputs = create_inputs(
            script_config.selected_fusion, shape, script_config.tensor_datatype
        )

        with FusionDefinition() as presched_fd:
            create_fusion_func(script_config.selected_fusion, inputs)(presched_fd)

        # unroll and vectorization configurations
        for parameter_config in generate_scheduler_configurations(
            script_config.num_dimensions
        ):
            perf_metric, _ = run_profile(
                script_config.selected_fusion, presched_fd, inputs, parameter_config
            )
            parameters.append((*shape, *flatten_configuration(parameter_config)))
            performance.append(perf_metric)
    return parameters, performance


# Separate collected data into training and test sets
def separate_data(script_config, parameters, performance):
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


# Measure model performance with RMSE
def test_model_rmse(clf, script_config, test_data):
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
            clf, shape, generate_scheduler_configurations(script_config.num_dimensions)
        )
        flattened_estimate_config = flatten_configuration(estimate_config)

        match_config = flattened_estimate_config == best_test_scheduler_config[shape]
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
        inputs = create_inputs(
            script_config.selected_fusion, shape, script_config.tensor_datatype
        )

        with FusionDefinition() as presched_fd:
            create_fusion_func(script_config.selected_fusion, inputs)(presched_fd)

        _, est_perf = run_profile(
            script_config.selected_fusion, presched_fd, inputs, estimate_config
        )
        _, nvf_perf = run_profile(script_config.selected_fusion, presched_fd, inputs)
        est_perf_faster = est_perf < nvf_perf
        print(
            f"{shape} \t estimate_perf:{est_perf:.5f} \t nvfuser_perf:{nvf_perf:.5f} \t is_estimated_config_faster:\t{est_perf_faster}"
        )
    print("=====================================================================")


# Given a decision tree model, compare the performance of its predicted configuration
# against nvfuser on a given fusion
def test_model(clf, script_config):
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
        inputs = create_inputs(
            script_config.selected_fusion,
            (script_config.empirical_batch_size, hidden_shape),
            script_config.tensor_datatype,
        )

        estimate_config = find_best_parameters(
            clf,
            (script_config.empirical_batch_size, hidden_shape),
            generate_scheduler_configurations(script_config.num_dimensions),
        )

        with FusionDefinition() as presched_fd:
            create_fusion_func(script_config.selected_fusion, inputs)(presched_fd)

        _, est_time_ms = run_profile(
            script_config.selected_fusion, presched_fd, inputs, estimate_config
        )
        est_perfs.append(est_time_ms)
        print(
            f"{script_config.empirical_batch_size}, {hidden_shape}, {estimate_config}, {est_time_ms:.3f}"
        )

    FusionCache.reset()
    nvf_perfs = []
    for hidden_shape in script_config.empirical_hidden_sizes:
        inputs = create_inputs(
            script_config.selected_fusion,
            (script_config.empirical_batch_size, hidden_shape),
            script_config.tensor_datatype,
        )

        with FusionDefinition() as presched_fd:
            create_fusion_func(script_config.selected_fusion, inputs)(presched_fd)

        _, nvf_time_ms = run_profile(script_config.selected_fusion, presched_fd, inputs)
        nvf_perfs.append(nvf_time_ms)
        print(
            f"{script_config.empirical_batch_size}, {hidden_shape}, {nvf_time_ms:.3f}"
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
        f"Batch Size = {script_config.empirical_batch_size}, Compare Decision Tree Heuristic vs NvFuser"
    )
    plt.legend(["decision_tree", "nvfuser"], loc="lower right")
    plt.savefig(
        f"pointwise_{script_config.selected_fusion.name}_empirical_batch_size_{script_config.empirical_batch_size}.png"
    )
    plt.close("all")


# Run sequence of steps to collect data, train and test model
def main():
    # ====================== Setup Script Configuration  =======================
    script_config = ScriptConfiguration(
        selected_fusion=FUSION.GELU_BIAS,
        num_dimensions=2,
        outer_shapes=[16384],
        inner_shapes=[128, 1024, 4096, 16384],
        tensor_datatype=torch.bfloat16,
        test_data_percentage=0.1,
        empirical_batch_size=16384,
        empirical_hidden_sizes=list(range(256, 32768, 256)),
    )

    # ============================ Run Experiments  ============================

    parameters, performance = collect_data(script_config)

    # ============================ Separate Data  ==============================

    train_data, test_data = separate_data(script_config, parameters, performance)

    # ========================= Train Regression Models  =======================

    # Apply decision tree regressor
    # Given input shapes and scheduler parameters, predict performance metric.
    from sklearn import ensemble

    train_inputs, train_perf = train_data
    clf = ensemble.RandomForestRegressor()
    clf = clf.fit(train_inputs, train_perf)

    # ========================= Test Regression Models  ========================
    test_model_rmse(clf, script_config, test_data)
    test_model(clf, script_config)


if __name__ == "__main__":
    main()

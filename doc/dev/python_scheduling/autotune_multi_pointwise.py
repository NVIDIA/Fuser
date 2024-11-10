# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import thunder
import torch
import itertools
import math
import collections
import random
from nvfuser import FusionCache, FusionDefinition, SchedulerType, DataType
from dataclasses import dataclass
from enum import Enum


# ============================ Description ============================
#
# 1. Define a nvfuser fusion and its pytorch eager mode reference.
#
# 2. Define the scheduler's configuration space.
#
# 3. Define features for the scheduler.
#
# 4. Profile the CUDA kernel performance by iterating over a set of input
# arguments and scheduler configurations.
#
# 5. Train a regression model to predict the desired performance metric given
# some input arguments, fusion features, and a scheduler configuration.
#
# 6. Given a new fusion and input arguments.
#  - Create features for the scheduler given fusion and input arguments.
#  - Predict performance using trained regression model for all scheduler
#    configurations.
#  - Select the configuration with highest predicted performance to run fusion.
#
# The selected performance metric is effective_bandwidth_gbs. The empirical
# scheduler selects the configuration that has the highest predicted
# effective_bandwidth_gbs.
#
# ============================ Configurations ============================


class FUSION(Enum):
    GELU_BIAS = 1
    SILU_MUL = 2
    BCAST_ADD = 3
    MUL = 4


# Settings for input tensor generation
num_dimensions = 2
outer_shapes = [256]
inner_shapes = [2**i for i in range(8, 15)]

# We profile a range of input shapes with various configurations.
# This argument determines how much of the profiled data to keep as a test set.
test_data_percentage = 0.0

# The selected batch size for empirical and nvfuser comparison.
empirical_batch_size = 16384

# The range of hidden sizes for empirical and nvfuser comparision.
empirical_hidden_sizes = list(range(256, 28672, 256))


# For pointwise scheduler, we test the cartesian product of vectorization and
# unroll factors.
def generate_parameter_configurations(num_dimensions):
    def _named_product(**items):
        config = collections.namedtuple("Configuration", items.keys())
        return itertools.starmap(config, itertools.product(*items.values()))

    warp_size = 32
    warp_group = warp_size * 4
    # limited to a maximum of 128 threads because of pointwise scheduler
    max_threads_per_cta = 128
    threads_per_cta = list(range(warp_group, max_threads_per_cta + 1, warp_group))

    configs = []
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

            config = _named_product(
                break_point=[bp],
                bdim=bdim_shapes,
                vectorize_factor=[1, 2, 4, 8],
                outer_unroll=outer_unroll_range,
                inner_unroll=inner_unroll_range,
            )
            configs.append(config)
    return itertools.chain(*configs)


def create_inputs(which_fusion, shape):
    def outer_bcast():
        return [
            torch.randn(1, shape[-1], dtype=torch.bfloat16, device="cuda"),
            torch.randn(*shape, dtype=torch.bfloat16, device="cuda"),
        ]

    def inner_bcast():
        return [
            torch.randn(shape[0], 1, dtype=torch.bfloat16, device="cuda"),
            torch.randn(*shape, dtype=torch.bfloat16, device="cuda"),
        ]

    def full():
        return [
            torch.randn(*shape, dtype=torch.bfloat16, device="cuda"),
            torch.randn(*shape, dtype=torch.bfloat16, device="cuda"),
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


# Given fusion, configuration, and input shapes, create features for decision
# tree describing the fusion.
#  1. Is the pointwise schedule using a 1D or 2D schedule?
#     is_2d_schedule = (bp > 0)
#  2. Calculate broadcast multiples and number of elements for the lhs and rhs
#     of break_point.
#  3. Calculate bytes per lhs or rhs using broadcast multiples and number of
#     elements for lhs and rhs of break_point.
#  4. Get arithmetic intensity of fusion.
#     arithmetic_intensity = (number of operations) divided by
#                            (number of elements in input and output tensors)
# Feature Summary:
#  1. is_2d_schedule : [0, 1] bool
#  2. lhs_bytes_gbs : float
#  2. rhs_bytes_gbs : float
#  4. arithmetic_intensity : float
def construct_features(which_fusion, configuration, inputs):
    # Create meta tensor for output tensor.
    def _create_output_meta_tensors(inputs):
        reference_tensor = list(inputs[0].shape)
        reference_dtype_size = 0
        reference_dtype = None

        # Create full tensor shape from inputs
        for t in inputs:
            if type(t) is not torch.Tensor:
                continue

            for idx, dim_size in enumerate(t.shape):
                if t.dtype.itemsize > reference_dtype_size:
                    reference_dtype_size = t.dtype.itemsize
                    reference_dtype = t.dtype
                reference_tensor[idx] = max(reference_tensor[idx], dim_size)

        return [torch.empty(reference_tensor, dtype=reference_dtype, device="meta")]

    # Calculate broadcast multiple and number of elements for lhs and rhs of
    # broadcast.
    def _calculate_break_point_features(break_point, inputs):
        lhs_bcast_multiples = 0
        rhs_bcast_multiples = 0
        lhs_num_elem = 0
        rhs_num_elem = 0

        outputs = _create_output_meta_tensors(inputs)
        for t in itertools.chain(inputs, outputs):
            for idx, dim_size in enumerate(t.shape):
                value = t.dtype.itemsize if dim_size > 1 else 0

                if idx < break_point:
                    lhs_num_elem += dim_size
                    lhs_bcast_multiples += value
                else:
                    rhs_num_elem += dim_size
                    rhs_bcast_multiples += value
        if lhs_num_elem > 0:
            lhs_bytes = math.log2(lhs_bcast_multiples * lhs_num_elem)
        else:
            lhs_bytes = 0

        if rhs_num_elem > 0:
            rhs_bytes = math.log2(rhs_bcast_multiples * rhs_num_elem)
        else:
            rhs_bytes = 0
        return lhs_bytes, rhs_bytes

    # Get arithmetic intensity given fusion.
    def _arithmetic_intensity(which_fusion):
        # The memory contribution of broadcasted tensor is small.
        epsilon = 5e-5
        if which_fusion == FUSION.GELU_BIAS:
            return 5.0 + epsilon
        elif which_fusion == FUSION.SILU_MUL:
            return 2.0
        elif which_fusion == FUSION.BCAST_ADD:
            return 0.5 + epsilon
        elif which_fusion == FUSION.MUL:
            return 1.0 / 3.0
        else:
            assert False

    is_2d_schedule = float(configuration.break_point > 0)
    log2_lhs_bytes, log2_rhs_bytes = _calculate_break_point_features(
        configuration.break_point, inputs
    )
    arithmetic_intensity = _arithmetic_intensity(which_fusion)
    return [is_2d_schedule, log2_lhs_bytes, log2_rhs_bytes, arithmetic_intensity]


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
            schedule_params.break_point = config.break_point
            schedule_params.vectorization_factor = config.vectorize_factor
            schedule_params.unroll_factor_outer = config.outer_unroll
            schedule_params.unroll_factor_inner = config.inner_unroll
            schedule_params.lparams.bdimx = config.bdim[0]
            schedule_params.lparams.bdimy = config.bdim[1]

        # Schedule fusion
        fd.sched.schedule()

    fd.schedule = inner_fn
    return fd


# Apply schedule decorator, run fusion, and profile performance
def run_profile(which_fusion, presched_fd, inputs, config=None):
    scheduled_fd = custom_pointwise_scheduler(presched_fd, config)
    nvf_outputs = scheduled_fd.execute(inputs, profile=True)

    # validate correctness
    assert torch.allclose(
        nvf_outputs[0], eager_reference(which_fusion, inputs), atol=1e-2, rtol=1e-2
    )

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
def find_best_parameters(predictor, which_fusion, parameter_configurations, inputs):
    map_config_to_performance = {}
    for config in parameter_configurations:
        features = construct_features(which_fusion, config, inputs)
        map_config_to_performance[config] = predictor.predict(
            [[*features, *flatten_config(config)]]
        )
    return argmax(map_config_to_performance)


# Converted NamedTuple to a Tuple. It flattens nested tuples. The function is
# used for compatibility with machine learning model.
def flatten_config(config):
    new_config = []
    for item in config:
        if type(item) is tuple:
            new_config.extend(item)
        else:
            new_config.append(item)
    return tuple(new_config)


# Given a decision tree model, compare the performance of its predicted configuration
# against nvfuser on a given fusion
def test_model(clf, which_fusion):
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
        inputs = create_inputs(which_fusion, (empirical_batch_size, hidden_shape))

        estimate_config = find_best_parameters(
            clf, which_fusion, generate_parameter_configurations(num_dimensions), inputs
        )

        with FusionDefinition() as presched_fd:
            create_fusion_func(which_fusion, inputs)(presched_fd)

        _, est_time_ms = run_profile(which_fusion, presched_fd, inputs, estimate_config)
        est_perfs.append(est_time_ms)
        print(
            f"{which_fusion.name}, {empirical_batch_size}, {hidden_shape}, {estimate_config}, {est_time_ms:.3f}"
        )

    FusionCache.reset()
    nvf_perfs = []
    for hidden_shape in empirical_hidden_sizes:
        inputs = create_inputs(which_fusion, (empirical_batch_size, hidden_shape))

        with FusionDefinition() as presched_fd:
            create_fusion_func(which_fusion, inputs)(presched_fd)

        _, nvf_time_ms = run_profile(which_fusion, presched_fd, inputs)
        nvf_perfs.append(nvf_time_ms)
        print(
            f"{which_fusion.name}, {empirical_batch_size}, {hidden_shape}, {nvf_time_ms:.3f}"
        )

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
    plt.savefig(
        f"pointwise_{which_fusion.name}_empirical_batchsize{empirical_batch_size}.png"
    )
    plt.close("all")


# Collect data for decision tree
def collect_data():
    parameters = []
    performance = []

    for selected_fusion in FUSION:
        for shape in itertools.product(outer_shapes, inner_shapes):
            print(selected_fusion.name, shape)
            inputs = create_inputs(selected_fusion, shape)

            with FusionDefinition() as presched_fd:
                create_fusion_func(selected_fusion, inputs)(presched_fd)

            # unroll and vectorization configurations
            for config in generate_parameter_configurations(num_dimensions):
                perf_metric, _ = run_profile(
                    selected_fusion, presched_fd, inputs, config
                )
                features = construct_features(selected_fusion, config, inputs)
                parameters.append((*features, *flatten_config(config)))
                performance.append(perf_metric)
    return parameters, performance


# Separate collected data into training and test sets
def separate_data(parameters, performance):
    train_inputs = []
    test_inputs = []
    train_perf = []
    test_perf = []

    for data, perf in zip(parameters, performance):
        shape = data[:num_dimensions]
        config = data[num_dimensions:]

        if test_data_percentage > 0 and random.random() < test_data_percentage:
            test_inputs.append(data)
            test_perf.append(perf)
        else:
            train_inputs.append(data)
            train_perf.append(perf)
    return (train_inputs, train_perf), (test_inputs, test_perf)


# Run sequence of steps to collect data, train and test model
def main():
    # ============================ Run Experiments  ================================

    parameters, performance = collect_data()

    # ============================ Separate Data  ==================================

    train_data, _ = separate_data(parameters, performance)

    # ========================= Train Regression Models  ===========================

    # Apply decision tree regressor
    # Given input shapes and scheduler parameters, predict performance metric.
    from sklearn import ensemble

    train_inputs, train_perf = train_data
    clf = ensemble.RandomForestRegressor()
    clf = clf.fit(train_inputs, train_perf)

    # ========================= Test Regression Models  ===========================
    for f in FUSION:
        test_model(clf, f)

    # ==============================================================================


if __name__ == "__main__":
    main()

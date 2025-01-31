# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
import math
from nvfuser import FusionDefinition, SchedulerType, DataType
from dataclasses import dataclass
from enum import Enum

from autotune_utils import (
    ScriptConfiguration,
    collect_data,
    separate_data,
    test_model_rmse,
    test_model,
)


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


class AutotunePointwise:
    class FUSION(Enum):
        GELU_BIAS = 1
        SILU_MUL = 2
        BCAST_ADD = 3
        MUL = 4

    @dataclass(unsafe_hash=True)
    class PointwiseConfiguration:
        break_point: int
        bdim: [int]
        vectorize_factor: int
        outer_unroll: int
        inner_unroll: int

    def __init__(self, selected_fusion):
        self.selected_fusion = selected_fusion

    def __repr__(self):
        return f"pointwise_{self.selected_fusion.name}"

    # For pointwise scheduler, we test the cartesian product of vectorization and
    # unroll factors.
    def generate_scheduler_configurations(self, input_shape):
        def _named_product(**items):
            return itertools.starmap(
                self.PointwiseConfiguration, itertools.product(*items.values())
            )

        num_dimensions = len(input_shape)
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
                    # total_unroll_factor is between [1, 9] given that outer and
                    # inner unroll factors are between [1, 3].
                    outer_unroll_range = range(1, 4)
                    inner_unroll_range = range(1, 4)

                scheduler_config = _named_product(
                    break_point=[bp],
                    bdim=bdim_shapes,
                    vectorize_factor=[1, 2, 4, 8],
                    outer_unroll=outer_unroll_range,
                    inner_unroll=inner_unroll_range,
                )
                scheduler_configs.append(scheduler_config)
        return itertools.chain(*scheduler_configs)

    def create_inputs(self, shape, tensor_datatype):
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

        if self.selected_fusion == self.FUSION.GELU_BIAS:
            return outer_bcast()
        elif self.selected_fusion in [self.FUSION.SILU_MUL, self.FUSION.MUL]:
            return full()
        elif self.selected_fusion == FUSION.BCAST_ADD:
            return inner_bcast()
        else:
            assert False

    # A decorator to create a pointwise fusion given some input arguments.
    def create_fusion_func(self, inputs):
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

        if self.selected_fusion == self.FUSION.GELU_BIAS:
            return gelu_bias
        elif self.selected_fusion == self.FUSION.SILU_MUL:
            return silu_mul
        elif self.selected_fusion == self.FUSION.BCAST_ADD:
            return bcast_add
        elif self.selected_fusion == self.FUSION.MUL:
            return mul
        else:
            assert False

    # The pytorch eager mode reference used to validating nvfuser kernel.
    def eager_reference(self, inputs):
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

        if self.selected_fusion == self.FUSION.GELU_BIAS:
            return gelu_bias(inputs)
        elif self.selected__fusion == self.FUSION.SILU_MUL:
            return silu_mul(inputs)
        elif self.selected_fusion == self.FUSION.BCAST_ADD:
            return bcast_add(inputs)
        elif self.selected_fusion == self.FUSION.MUL:
            return mul(inputs)
        else:
            assert False

    # Apply scheduler with custom parameters using decorator
    def custom_scheduler(self, fd, scheduler_config):
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


# Run sequence of steps to collect data, train and test model
def main():
    # ====================== Setup Script Configuration  =======================
    script_config = ScriptConfiguration(
        num_dimensions=2,
        outer_shapes=[16384],
        inner_shapes=[128, 1024, 4096, 16384],
        tensor_datatype=torch.bfloat16,
        test_data_percentage=0.1,
        empirical_batch_size=16384,
        empirical_hidden_sizes=list(range(256, 32768, 256)),
    )

    autotune_config = AutotunePointwise(
        selected_fusion=AutotunePointwise.FUSION.GELU_BIAS
    )

    # ============================ Run Experiments  ============================

    parameters, performance = collect_data(script_config, autotune_config)

    # ============================ Separate Data  ==============================

    train_data, test_data = separate_data(script_config, parameters, performance)

    # ========================= Train Regression Models  =======================

    # Apply machine learning regressor
    # Given input shapes and scheduler parameters, predict performance metric.
    from sklearn import ensemble

    train_inputs, train_perf = train_data
    clf = ensemble.RandomForestRegressor()
    clf = clf.fit(train_inputs, train_perf)

    # ========================= Test Regression Models  ========================
    test_model_rmse(clf, script_config, autotune_config, test_data)
    test_model(clf, script_config, autotune_config)


if __name__ == "__main__":
    main()

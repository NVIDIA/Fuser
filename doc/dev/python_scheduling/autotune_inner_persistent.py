# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
from nvfuser import (
    FusionDefinition,
    SchedulerType,
    DataType,
    ParallelType,
    get_registers_per_thread,
)
from enum import Enum
from dataclasses import dataclass

from autotune_utils import (
    ScriptConfiguration,
    collect_data,
    separate_data,
    test_model_rmse,
    test_model,
    at_least_one_div,
    ceil_div,
    floor_div,
)


# ================================ Description ================================

# This script defines four inner persistent fusions:
#
# 1. LayerNorm
#    b = layer_norm(a, weight, bias)
#
# 2. Dropout + Add + LayerNorm
#    b = layer_norm(a + dropout(b), weight, bias)
#
# 3. RmsNorm
#    b = rms_norm(a, weight=w)
#
# 4. Dropout + Add + RmsNorm
#    return rms_norm(a + dropout(b), weight)
#
# 5. Softmax
#    return softmax(a, dim=-1)
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

# =============================================================================


class AutotuneInnerPersistent:
    class FUSION(Enum):
        LAYERNORM = 1
        DROPOUT_ADD_LAYERNORM = 2
        RMSNORM = 3
        DROPOUT_ADD_RMSNORM = 4
        SOFTMAX = 4

    @dataclass(unsafe_hash=True)
    class InnerPersistentConfiguration:
        # The x axis of CTA.
        bdimx: int = -1
        # The y axis of CTA.
        bdimy: int = -1

        project_persistent_buffers: bool = False
        padded_bdimx: int = -1
        is_pad_bdimx: bool = False
        persistent_batch_size: int = -1
        registers_per_thread: int = -1
        # The vectorization factor for inner reduction domain.
        vectorize_factor: int = 1
        # The grid size for the outer iteration domain.
        # If grdim > 1, then godim corresponds with y axis of the grid.
        # Otherwise, it is the x axis of the grid.
        godim: int = -1

    def __init__(self, selected_fusion):
        self.selected_fusion = selected_fusion

        # gpu device properties are defined globally
        assert torch.cuda.is_available()
        self.gpu_properties = torch.cuda.get_device_properties(device=0)

    def __repr__(self):
        return f"inner_persistent_{self.selected_fusion.name}"

    def convert_to_inner_persistent_params(self, scheduler_config, reduction_params):
        warp_size = 32
        max_number_of_threads_cta = 1024
        grid_x_limit = 2147483647

        reduction_params.persistent_kernel = True
        reduction_params.fastest_dim = True
        # TODO fix project_persistent_buffers
        reduction_params.project_persistent_buffers = (
            scheduler_config.project_persistent_buffers
        )
        reduction_params.cparams.index_type = DataType.Int32
        reduction_params.cparams.maxrregcount = scheduler_config.registers_per_thread

        reduction_params.cross_block_inner_reduction = True
        reduction_params.block_dim_inner_reduction = ParallelType.block_x
        reduction_params.pad_inner_reduction_to_warp = scheduler_config.is_pad_bdimx
        reduction_params.batches_per_block_inner_reduction = (
            scheduler_config.persistent_batch_size
        )
        reduction_params.unroll_factor_inner_reduction = (
            scheduler_config.vectorize_factor
        )
        reduction_params.vectorize_inner_reduction = (
            scheduler_config.vectorize_factor > 1
        )
        reduction_params.multiple_reds_per_blk = scheduler_config.bdimy > 1

        gdimx = -1
        if scheduler_config.godim > 1:
            reduction_params.grid_dim_iter_dom = ParallelType.grid_x
            if scheduler_config.godim > grid_x_limit:
                reduction_params.split_grid_dim_iter_dom_outer = True
                gdimx = grid_x_limit

        reduction_params.lparams.gdimx = gdimx

        # Reset CTA dimensions to avoid failing LaunchParams::assertValid
        reduction_params.lparams.bdimx = -1
        reduction_params.lparams.bdimy = -1
        reduction_params.lparams.bdimz = -1

        reduction_params.lparams.bdimx = scheduler_config.bdimx
        reduction_params.lparams.bdimy = scheduler_config.bdimy

    def get_max_persistent_batch(
        self, buffer_bytes_per_batch, target_threads_per_cta, register_overhead
    ):
        register_per_thread = get_registers_per_thread(target_threads_per_cta)
        register_for_buffer = register_per_thread - register_overhead

        bytes_per_register = 4
        batch_from_register = at_least_one_div(
            register_for_buffer * bytes_per_register, buffer_bytes_per_batch
        )

        max_batches_per_block = 16
        return min(max_batches_per_block, batch_from_register)

    # For persistent scheduler, we test the cartesian product of vectorization and
    # unroll factors.
    def generate_scheduler_configurations(self, input_shape):
        threads_per_cta_options = [128, 256, 512, 1024]
        vectorization_factor_options = [1, 2, 4, 8]

        dtype_size = 2
        warp_size = 32
        register_file_size = (256 * 1024) // 2
        register_overhead = 32
        num_iterations, num_reductions = input_shape

        # TODO fix softmax calculation
        # iteration domain of a tensor, weight, and bias
        # layer_norm and rms_norm = num_iterations + 2 * num_reductions
        # softmax = num_iterations
        max_persistent_buffer_size = num_iterations + 2 * num_reductions * dtype_size

        for (
            target_threads_per_cta,
            vectorize_factor,
        ) in itertools.product(
            threads_per_cta_options,
            vectorization_factor_options,
        ):
            num_reduction_after_vectorize = num_reductions // vectorize_factor
            batches_per_cta_min = ceil_div(
                num_reduction_after_vectorize, target_threads_per_cta
            )

            buffer_bytes_per_batch = (
                max_persistent_buffer_size / num_reductions / vectorize_factor
            )
            batches_per_cta_max = min(
                batches_per_cta_min,
                self.get_max_persistent_batch(
                    buffer_bytes_per_batch, target_threads_per_cta, register_overhead
                ),
            )

            for persistent_batch_size in range(
                batches_per_cta_min, batches_per_cta_max + 1
            ):
                max_multi_reduction_factor = min(
                    at_least_one_div(register_file_size, max_persistent_buffer_size),
                    ceil_div(num_iterations, self.gpu_properties.multi_processor_count),
                )

                # Build scheduler configuration
                scheduler_config = self.InnerPersistentConfiguration(
                    vectorize_factor=vectorize_factor,
                    persistent_batch_size=persistent_batch_size,
                )
                scheduler_config.bdimx = at_least_one_div(
                    num_reduction_after_vectorize, persistent_batch_size
                )
                scheduler_config.bdimy = min(
                    at_least_one_div(target_threads_per_cta, scheduler_config.bdimx),
                    max_multi_reduction_factor,
                )

                if scheduler_config.bdimx % warp_size == 0:
                    scheduler_config.padded_bdimx = scheduler_config.bdimx
                else:
                    scheduler_config.padded_bimx = scheduler_config.bdimx + (
                        warp_size - scheduler_config.bdimx % warp_size
                    )

                scheduler_config.is_pad_bdimx = (
                    scheduler_config.bdimx > 16
                    and scheduler_config.padded_bdimx * scheduler_config.bdimy
                    <= target_threads_per_cta
                )

                scheduler_config.godim = ceil_div(
                    num_iterations, scheduler_config.bdimy
                )
                yield scheduler_config

    def create_inputs(self, shape, tensor_datatype):
        a = torch.randn(*shape, dtype=tensor_datatype, device="cuda")
        b = torch.randn(*shape, dtype=tensor_datatype, device="cuda")
        weight = torch.randn(shape[-1], dtype=tensor_datatype, device="cuda")
        bias = torch.randn(shape[-1], dtype=tensor_datatype, device="cuda")

        if self.selected_fusion == self.FUSION.LAYERNORM:
            return [a, weight, bias]
        elif self.selected_fusion == self.FUSION.DROPOUT_ADD_LAYERNORM:
            return [a, b, weight, bias]
        elif self.selected_fusion == self.FUSION.RMSNORM:
            return [a, weight]
        elif self.selected_fusion == self.FUSION.DROPOUT_ADD_RMSNORM:
            return [a, b, weight]
        elif self.selected_fusion == self.FUSION.SOFTMAX:
            return [a]
        else:
            assert False

    # A decorator to create a persistent fusion given some input arguments.
    def create_fusion_func(self):
        def layernorm_fusion(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[-1],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T2 = fd.define_tensor(
                shape=[-1],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T3 = fd.ops.cast(T0, dtype=DataType.Float)
            T4, T5 = fd.ops.var_mean(T3, dims=[-1], correction=0, keepdim=False)
            T9 = fd.ops.broadcast(T4, is_broadcast_dim=[False, True])
            T13 = fd.ops.broadcast(T5, is_broadcast_dim=[False, True])
            S14 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
            T15 = fd.ops.add(T9, S14)
            T16 = fd.ops.rsqrt(T15)
            T21 = fd.ops.sub(T3, T13)
            T26 = fd.ops.mul(T21, T16)
            T31 = fd.ops.cast(T1, dtype=DataType.Float)
            T32 = fd.ops.mul(T26, T31)
            T37 = fd.ops.cast(T2, dtype=DataType.Float)
            T38 = fd.ops.add(T32, T37)
            T39 = fd.ops.cast(T38, dtype=DataType.BFloat16)
            fd.add_output(T39)

        def dropout_add_layernorm(fd: FusionDefinition) -> None:
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
            T2 = fd.define_tensor(
                shape=[-1],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T3 = fd.define_tensor(
                shape=[-1],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            S4 = fd.define_scalar(0.00000, dtype=DataType.Double)
            S5 = fd.define_scalar(1.00000, dtype=DataType.Double)
            S6 = fd.define_scalar(10, dtype=DataType.Int)
            S7 = fd.define_scalar(12, dtype=DataType.Int)
            T9 = fd.ops.uniform(S4, S5, shape=[S6, S7], dtype=DataType.BFloat16)
            S10 = fd.define_scalar(0.500000, dtype=DataType.Double)
            T11 = fd.ops.lt(T9, S10)
            T12 = fd.ops.cast(T0, dtype=DataType.Float)
            T13 = fd.ops.cast(T11, dtype=DataType.Float)
            T14 = fd.ops.mul(T12, T13)
            S15 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T16 = fd.ops.mul(T14, S15)
            T17 = fd.ops.cast(T1, dtype=DataType.Float)
            T18 = fd.ops.add(T17, T16)
            T19, T20 = fd.ops.var_mean(T18, dims=[1], correction=0, keepdim=False)
            T24 = fd.ops.broadcast(T19, is_broadcast_dim=[False, True])
            T28 = fd.ops.broadcast(T20, is_broadcast_dim=[False, True])
            S29 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
            T30 = fd.ops.add(T24, S29)
            T31 = fd.ops.rsqrt(T30)
            T36 = fd.ops.sub(T18, T28)
            T41 = fd.ops.mul(T36, T31)
            T45 = fd.ops.broadcast(T2, is_broadcast_dim=[True, False])
            T46 = fd.ops.cast(T45, dtype=DataType.Float)
            T47 = fd.ops.mul(T41, T46)
            T51 = fd.ops.broadcast(T3, is_broadcast_dim=[True, False])
            T52 = fd.ops.cast(T51, dtype=DataType.Float)
            T53 = fd.ops.add(T47, T52)
            T54 = fd.ops.cast(T53, dtype=DataType.BFloat16)
            fd.add_output(T54)

        def rmsnorm_fusion(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[-1],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T2 = fd.ops.cast(T0, dtype=DataType.Float)
            T3 = fd.ops.mul(T2, T2)
            T4 = fd.ops.sum(T3, dims=[-1], keepdim=False, dtype=DataType.Null)
            T8 = fd.ops.broadcast(T4, is_broadcast_dim=[False, True])
            S9 = fd.define_scalar(12.0000, dtype=DataType.Double)
            S10 = fd.ops.reciprocal(S9)
            T11 = fd.ops.mul(T8, S10)
            S12 = fd.define_scalar(0.00781250, dtype=DataType.Double)
            T13 = fd.ops.add(T11, S12)
            T14 = fd.ops.rsqrt(T13)
            T15 = fd.ops.cast(T14, dtype=DataType.BFloat16)
            T20 = fd.ops.cast(T15, dtype=DataType.Float)
            T21 = fd.ops.mul(T2, T20)
            T26 = fd.ops.cast(T21, dtype=DataType.Float)
            T27 = fd.ops.mul(T21, T26)
            T28 = fd.ops.cast(T27, dtype=DataType.BFloat16)
            fd.add_output(T28)

        def rmsnorm_fusion(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[-1],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T2 = fd.ops.cast(T0, dtype=DataType.Float)
            T3 = fd.ops.mul(T2, T2)
            T4 = fd.ops.sum(T3, dims=[1], keepdim=False, dtype=DataType.Null)
            T8 = fd.ops.broadcast(T4, is_broadcast_dim=[False, True])
            S9 = fd.define_scalar(T0.shape()[-0], dtype=DataType.Double)
            S10 = fd.ops.reciprocal(S9)
            T11 = fd.ops.mul(T8, S10)
            S12 = fd.define_scalar(0.00781250, dtype=DataType.Double)
            T13 = fd.ops.add(T11, S12)
            T14 = fd.ops.rsqrt(T13)
            T15 = fd.ops.cast(T14, dtype=DataType.BFloat16)
            T20 = fd.ops.cast(T15, dtype=DataType.Float)
            T21 = fd.ops.mul(T2, T20)
            T25 = fd.ops.broadcast(T1, is_broadcast_dim=[True, False])
            T26 = fd.ops.cast(T25, dtype=DataType.Float)
            T27 = fd.ops.mul(T21, T26)
            T28 = fd.ops.cast(T27, dtype=DataType.BFloat16)
            fd.add_output(T28)

        def dropout_add_rmsnorm(fd: FusionDefinition) -> None:
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
            T2 = fd.define_tensor(
                shape=[-1],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            S3 = fd.define_scalar(0.00000, dtype=DataType.Double)
            S4 = fd.define_scalar(1.00000, dtype=DataType.Double)
            T8 = fd.ops.uniform(S3, S4, shape=T0.shape(), dtype=DataType.BFloat16)
            S9 = fd.define_scalar(0.500000, dtype=DataType.Double)
            T10 = fd.ops.lt(T8, S9)
            T11 = fd.ops.cast(T0, dtype=DataType.Float)
            T12 = fd.ops.cast(T10, dtype=DataType.Float)
            T13 = fd.ops.mul(T11, T12)
            S14 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T15 = fd.ops.mul(T13, S14)
            T16 = fd.ops.cast(T1, dtype=DataType.Float)
            T17 = fd.ops.add(T16, T15)
            T18 = fd.ops.mul(T17, T17)
            T19 = fd.ops.sum(T18, dims=[1], keepdim=False, dtype=DataType.Null)
            T23 = fd.ops.broadcast(T19, is_broadcast_dim=[False, True])
            S24 = fd.define_scalar(T0.shape()[0], dtype=DataType.Double)
            S25 = fd.ops.reciprocal(S24)
            T26 = fd.ops.mul(T23, S25)
            S27 = fd.define_scalar(0.00781250, dtype=DataType.Double)
            T28 = fd.ops.add(T26, S27)
            T29 = fd.ops.rsqrt(T28)
            T30 = fd.ops.cast(T29, dtype=DataType.BFloat16)
            T35 = fd.ops.cast(T30, dtype=DataType.Float)
            T36 = fd.ops.mul(T17, T35)
            T40 = fd.ops.broadcast(T2, is_broadcast_dim=[True, False])
            T41 = fd.ops.cast(T40, dtype=DataType.Float)
            T42 = fd.ops.mul(T36, T41)
            T43 = fd.ops.cast(T42, dtype=DataType.BFloat16)
            fd.add_output(T43)

        def softmax_fusion(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.ops.cast(T0, dtype=DataType.Float)
            T2 = fd.ops.max(T1, dims=[-1], keepdim=False, dtype=DataType.Null)
            T6 = fd.ops.broadcast(T2, is_broadcast_dim=[False, True])
            T11 = fd.ops.sub(T1, T6)
            T12 = fd.ops.exp(T11)
            T13 = fd.ops.sum(T12, dims=[1], keepdim=False, dtype=DataType.Null)
            T17 = fd.ops.broadcast(T13, is_broadcast_dim=[False, True])
            T22 = fd.ops.reciprocal(T17)
            T23 = fd.ops.mul(T12, T22)
            T24 = fd.ops.cast(T23, dtype=DataType.BFloat16)
            fd.add_output(T24)

        if self.selected_fusion == self.FUSION.LAYERNORM:
            return layernorm_fusion
        elif self.selected_fusion == self.FUSION.DROPOUT_ADD_LAYERNORM:
            return dropout_add_layernorm
        elif self.selected_fusion == self.FUSION.RMSNORM:
            return rmsnorm
        elif self.selected_fusion == self.FUSION.DROPOUT_ADD_RMSNORM:
            return dropout_add_rmsnorm
        elif self.selected_fusion == self.FUSION.SOFTMAX:
            return softmax
        else:
            assert False

    # The pytorch eager mode reference used to validating nvfuser kernel.
    def eager_reference(self, inputs):
        def layernorm(inputs):
            a, weight, bias = inputs
            return torch.nn.functional.layer_norm(
                a, normalized_shape=[a.shape[-1]], weight=weight, bias=bias
            )

        def dropout_add_layernorm(inputs):
            a, b, weight, bias = inputs
            return torch.nn.functional.layer_norm(
                a + torch.nn.functional.dropout(b),
                normalized_shape=[a.shape[-1]],
                weight=weight,
                bias=bias,
            )

        def rmsnorm(inputs):
            a, weight = inputs
            return torch.nn.functional.rms_norm(
                a, normalized_shape=[a.shape[-1]], weight=weight
            )

        def dropout_add_rmsnorm(inputs):
            a, b, weight = inputs
            return torch.nn.functional.rms_norm(
                a + torch.nn.functional.dropout(b),
                normalized_shape=[a.shape[-1]],
                weight=weight,
            )

        def softmax(inputs):
            return torch.nn.functional.softmax(inputs[0], dim=-1)

        if self.selected_fusion == self.FUSION.LAYERNORM:
            return layernorm(inputs)
        elif self.selected_fusion == self.FUSION.DROPOUT_ADD_LAYERNORM:
            return dropout_add_layernorm(inputs)
        elif self.selected_fusion == self.FUSION.RMSNORM:
            return rmsnorm(inputs)
        elif self.selected_fusion == self.FUSION.DROPOUT_ADD_RMSNORM:
            return dropout_add_rmsnorm(inputs)
        elif self.selected_fusion == self.FUSION.SOFTMAX:
            return softmax(inputs)
        else:
            assert False

    # Apply scheduler with custom parameters using decorator
    def custom_scheduler(self, fd, scheduler_config):
        def inner_fn():
            # Check if compatible with persistent scheduler
            status, _ = fd.sched.can_schedule(SchedulerType.inner_persistent)
            assert status

            # persistent scheduler uses reduction parameters
            reduction_params = fd.sched.compute_inner_persistent_heuristics()

            # Modify original parameters
            if scheduler_config is not None:
                self.convert_to_inner_persistent_params(
                    scheduler_config, reduction_params
                )

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

    autotune_config = AutotuneInnerPersistent(
        selected_fusion=AutotuneInnerPersistent.FUSION.LAYERNORM
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

# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import itertools
from nvfuser import FusionDefinition, SchedulerType, DataType, ParallelType
from enum import Enum
from dataclasses import dataclass

from autotune_utils import (
    ScriptConfiguration,
    collect_data,
    separate_data,
    test_model_rmse,
    test_model,
    ceil_div,
    floor_div,
)


# ================================ Description ================================

# This script defines four inner reduction fusions:
#
# 1. Inner Sum
#    y = sum(x, dim=-1)
#
# 2. Add Sum
#    z = sum(x1 + x2 + x3 + x4, dim=-1)
#
# 3. Tanh Sum
#    y = sum(tanh(x), dim=-1)
#
# 4. Exp Sum
#    z = sum(exp(x), dim=-1)
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


class AutotuneInnerReduction:
    class FUSION(Enum):
        INNER_SUM = 1
        ADD_SUM = 2
        TANH_SUM = 3
        EXP_SUM = 4

    @dataclass(unsafe_hash=True)
    class InnerReductionConfiguration:
        # The vectorization factor for inner reduction domain.
        vectorize_factor: int = 1
        # The unroll factor for the inner reduction domain.
        reduction_unroll_factor: int = 1
        # The unroll factor for the outer iteration domain.
        iteration_unroll_factor: int = 1
        # The grid size for the outer iteration domain.
        # If grdim > 1, then godim corresponds with y axis of the grid.
        # Otherwise, it is the x axis of the grid.
        godim: int = -1
        # The grid size for the inner reduction domain. It corresponds
        # with x axis of the grid when it is >1.
        grdim: int = -1
        # The x axis of CTA. It corresponds with inner reduction domain.
        bdimx: int = -1
        # The y axis of CTA. It corresponds with outer reduction domain.
        # If it is non-zero, then there are multiple reduction per CTA.
        bdimy: int = -1

    def __init__(self, selected_fusion):
        self.selected_fusion = selected_fusion

        # gpu device properties are defined globally
        assert torch.cuda.is_available()
        self.gpu_properties = torch.cuda.get_device_properties(device=0)

    def __repr__(self):
        return f"inner_reduction_{self.selected_fusion.name}"

    def convert_to_inner_reduction_params(self, scheduler_config, reduction_params):
        warp_size = 32
        max_number_of_threads_cta = 1024
        grid_x_limit = 2147483647
        grid_y_limit = 65535

        reduction_params.schedule_3D = False
        reduction_params.fastest_dim = True
        reduction_params.cross_block_inner_reduction = True
        reduction_params.block_dim_inner_reduction = ParallelType.block_x
        reduction_params.cross_grid_inner_reduction = scheduler_config.grdim > 1
        reduction_params.multiple_reds_per_blk = scheduler_config.bdimy > 1
        reduction_params.pad_inner_reduction_to_warp = (
            scheduler_config.bdimx > warp_size
        ) and (
            (scheduler_config.bdimx * scheduler_config.bdimy)
            < max_number_of_threads_cta
        )
        reduction_params.unroll_factor_inner_reduction = (
            scheduler_config.vectorize_factor
        )
        reduction_params.vectorize_inner_reduction = (
            scheduler_config.vectorize_factor > 1
        )
        reduction_params.unroll_factor_top_of_vectorization = (
            scheduler_config.reduction_unroll_factor
        )

        if scheduler_config.bdimy > 1:
            reduction_params.block_dim_iter_dom = ParallelType.block_y

        reduction_params.unroll_factor_iter_dom = (
            scheduler_config.iteration_unroll_factor
        )

        gdimx = -1
        gdimy = -1

        if scheduler_config.grdim > 1:
            reduction_params.grid_dim_inner_reduction = ParallelType.grid_x
            reduction_params.grid_dim_iter_dom = ParallelType.grid_y

            reduction_params.split_grid_dim_iter_dom_inner = True
            gdimx = min(scheduler_config.grdim, grid_x_limit)
            gdimy = min(scheduler_config.godim, grid_y_limit)
            if scheduler_config.godim > grid_y_limit:
                reduction_params.split_grid_dim_iter_dom_outer = True
        else:
            reduction_params.grid_dim_iter_dom = ParallelType.grid_x
            gdimx = min(scheduler_config.godim, grid_x_limit)
            if scheduler_config.godim > grid_x_limit:
                reduction_params.split_grid_dim_inner_reduction = True

        reduction_params.lparams.gdimx = gdimx
        reduction_params.lparams.gdimy = gdimy

        # Reset CTA dimensions to avoid failing LaunchParams::assertValid
        reduction_params.lparams.bdimx = -1
        reduction_params.lparams.bdimy = -1
        reduction_params.lparams.bdimz = -1

        reduction_params.lparams.bdimx = scheduler_config.bdimx
        reduction_params.lparams.bdimy = scheduler_config.bdimy

    # For reduction scheduler, we test the cartesian product of vectorization and
    # unroll factors.
    def generate_scheduler_configurations(self, input_shape):
        threads_per_cta_options = [128, 256, 512, 1024]
        vectorization_factor_options = [1, 2, 4, 8]
        reduction_unroll_factor_options = list(range(1, 6))
        iteration_unroll_factor_options = list(range(1, 6))
        warp_size = 32

        num_iterations, num_reductions = input_shape

        for (
            threads_per_cta,
            vectorize_factor,
            reduction_unroll_factor,
            iteration_unroll_factor,
        ) in itertools.product(
            threads_per_cta_options,
            vectorization_factor_options,
            reduction_unroll_factor_options,
            iteration_unroll_factor_options,
        ):
            scheduler_config = self.InnerReductionConfiguration(
                vectorize_factor=vectorize_factor,
                reduction_unroll_factor=reduction_unroll_factor,
                iteration_unroll_factor=iteration_unroll_factor,
            )
            scheduler_config.bdimx = min(
                threads_per_cta,
                max(
                    warp_size,
                    ceil_div(num_reductions, scheduler_config.vectorize_factor),
                ),
            )
            scheduler_config.bdimy = min(
                threads_per_cta,
                max(1, floor_div(threads_per_cta, scheduler_config.bdimx)),
            )
            scheduler_config.godim = ceil_div(
                num_iterations, scheduler_config.bdimy * iteration_unroll_factor
            )

            # number of reduction elements not handled by a CTA
            remaining_reduction = ceil_div(
                ceil_div(
                    ceil_div(num_reductions, vectorize_factor), scheduler_config.bdimx
                ),
                reduction_unroll_factor,
            )

            if iteration_unroll_factor == 1 and remaining_reduction > 1:
                # all remaining reduction goes to grdim
                scheduler_config.grdim = remaining_reduction
                yield scheduler_config

                # When iteration dim is small, there may be unused SMs. We need
                # to shift work from block reduction to grid reduction to
                # increase SM usage.
                godim = scheduler_config.godim
                grdim = 1
                while (
                    godim * grdim * 2 <= self.gpu_properties.multi_processor_count
                    and (remaining_reduction / grdim) >= 2
                ):
                    grdim *= 2
                scheduler_config.grdim = grdim
                yield scheduler_config

            # grid stride across reduction iterDomain is 1
            scheduler_config.grdim = 1
            yield scheduler_config

    def create_inputs(self, shape, tensor_datatype):
        def inner_fn(num_inputs):
            return [
                torch.randn(*shape, dtype=tensor_datatype, device="cuda")
                for _ in range(num_inputs)
            ]

        if self.selected_fusion == self.FUSION.ADD_SUM:
            return inner_fn(num_inputs=4)
        elif self.selected_fusion in [
            self.FUSION.INNER_SUM,
            self.FUSION.TANH_SUM,
            self.FUSION.EXP_SUM,
        ]:
            return inner_fn(num_inputs=1)
        else:
            assert False

    # A decorator to create a reduction fusion given some input arguments.
    def create_fusion_func(self, inputs):
        def sum_fusion(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.ops.cast(T0, dtype=DataType.Float)
            T2 = fd.ops.sum(T1, dims=[1], keepdim=False, dtype=DataType.Null)
            T3 = fd.ops.cast(T2, dtype=DataType.BFloat16)
            fd.add_output(T3)

        def add_sum(fd: FusionDefinition) -> None:
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
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T3 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T4 = fd.ops.cast(T0, dtype=DataType.Float)
            T5 = fd.ops.cast(T1, dtype=DataType.Float)
            T6 = fd.ops.add(T4, T5)
            T7 = fd.ops.cast(T2, dtype=DataType.Float)
            T8 = fd.ops.add(T6, T7)
            T9 = fd.ops.cast(T3, dtype=DataType.Float)
            T10 = fd.ops.add(T8, T9)
            T11 = fd.ops.sum(T10, dims=[1], keepdim=False, dtype=DataType.Null)
            T12 = fd.ops.cast(T11, dtype=DataType.BFloat16)
            fd.add_output(T12)

        def tanh_sum(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.ops.cast(T0, dtype=DataType.Float)
            T2 = fd.ops.tanh(T1)
            T3 = fd.ops.sum(T2, dims=[1], keepdim=False, dtype=DataType.Null)
            T4 = fd.ops.cast(T3, dtype=DataType.BFloat16)
            fd.add_output(T4)

        def exp_sum(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.ops.cast(T0, dtype=DataType.Float)
            T2 = fd.ops.exp(T1)
            T3 = fd.ops.sum(T2, dims=[1], keepdim=False, dtype=DataType.Null)
            T4 = fd.ops.cast(T3, dtype=DataType.BFloat16)
            fd.add_output(T4)

        if self.selected_fusion == self.FUSION.INNER_SUM:
            return sum_fusion
        elif self.selected_fusion == self.FUSION.ADD_SUM:
            return add_sum
        elif self.selected_fusion == self.FUSION.TANH_SUM:
            return tanh_sum
        elif self.selected_fusion == self.FUSION.EXP_SUM:
            return exp_sum
        else:
            assert False

    # The pytorch eager mode reference used to validating nvfuser kernel.
    def eager_reference(self, inputs):
        def sum_fusion(inputs):
            return torch.sum(inputs[0], dim=-1)

        def add_sum(inputs):
            return torch.sum(inputs[0] + inputs[1] + inputs[2] + inputs[3], dim=-1)

        def tanh_sum(inputs):
            return torch.sum(torch.tanh(inputs[0]), dim=-1)

        def exp_sum(inputs):
            return torch.sum(torch.exp(inputs[0]), dim=-1)

        if self.selected_fusion == self.FUSION.INNER_SUM:
            return sum_fusion(inputs)
        elif self.selected_fusion == self.FUSION.ADD_SUM:
            return add_sum(inputs)
        elif self.selected_fusion == self.FUSION.TANH_SUM:
            return tanh_sum(inputs)
        elif self.selected_fusion == self.FUSION.EXP_SUM:
            return exp_sum(inputs)
        else:
            assert False

    # Apply scheduler with custom parameters using decorator
    def custom_scheduler(self, fd, scheduler_config):
        def inner_fn():
            # Check if compatible with reduction scheduler
            status, _ = fd.sched.can_schedule(SchedulerType.reduction)
            assert status

            reduction_params = fd.sched.compute_reduction_heuristics()

            # Modify original parameters
            if scheduler_config is not None:
                self.convert_to_inner_reduction_params(
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

    autotune_config = AutotuneInnerReduction(
        selected_fusion=AutotuneInnerReduction.FUSION.INNER_SUM
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

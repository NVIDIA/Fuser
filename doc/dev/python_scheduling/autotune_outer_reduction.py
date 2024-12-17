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
    round_up_pow2,
    round_up_multiple_of,
    round_down_pow2_or_multiple_of,
)


# ================================ Description ================================

# This script defines four outer reduction fusions:
#
# 1. Outer Sum
#    y = sum(x, dim=0)
#
# 2. Gelu-Bias Backward Function
#    y = gelu(x[i0, i1] + bias[broadcast, i1], approximate='tanh')
#    dx, dbias = grad(y)
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


class AutotuneOuterReduction:
    class FUSION(Enum):
        OUTER_SUM = 1
        GELU_BIAS_BWD = 2

    @dataclass(unsafe_hash=True)
    class OuterReductionConfiguration:
        # iteration parameters
        # iteration elements = iteration_unroll * bdimx * gidim
        # The unroll factor for the inner iteration domain.
        iteration_unroll_factor: int = 1
        # The x axis of CTA. It corresponds with inner iteration domain.
        bdimx: int = -1
        # The grid size for the inner iteration domain.
        gidim: int = -1

        # reduction parameters
        # reduction elements = reduction_unroll * bdimy * grdim * reduction_serial
        # The unroll factor for the outer reduction domain.
        reduction_unroll_factor: int = 1
        # The y axis of CTA. It corresponds with outer reduction domain.
        bdimy: int = -1
        # The grid size for the outer reduction domain.
        grdim: int = -1

    def __init__(self, selected_fusion):
        self.selected_fusion = selected_fusion

        # gpu device properties are defined globally
        assert torch.cuda.is_available()
        self.gpu_properties = torch.cuda.get_device_properties(device=0)

    def __repr__(self):
        return f"outer_reduction_{self.selected_fusion.name}"

    def convert_to_outer_reduction_params(self, scheduler_config, reduction_params):
        warp_size = 32
        max_number_of_threads_cta = 1024
        grid_x_limit = 2147483647
        grid_y_limit = 65535

        # Configure register count to get 50% occupancy with 1024 threads per SM.
        target_threads_per_sm = self.gpu_properties.max_threads_per_multi_processor // 2
        threads_per_cta = scheduler_config.bdimx * scheduler_config.bdimy
        threads_per_sm = (
            at_least_one_div(target_threads_per_sm, threads_per_cta) * threads_per_cta
        )
        registers_per_thread = get_registers_per_thread(threads_per_sm)
        reduction_params.cparams.maxrregcount = registers_per_thread

        reduction_params.cross_block_inner_reduction = (
            scheduler_config.bdimy > 1 or scheduler_config.grdim > 1
        )
        reduction_params.cross_grid_inner_reduction = scheduler_config.grdim > 1

        gdimx = -1
        gdimy = -1

        if scheduler_config.grdim > 1:
            reduction_params.split_grid_dim_inner_reduction = True
            reduction_params.grid_dim_inner_reduction = ParallelType.grid_y
            gdimy = min(scheduler_config.grdim, grid_y_limit)

        reduction_params.multiple_reds_per_blk = (
            scheduler_config.bdimx > 1 or scheduler_config.iteration_unroll_factor > 1
        )

        if reduction_params.multiple_reds_per_blk:
            reduction_params.block_dim_iter_dom = ParallelType.block_x

        reduction_params.grid_dim_iter_dom = ParallelType.grid_x
        if scheduler_config.gidim > grid_x_limit:
            reduction_params.split_grid_dim_iter_dom_outer = True
            gdimx = grid_x_limit

        reduction_params.flip_grid = False
        if reduction_params.cross_block_inner_reduction:
            if reduction_params.block_dim_iter_dom == ParallelType.block_x:
                reduction_params.block_dim_inner_reduction = ParallelType.block_y
            else:
                reduction_params.block_dim_inner_reduction = ParallelType.block_x

        reduction_params.unroll_factor_inner_reduction = (
            scheduler_config.reduction_unroll_factor
        )
        reduction_params.unroll_factor_iter_dom = (
            scheduler_config.iteration_unroll_factor
        )
        reduction_params.vectorize_iter_dom = (
            scheduler_config.iteration_unroll_factor > 1
        )

        reduction_params.lparams.gdimx = gdimx
        reduction_params.lparams.gdimy = gdimy

        # Reset CTA dimensions to avoid failing LaunchParams::assertValid
        reduction_params.lparams.bdimx = -1
        reduction_params.lparams.bdimy = -1
        reduction_params.lparams.bdimz = -1

        if reduction_params.multiple_reds_per_blk:
            reduction_params.lparams.bdimx = scheduler_config.bdimx
            reduction_params.lparams.bdimy = scheduler_config.bdimy
        else:
            reduction_params.lparams.bdimx = scheduler_config.bdimy

    # For reduction scheduler, we test the cartesian product of vectorization and
    # unroll factors.
    def generate_scheduler_configurations(self, input_shape):
        threads_per_cta_options = [128, 256, 512, 1024]
        vectorization_factor_options = [1, 2, 4, 8]
        reduction_unroll_factor_options = list(range(1, 6))
        warp_size = 32

        num_iterations, num_reductions = input_shape
        num_elements = num_iterations * num_reductions

        def get_block_outer_reduction_configurations(
            threads_per_cta, vectorize_factor, reduction_unroll_factor
        ):
            scheduler_config = self.OuterReductionConfiguration(
                reduction_unroll_factor=reduction_unroll_factor,
                iteration_unroll_factor=vectorize_factor,
            )

            bdimx = 8
            gidim = min(
                ceil_div(num_iterations, bdimx * vectorize_factor),
                self.gpu_properties.multi_processor_count,
            )
            bdimx = min(
                ceil_div(num_iterations, gidim * vectorize_factor),
                threads_per_cta,
            )
            bdimx = min(round_up_pow2(bdimx), round_up_multiple_of(bdimx, 32))
            gidim = min(
                ceil_div(num_iterations, bdimx * vectorize_factor),
                self.gpu_properties.multi_processor_count,
            )

            bdimy = min(num_reductions, threads_per_cta // bdimx)

            while (
                bdimy * bdimx * 2 <= threads_per_cta
            ) and gidim / 2 >= self.gpu_properties.multi_processor_count:
                bdimx *= 2
                gidim /= 2

            scheduler_config.bdimx = bdimx
            scheduler_config.bdimy = bdimy
            scheduler_config.gidim = gidim
            yield scheduler_config

        def get_grid_outer_reduction_configurations(
            threads_per_cta, vectorize_factor, reduction_unroll_factor
        ):
            scheduler_config = self.OuterReductionConfiguration(
                reduction_unroll_factor=reduction_unroll_factor,
                iteration_unroll_factor=vectorize_factor,
            )

            gidim = 1
            grdim = 1
            bdimx = 1
            bdimy = 1

            bdimx = min(
                ceil_div(
                    num_iterations,
                    vectorize_factor * self.gpu_properties.multi_processor_count,
                ),
                threads_per_cta,
            )
            bdimx = round_up_pow2(bdimx)

            bdimy = min(ceil_div(threads_per_cta, bdimx), num_reductions)
            bdimy = round_down_pow2_or_multiple_of(bdimy, 8)

            gidim = ceil_div(num_iterations, gidim * bdimx * vectorize_factor)
            num_reductions_available = ceil_div(
                num_reductions, grdim * bdimy * reduction_unroll_factor
            )
            grdim = min(
                num_reductions_available,
                ceil_div(self.gpu_properties.multi_processor_count, gidim),
            )

            # target_num_ctas = min(
            #    ceil_div(self.gpu_properties.multi_processor_count, 4),
            #    ceil_div(num_elements, threads_per_cta),
            # )
            # TODO if gidm * grdim < target_num_ctas, then increase grdim
            # TODO if (num_reductions_available > 16) and number of potential
            # ctas < 2 * sm_count, then increase grdim carefully

            scheduler_config.bdimx = bdimx
            scheduler_config.bdimy = bdimy
            scheduler_config.gidim = gidim
            scheduler_config.grdim = grdim
            yield scheduler_config

        for (
            threads_per_cta,
            vectorize_factor,
            reduction_unroll_factor,
        ) in itertools.product(
            threads_per_cta_options,
            vectorization_factor_options,
            reduction_unroll_factor_options,
        ):
            yield from get_block_outer_reduction_configurations(
                threads_per_cta, vectorize_factor, reduction_unroll_factor
            )
            # yield from get_grid_outer_reduction_configurations(
            #    threads_per_cta, vectorize_factor, reduction_unroll_factor
            # )

    def create_inputs(self, shape, tensor_datatype):
        if self.selected_fusion == self.FUSION.OUTER_SUM:
            return [torch.randn(*shape, dtype=tensor_datatype, device="cuda")]
        elif self.selected_fusion == self.FUSION.GELU_BIAS_BWD:
            b = torch.randn(
                [1, shape[-1]],
                dtype=tensor_datatype,
                device="cuda",
                requires_grad=False,
            )
            a = torch.randn(
                *shape, dtype=tensor_datatype, device="cuda", requires_grad=True
            )
            g = torch.randn(
                *shape, dtype=tensor_datatype, device="cuda", requires_grad=True
            )
            return [b, a, g]
        else:
            assert False

    # A decorator to create a reduction fusion given some input arguments.
    def create_fusion_func(self):
        def sum_fusion(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.ops.cast(T0, dtype=DataType.Float)
            T2 = fd.ops.sum(T1, dims=[0], keepdim=False, dtype=DataType.Null)
            T3 = fd.ops.cast(T2, dtype=DataType.BFloat16)
            fd.add_output(T3)

        def gelu_bias_bwd(fd: FusionDefinition) -> None:
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
            T2 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T7 = fd.ops.cast(T0, dtype=DataType.Float)
            T8 = fd.ops.cast(T1, dtype=DataType.Float)
            T9 = fd.ops.add(T8, T7)
            T10 = fd.ops.mul(T9, T9)
            T11 = fd.ops.mul(T10, T9)
            S12 = fd.define_scalar(0.0447150, dtype=DataType.Double)
            T13 = fd.ops.mul(S12, T11)
            T14 = fd.ops.add(T9, T13)
            S15 = fd.define_scalar(0.797885, dtype=DataType.Double)
            T16 = fd.ops.mul(S15, T14)
            T17 = fd.ops.tanh(T16)
            T18 = fd.ops.mul(T17, T17)
            T19 = fd.ops.cast(T2, dtype=DataType.Float)
            S20 = fd.define_scalar(0.500000, dtype=DataType.Double)
            T21 = fd.ops.mul(S20, T9)
            S22 = fd.define_scalar(1.00000, dtype=DataType.Double)
            T23 = fd.ops.sub(S22, T18)
            T24 = fd.ops.mul(T21, T19)
            T25 = fd.ops.mul(T24, T23)
            S26 = fd.define_scalar(1.00000, dtype=DataType.Double)
            T27 = fd.ops.add(S26, T17)
            S28 = fd.define_scalar(0.797885, dtype=DataType.Double)
            T29 = fd.ops.mul(S28, T25)
            T30 = fd.ops.mul(T27, T19)
            S31 = fd.define_scalar(0.0447150, dtype=DataType.Double)
            T32 = fd.ops.mul(S31, T29)
            S33 = fd.define_scalar(0.500000, dtype=DataType.Double)
            T34 = fd.ops.mul(S33, T30)
            T35 = fd.ops.mul(T9, T32)
            T36 = fd.ops.mul(T10, T32)
            T37 = fd.ops.add(T29, T34)
            T38 = fd.ops.mul(T9, T35)
            T39 = fd.ops.add(T37, T36)
            T40 = fd.ops.add(T39, T38)
            T41 = fd.ops.add(T40, T38)
            T42 = fd.ops.sum(T41, dims=[0], keepdim=False)
            T43 = fd.ops.cast(T42, dtype=DataType.BFloat16)
            fd.add_output(T43)

        if self.selected_fusion == self.FUSION.OUTER_SUM:
            return sum_fusion
        elif self.selected_fusion == self.FUSION.GELU_BIAS_BWD:
            return gelu_bias_bwd
        else:
            assert False

    # The pytorch eager mode reference used to validating nvfuser kernel.
    def eager_reference(self, inputs):
        def sum_fusion(inputs):
            return torch.sum(inputs[0], dim=0)

        def gelu_bias_bwd(inputs):
            b, a, g = inputs
            a.retain_grad()
            out = torch.nn.functional.gelu(a + b, approximate="tanh")
            out.retain_grad()
            out.backward(g.to(torch.double))
            return a.grad.to(torch.double).sum(dim=0)

        if self.selected_fusion == self.FUSION.OUTER_SUM:
            return sum_fusion(inputs)
        elif self.selected_fusion == self.FUSION.GELU_BIAS_BWD:
            return gelu_bias_bwd(inputs)
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
                self.convert_to_outer_reduction_params(
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

    autotune_config = AutotuneOuterReduction(
        selected_fusion=AutotuneOuterReduction.FUSION.OUTER_SUM
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

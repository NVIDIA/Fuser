# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from typing import Callable
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.jit_utils import RUN_CUDA

from nvfuser import FusionDefinition, DataType, ParallelType, MemoryType

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM


def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


@unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
@unittest.skipIf(is_pre_volta(), "Only supported on Volta and newer devices.")
class TestScheduleOps(TestCase):
    def sched_op_in_definition_error(self, sched_op_fn: Callable):
        """
        Common function to test for an error when a schedule op is used in a definition
        """
        inputs = [
            torch.randn(8, 8, 8, device="cuda"),
        ]

        def fusion_fn(fd: FusionDefinition):
            fd.t0 = fd.from_pytorch(inputs[0], static_sizes=True)
            fd.t1 = fd.ops.tanh(fd.t0)
            fd.add_output(fd.t1)

        class DefError(FusionDefinition):
            def definition(self):
                fusion_fn(self)
                sched_op_fn(self)

        with self.assertRaisesRegex(
            RuntimeError, "Attempting to use a SchedOperators Op prior to definition!"
        ):
            fd = DefError()
            _ = fd.execute(inputs)

    def check_input_error(
        self, sched_fn: Callable, error_msg: str, error_type=RuntimeError
    ):
        """
        Common function to test for an input error to a schedule op
        """
        inputs = [
            torch.randn(8, 8, 8, device="cuda"),
        ]

        def fusion_fn(fd: FusionDefinition):
            fd.t0 = fd.from_pytorch(inputs[0], static_sizes=True)
            fd.t1 = fd.ops.sum(fd.t0, dim=-1)
            fd.add_output(fd.t1)

        class InputError(FusionDefinition):
            def definition(self):
                fusion_fn(self)

            def schedule(self):
                sched_fn(self)

        with self.assertRaisesRegex(error_type, error_msg):
            fd = InputError()
            _ = fd.execute(inputs)

    def valid_use(self, sched_op_fn: Callable):
        """
        Common function to test op works in a common case
        """
        inputs = [
            torch.randn(8, 8, 8, device="cuda"),
        ]

        def fusion_fn(fd: FusionDefinition):
            fd.t0 = fd.from_pytorch(inputs[0], static_sizes=True)
            fd.t1 = fd.ops.sum(fd.t0, dim=-1)
            fd.add_output(fd.t1)

        class BasicValid(FusionDefinition):
            def definition(self):
                fusion_fn(self)

            def schedule(self):
                sched_op_fn(self)

        fd = BasicValid()
        # TODO: This can cause warnings from the FusionCache. It would be good
        # to capture them, instead.
        # Warning: You are overwriting the current user schedule for a definition!
        nvf_user_out = fd.execute(inputs)
        nvf_out = fd.execute(inputs, override_user_schedule=True)
        self.assertEqual(nvf_user_out, nvf_out)

    def test_merge_op(self):
        self.sched_op_in_definition_error(lambda fd: fd.sched.merge(fd.t1, 1))

        # Erorr check merge dimension. The dimension to merge is +1 from
        # the relative dimension indicated
        self.check_input_error(
            lambda fd: fd.sched.merge(fd.t1, 1),
            "Merging IterDomains requires that their iteration types match.",
        )
        self.check_input_error(
            lambda fd: fd.sched.merge(fd.t1, 2),
            "Tried to access out of boundary index 3. total index: 3",
        )
        # TODO: I am not sure why this error doesn't match the previous error
        # The previous error seems like it should match as they represent the
        # same merge position -1 and 2 in a 3 dimensional tensor
        # https://github.com/NVIDIA/Fuser/issues/171
        self.check_input_error(
            lambda fd: fd.sched.merge(fd.t1, -2),
            "Merging IterDomains requires that their iteration types match",
        )
        self.check_input_error(
            lambda fd: fd.sched.merge(fd.t1, -4),
            "Tried to access out of boundary index -1. total index: 3",
        )

        self.valid_use(lambda fd: fd.sched.merge(fd.t1, 0))
        self.valid_use(lambda fd: fd.sched.merge(fd.t1, -3))

    def test_reduction_factor_op(self):
        self.sched_op_in_definition_error(
            lambda fd: fd.sched.reduction_factor(fd.t1, [-1])
        )

        def error1_fn(fd: FusionDefinition):
            fd.sched.split(fd.t1, 2, 2)
            fd.sched.reduction_factor(fd.t1, [1])

        self.check_input_error(
            error1_fn, "Cannot rfactor axes that are not reduction axes."
        )

        def error2_fn(fd: FusionDefinition):
            fd.sched.split(fd.t1, 2, 2)
            fd.sched.reduction_factor(fd.t1, [-3])

        self.check_input_error(
            error2_fn, "Cannot rfactor axes that are not reduction axes."
        )

        def error3_fn(fd: FusionDefinition):
            fd.sched.split(fd.t1, 2, 2)
            fd.sched.reduction_factor(fd.t1, [2, 3])

        self.check_input_error(
            error3_fn, "Must have at least one reduction axis not marked as rfactor."
        )

        def sched_fn(fd: FusionDefinition):
            fd.sched.split(fd.t1, 2, 2)
            fd.sched.reduction_factor(fd.t1, [2])

        self.valid_use(sched_fn)

        # Donut whole factoring of reduction dims
        def sched1_fn(fd: FusionDefinition):
            fd.sched.split(fd.t1, 2, 4)
            fd.sched.split(fd.t1, 2, 2)
            fd.sched.reduction_factor(fd.t1, [2, 4])

        self.valid_use(sched1_fn)

        def sched2_fn(fd: FusionDefinition):
            fd.sched.split(fd.t1, 2, 4)
            fd.sched.split(fd.t1, 2, 2)
            fd.sched.reduction_factor(fd.t1, [3])

        self.valid_use(sched2_fn)

        # NOTE: The binding function for the "rfactor" alias is identical so
        # only proof of existence is needed
        def sched_fn_alias(fd: FusionDefinition):
            fd.sched.split(fd.t1, 2, 2)
            fd.sched.rfactor(fd.t1, [2])

        self.valid_use(sched_fn_alias)

    def test_reorder_op(self):
        self.sched_op_in_definition_error(
            lambda fd: fd.sched.reorder(fd.t1, {0: 1, 1: 0})
        )

        # Error checks of reorder dict
        self.check_input_error(
            lambda fd: fd.sched.reorder(fd.t1, {0: 3}),
            "Reorder axes are not within the number of dimensions of the provided domain",
        )
        self.check_input_error(
            lambda fd: fd.sched.reorder(fd.t1, {3: 0}),
            "Reorder axes are not within the number of dimensions of the provided domain",
        )
        self.check_input_error(
            lambda fd: fd.sched.reorder(fd.t1, {-4: 0}),
            'Found "old" position that\'s less than 0 even though already adjusted by nDims: -1',
        )
        self.check_input_error(
            lambda fd: fd.sched.reorder(fd.t1, {0: -4}),
            'Found "new" position that\'s less than 0 even though already adjusted by nDims: -1',
        )

        self.valid_use(lambda fd: fd.sched.reorder(fd.t1, {0: 1, 1: 0}))
        self.valid_use(lambda fd: fd.sched.reorder(fd.t1, {0: 1, 0: 1}))
        self.valid_use(lambda fd: fd.sched.reorder(fd.t1, {0: 0}))
        self.valid_use(lambda fd: fd.sched.reorder(fd.t1, {}))

    def test_split_op(self):
        self.sched_op_in_definition_error(lambda fd: fd.sched.split(fd.t1, 1, 2))

        # Error checking split dimension
        self.check_input_error(
            lambda fd: fd.sched.split(fd.t1, 3, 2),
            "Tried to access out of boundary index 3. total index: 3",
        )
        self.check_input_error(
            lambda fd: fd.sched.split(fd.t1, -4, 2),
            "Tried to access out of boundary index -1. total index: 3",
        )

        # Error checking split factor.
        # NOTE: ceildiv will always turn a split greater than the dimension
        # size into 1.
        self.check_input_error(
            lambda fd: fd.sched.split(fd.t1, 1, 0),
            "Invalid factor for split. Factor must be greater than 0. Factor = 0",
        )
        # NOTE: While a negative split is not allowed, it does not make sense
        # why the error is a TypeError given -1 is a valid int
        self.check_input_error(
            lambda fd: fd.sched.split(fd.t1, 1, -1),
            "Invalid factor for split. Factor must be greater than 0. Factor = -1",
        )

        self.valid_use(lambda fd: fd.sched.split(fd.t1, 1, 2))
        self.valid_use(lambda fd: fd.sched.split(fd.t1, -1, 2))

    def test_pointwise_basic_user_schedule(self):
        """
        Implement a simple pointwise kernel with user defined schedule
         * Uses merge, split, parallelize schedule operations
        """
        inputs = [
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        ]

        class Pointwise(FusionDefinition):
            def definition(self):
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.from_pytorch(inputs[1])
                self.t2 = self.ops.add(self.t0, self.t1)
                self.add_output(self.t2)

            def schedule(self):
                fd.sched.merge(self.t2, dim=0)
                fd.sched.split(self.t2, dim=0, factor=128)
                fd.sched.parallelize(self.t2, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(self.t2, axis := 1, ParallelType.block_x)

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = inputs[0] + inputs[1]
        self.assertEqual(eager_out, nvf_out[0])

    def test_pointwise_smem_cache_user_schedule(self):
        """
        Implement a simple pointwise kernel with user defined schedule
         * Uses the following schedule operations:
         * merge, split, parallelize
         * cache_after, cache_before, set_memory_type
        """
        inputs = [
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        ]

        class Pointwise(FusionDefinition):
            def definition(self):
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.from_pytorch(inputs[1])
                self.t2 = self.ops.add(self.t0, self.t1)
                self.add_output(self.t2)

            def schedule(self):
                cache_after_t0 = fd.sched.cache_after(self.t0)
                cache_after_t1 = fd.sched.cache_after(self.t1)
                cache_before_t2 = fd.sched.cache_before(self.t2)
                fd.sched.set_memory_type(cache_after_t0, MemoryType.shared)
                fd.sched.set_memory_type(cache_after_t1, MemoryType.shared)
                fd.sched.set_memory_type(cache_before_t2, MemoryType.shared)

                all_tvs = [cache_after_t0, cache_after_t1, cache_before_t2, self.t2]
                for tv in all_tvs:
                    fd.sched.merge(tv, dim=0)
                    fd.sched.split(tv, dim=0, factor=128)
                    fd.sched.parallelize(tv, axis := 0, ParallelType.grid_x)
                    fd.sched.parallelize(tv, axis := 1, ParallelType.block_x)

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = inputs[0] + inputs[1]
        self.assertEqual(eager_out, nvf_out[0])

    def test_pointwise_transform_user_schedule(self):
        """
        Implement a simple pointwise kernel with user defined schedule
         * Uses the following schedule operations:
         * merge, split, parallelize, cache_after, cache_before, set_memory_type
         * transform_like, parallelize_like
        """
        inputs = [
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        ]

        class Pointwise(FusionDefinition):
            def definition(self):
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.from_pytorch(inputs[1])
                self.t2 = self.ops.add(self.t0, self.t1)
                self.add_output(self.t2)

            def schedule(self):
                cache_after_t0 = fd.sched.cache_after(self.t0)
                cache_after_t1 = fd.sched.cache_after(self.t1)
                cache_before_t2 = fd.sched.cache_before(self.t2)
                fd.sched.set_memory_type(cache_after_t0, MemoryType.shared)
                fd.sched.set_memory_type(cache_after_t1, MemoryType.shared)
                fd.sched.set_memory_type(cache_before_t2, MemoryType.shared)

                fd.sched.merge(self.t2, dim=0)
                fd.sched.split(self.t2, dim=0, factor=128)
                fd.sched.transform_like(self.t2)

                fd.sched.parallelize(self.t2, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(self.t2, axis := 1, ParallelType.block_x)
                fd.sched.parallelize_like(self.t2)

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = inputs[0] + inputs[1]
        self.assertEqual(eager_out, nvf_out[0])

    def test_pointwise_inline_most_user_schedule(self):
        """
        Implement a simple pointwise kernel with user defined schedule
         * Uses the following schedule operations:
         * merge, split, parallelize, cache_after, cache_before, set_memory_type
         * transform_like, parallelize_like
         * inline_most
        """
        inputs = [
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        ]

        class Pointwise(FusionDefinition):
            def definition(self):
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.from_pytorch(inputs[1])
                self.t2 = self.ops.add(self.t0, self.t1)
                self.add_output(self.t2)

            def schedule(self):
                cache_after_t0 = fd.sched.cache_after(self.t0)
                cache_after_t1 = fd.sched.cache_after(self.t1)
                cache_before_t2 = fd.sched.cache_before(self.t2)
                fd.sched.set_memory_type(cache_after_t0, MemoryType.shared)
                fd.sched.set_memory_type(cache_after_t1, MemoryType.shared)
                fd.sched.set_memory_type(cache_before_t2, MemoryType.shared)

                fd.sched.merge(self.t2, dim=0)
                fd.sched.split(self.t2, dim=0, factor=128)
                fd.sched.transform_like(self.t2)

                fd.sched.parallelize(self.t2, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(self.t2, axis := 1, ParallelType.block_x)
                fd.sched.parallelize_like(self.t2)

                fd.sched.inline_most()

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = inputs[0] + inputs[1]
        self.assertEqual(eager_out, nvf_out[0])

    def test_pointwise_inline_at_user_schedule(self):
        """
        Implement a simple pointwise kernel with user defined schedule
         * Uses the following schedule operations:
         * merge, split, parallelize, cache_after, cache_before, set_memory_type
         * transform_like, parallelize_like
         * inline_like
        """
        inputs = [
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        ]

        class Pointwise(FusionDefinition):
            def definition(self):
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.from_pytorch(inputs[1])
                self.t2 = self.ops.add(self.t0, self.t1)
                self.add_output(self.t2)

            def schedule(self):
                cache_after_t0 = fd.sched.cache_after(self.t0)
                cache_after_t1 = fd.sched.cache_after(self.t1)
                cache_before_t2 = fd.sched.cache_before(self.t2)
                fd.sched.set_memory_type(cache_after_t0, MemoryType.shared)
                fd.sched.set_memory_type(cache_after_t1, MemoryType.shared)
                fd.sched.set_memory_type(cache_before_t2, MemoryType.shared)

                fd.sched.merge(self.t2, dim=0)
                fd.sched.split(self.t2, dim=0, factor=128)
                fd.sched.split(self.t2, dim=0, factor=4)
                fd.sched.transform_like(self.t2)

                fd.sched.parallelize(self.t2, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(self.t2, axis := -1, ParallelType.block_x)
                fd.sched.parallelize_like(self.t2)

                fd.sched.inline_at(self.t2, pos=1)

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = inputs[0] + inputs[1]
        self.assertEqual(eager_out, nvf_out[0])

    def test_var_mean_user_schedule(self):
        """
        Implement a simple normalization kernel with a user defined schedule
         * Uses the following schedule operations:
         * merge, split, parallelize, cache_after, cache_before, set_memory_type
         * transform_like, parallelize_like
         * inline_like
         * predicates: is_reduction, equality operator
        """
        tensor_size = 4
        inputs = [torch.randn(tensor_size, tensor_size, device="cuda")]

        class VarMean(FusionDefinition):
            def definition(self):
                self.t0 = fd.from_pytorch(inputs[0])
                self.s0 = fd.define_scalar(1e-6, dtype=DataType.Double)
                self.norm_const = fd.define_scalar(tensor_size, dtype=DataType.Int)

                self.sum0 = fd.ops.sum(self.t0, dims=[-1])
                # NOTE Manually broadcast because fusion definition cannot access hidden reduction tensor view.
                self.bcast_sum0 = fd.ops.broadcast(self.sum0, [False, True])
                self.mean = fd.ops.div(self.bcast_sum0, self.norm_const)

                self.diff = fd.ops.sub(self.t0, self.mean)
                self.diff_sq = fd.ops.mul(self.diff, self.diff)
                self.sum1 = fd.ops.sum(self.diff_sq, dims=[-1])
                # NOTE Manually broadcast because fusion definition cannot access hidden reduction tensor view.
                self.bcast_sum1 = fd.ops.broadcast(self.sum1, [False, True])
                self.var = fd.ops.div(self.bcast_sum1, self.norm_const)

                self.t0_diff = fd.ops.sub(self.t0, self.mean)
                self.var_eps = fd.ops.sqrt(fd.ops.add(self.var, self.s0))
                self.t0_norm = fd.ops.div(self.t0_diff, self.var_eps)
                self.add_output(self.t0_norm)

            def schedule(self):
                cache_after_t0 = fd.sched.cache_after(self.t0)
                fd.sched.set_memory_type(cache_after_t0, MemoryType.shared)

                cache_before_t0_norm = fd.sched.cache_before(self.t0_norm)
                cache_tvs = [cache_after_t0, cache_before_t0_norm]

                reference_tv = self.mean

                # Schedule Reference
                fd.sched.split(reference_tv, dim=-1, factor=256 * 4)
                fd.sched.split(reference_tv, dim=-1, factor=4)
                fd.sched.transform_like(reference_tv)

                # Add rfactor
                reduction_tvs = list(
                    filter(fd.sched.is_reduction, fd.sched.get_all_tensors())
                )
                assert len(reduction_tvs) == 2
                rfactor_tvs = [fd.sched.rfactor(tv, dims=[-1]) for tv in reduction_tvs]

                # Add common parallelization
                fd.sched.parallelize(reference_tv, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(reference_tv, axis := -2, ParallelType.block_x)
                fd.sched.parallelize_like(reference_tv)

                # Vectorize input load and output store
                fd.sched.parallelize(cache_after_t0, axis := -1, ParallelType.vectorize)
                fd.sched.parallelize(self.t0_norm, axis := -1, ParallelType.vectorize)

                # Add computeAt
                fd.sched.inline_most()

        fd = VarMean()
        nvf_out = fd.execute(inputs)
        var, mean = torch.var_mean(inputs[0], dim=-1, correction=0, keepdim=True)
        eager_out = (inputs[0] - mean) / torch.sqrt(var + 1e-6)
        self.assertEqual(eager_out, nvf_out[0])


if __name__ == "__main__":
    run_tests()

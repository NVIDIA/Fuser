# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from typing import Callable

import torch
from python.utils import is_pre_volta, is_pre_hopper
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.jit_utils import RUN_CUDA
import pytest

from nvfuser import (
    FusionCache,
    FusionDefinition,
    DataType,
    ParallelType,
    MemoryType,
    SchedulerType,
    LoadStoreOpType,
)

# NOTE We cannot iterate pybind11 enum directly, so we extract the entries here.
all_scheduler_heuristics = [
    heuristic
    for heuristic, _ in SchedulerType.__entries.values()
    if not SchedulerType.none
]


# A helper function to test heuristic schedulers with user schedules
def _apply_scheduler_helper(schedule, selected_heuristic):
    available_heuristics = schedule.find_compatible_schedulers()

    # Assume that only a single heuristic is available for fusion
    assert len(available_heuristics) == 1

    # Check that only selected heuristic is available as a scheduler
    assert set(available_heuristics) == set([selected_heuristic])

    # Double-check with can_schedule
    status, _ = schedule.can_schedule(selected_heuristic)
    assert status

    # Check that the other schedulers are not compatible with this fusion
    assert all(
        [
            not schedule.can_schedule(h)[0]
            for h in all_scheduler_heuristics
            if h is not selected_heuristic
        ]
    )

    # Apply selected scheduler
    schedule.schedule(selected_heuristic)


@pytest.mark.skipif(not RUN_CUDA, reason="requires CUDA")
@pytest.mark.skipif(is_pre_volta(), reason="Only supported on Volta and newer devices.")
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

        # reset cache to avoid follow up test picking up the manual user schedule
        FusionCache.get().reset()

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

        # reset cache to avoid follow up test picking up the manual user schedule
        FusionCache.get().reset()

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
        self.assertTrue(fd._exist_schedule(inputs))

    def test_print(self):
        """
        Test to_string and user_schedule_ir print functions
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
                fd.sched.merge(self.t0, dim=0)
                fd.sched.merge(self.t1, dim=0)
                fd.sched.merge(self.t2, dim=0)
                assert len(fd.sched.to_string(self.t0)) > 0
                assert len(fd.sched.to_string(self.t1)) > 0
                assert len(fd.sched.to_string(self.t2)) > 0

                user_ir = fd.sched.user_schedule_ir()
                assert len(user_ir) > 0
                assert user_ir != "User schedule is not defined."

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = inputs[0] + inputs[1]
        self.assertEqual(eager_out, nvf_out[0])

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

                all_tensors = [cache_after_t0, cache_after_t1, cache_before_t2, self.t2]
                for tensor in all_tensors:
                    fd.sched.merge(tensor, dim=0)
                    fd.sched.split(tensor, dim=0, factor=128)
                    fd.sched.parallelize(tensor, axis := 0, ParallelType.grid_x)
                    fd.sched.parallelize(tensor, axis := 1, ParallelType.block_x)

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = inputs[0] + inputs[1]
        self.assertEqual(eager_out, nvf_out[0])

    def test_pointwise_partial_transform_user_schedule(self):
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
                # NOTE Manual broadcast is required so reduction TensorView is
                # available in python frontend.
                self.t2 = self.ops.max(self.t0, dims=[-1], keepdim=False)
                self.t3 = self.ops.broadcast(self.t2, is_broadcast_dim=[False, True])
                self.t4 = self.ops.sub(self.t0, self.t3)
                self.t5 = self.ops.add(self.t4, self.t1)
                self.add_output(self.t5)

            def schedule(self):
                # Initial selected tensors is all original fusion TensorViews
                selected_tensors = [self.t2, self.t3, self.t4, self.t5]

                # Create cache tensors
                cache_after_t0 = fd.sched.cache_after(self.t0)
                cache_after_t1 = fd.sched.cache_after(self.t1)
                cache_before_t5 = fd.sched.cache_before(self.t5)

                # Place all intermediate tensors in shared memory because
                # we are not using computeAt
                fd.sched.set_memory_type(cache_after_t0, MemoryType.shared)
                fd.sched.set_memory_type(cache_after_t1, MemoryType.shared)
                fd.sched.set_memory_type(cache_before_t5, MemoryType.shared)
                fd.sched.set_memory_type(self.t4, MemoryType.shared)

                # Schedule all TensorViews except cache_after_t0 in the same way.
                selected_tensors.extend([cache_after_t1, cache_before_t5])
                fd.sched.split(self.t5, dim=1, factor=128)
                fd.sched.transform_like(self.t5, selected_tensors)

                # NOTE T2 was not transformed despite being a selected node,
                # so we manually schedule it.
                # TODO Improve error message to show warning if some selected
                # tensors are not transformed.
                fd.sched.split(self.t2, dim=1, factor=128)
                fd.sched.transform_like(self.t2, selected_tensors)

                # Create rfactor and add to selected tensors
                rfactor_t2 = fd.sched.reduction_factor(self.t2, dims=[1])
                selected_tensors.append(rfactor_t2)

                fd.sched.parallelize(self.t5, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(self.t5, axis := -1, ParallelType.block_x)
                fd.sched.parallelize_like(self.t5, pos := -1, selected_tensors)

                # NOTE Parallelize T2 and rfactor_t2 separately
                fd.sched.parallelize(self.t2, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(self.t2, axis := -1, ParallelType.block_x)
                fd.sched.parallelize_like(self.t2, pos := -1, selected_tensors)

                # Vectorize load t0 into shared memory
                fd.sched.split(cache_after_t0, dim=1, factor=4)
                fd.sched.parallelize(cache_after_t0, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(cache_after_t0, axis := 1, ParallelType.block_x)
                fd.sched.parallelize(cache_after_t0, axis := 2, ParallelType.vectorize)

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        max_input0_values, max_input0_indices = torch.max(
            inputs[0], dim=-1, keepdim=True
        )
        eager_out = (inputs[0] - max_input0_values) + inputs[1]
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
         * inline_at
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

    def test_pointwise_inline_selected_user_schedule(self):
        """
        Implement a simple pointwise kernel with user defined schedule
         * Uses the following schedule operations:
         * merge, split, parallelize, cache_after, cache_before, set_memory_type
         * transform_like, parallelize_like
         * inline_most, inline_at
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
                self.t3 = self.ops.exp(self.t2)
                self.add_output(self.t3)

            def schedule(self):
                cache_after_t0 = fd.sched.cache_after(self.t0)
                cache_after_t1 = fd.sched.cache_after(self.t1)
                cache_before_t3 = fd.sched.cache_before(self.t3)
                fd.sched.set_memory_type(cache_after_t0, MemoryType.shared)
                fd.sched.set_memory_type(cache_after_t1, MemoryType.shared)
                fd.sched.set_memory_type(cache_before_t3, MemoryType.shared)

                fd.sched.merge(self.t3, dim=0)
                fd.sched.split(self.t3, dim=0, factor=128)
                fd.sched.split(self.t3, dim=0, factor=4)
                fd.sched.transform_like(self.t3)

                fd.sched.parallelize(self.t3, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(self.t3, axis := -1, ParallelType.block_x)
                fd.sched.parallelize_like(self.t3)

                fd.sched.inline_at(
                    self.t2, pos=1, selected_tensors=[cache_after_t0, cache_after_t1]
                )
                fd.sched.inline_most(
                    selected_tensors=[cache_before_t3, self.t2, self.t3]
                )

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = torch.exp(inputs[0] + inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_var_mean_user_schedule(self):
        """
        Implement a simple normalization kernel with a user defined schedule
         * Uses the following schedule operations:
         * merge, split, parallelize
         * cache_after, cache_before, cache_fork, set_memory_type
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

                self.bcast_sum0 = fd.ops.sum(self.t0, dims=[-1], keepdim=True)
                self.mean = fd.ops.div(self.bcast_sum0, self.norm_const)
                self.add_output(self.mean)

                self.diff = fd.ops.sub(self.t0, self.mean)
                self.diff_sq = fd.ops.mul(self.diff, self.diff)
                self.bcast_sum1 = fd.ops.sum(self.diff_sq, dims=[-1], keepdim=True)
                self.var = fd.ops.div(self.bcast_sum1, self.norm_const)

                self.t0_diff = fd.ops.sub(self.t0, self.mean)
                self.invstd = fd.ops.rsqrt(fd.ops.add(self.var, self.s0))
                self.add_output(self.invstd)

                self.t0_norm = fd.ops.mul(self.t0_diff, self.invstd)
                self.add_output(self.t0_norm)

            def schedule(self):
                cache_after_t0 = fd.sched.cache_after(self.t0)
                fd.sched.set_memory_type(cache_after_t0, MemoryType.shared)

                cache_before_t0_norm = fd.sched.cache_before(self.t0_norm)
                cache_fork_mean = fd.sched.cache_fork(self.mean)
                cache_fork_invstd = fd.sched.cache_fork(self.invstd)
                cache_tensors = [
                    cache_after_t0,
                    cache_before_t0_norm,
                    cache_fork_mean,
                    cache_fork_invstd,
                ]

                reference_tensor = self.mean

                # Schedule Reference
                fd.sched.split(reference_tensor, dim=-1, factor=256 * 4)
                fd.sched.split(reference_tensor, dim=-1, factor=4)
                fd.sched.transform_like(reference_tensor)

                # Add rfactor
                reduction_tensors = list(
                    filter(fd.sched.is_reduction, fd.sched.tensors())
                )
                assert len(reduction_tensors) == 2
                rfactor_tensors = [
                    fd.sched.rfactor(tensor, dims=[-1]) for tensor in reduction_tensors
                ]

                # Add common parallelization
                fd.sched.parallelize(reference_tensor, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(reference_tensor, axis := -2, ParallelType.block_x)
                fd.sched.parallelize_like(reference_tensor)

                # Vectorize input load and output store
                fd.sched.parallelize(cache_after_t0, axis := -1, ParallelType.vectorize)
                fd.sched.parallelize(self.t0_norm, axis := -1, ParallelType.vectorize)

                # Add computeAt
                fd.sched.inline_most()

        fd = VarMean()
        nvf_mean, nvf_invstd, nvf_out = fd.execute(inputs)
        var, mean = torch.var_mean(inputs[0], dim=-1, correction=0, keepdim=True)
        invstd = torch.rsqrt(var + 1e-6)
        eager_out = (inputs[0] - mean) * invstd
        self.assertEqual(mean, nvf_mean)
        self.assertEqual(invstd, nvf_invstd)
        self.assertEqual(eager_out, nvf_out)

    def test_var_mean_tma_user_schedule(self):
        """
        Implement a simple normalization kernel using TMA ops with a user defined schedule
        """
        tensor_size = 4096
        use_tma_ops = not is_pre_hopper()
        inputs = [
            torch.randn(tensor_size, tensor_size, dtype=torch.bfloat16, device="cuda")
        ]

        class VarMean(FusionDefinition):
            def definition(self):
                self.t0 = fd.from_pytorch(inputs[0])
                self.s0 = fd.define_scalar(1e-6, dtype=DataType.Double)
                self.norm_const = fd.define_scalar(tensor_size, dtype=DataType.Int)

                self.mean_cast = fd.ops.cast(self.t0, dtype=DataType.Float)
                self.bcast_sum0 = fd.ops.sum(self.mean_cast, dims=[-1], keepdim=True)
                self.mean = fd.ops.div(self.bcast_sum0, self.norm_const)

                self.var_cast = fd.ops.cast(self.t0, dtype=DataType.Float)
                self.diff = fd.ops.sub(self.var_cast, self.mean)
                self.diff_sq = fd.ops.mul(self.diff, self.diff)
                self.bcast_sum1 = fd.ops.sum(self.diff_sq, dims=[-1], keepdim=True)
                self.var = fd.ops.div(self.bcast_sum1, self.norm_const)

                self.t0_cast = fd.ops.cast(self.t0, dtype=DataType.Float)
                self.t0_diff = fd.ops.sub(self.t0_cast, self.mean)
                self.var_eps = fd.ops.sqrt(fd.ops.add(self.var, self.s0))
                self.t0_norm = fd.ops.div(self.t0_diff, self.var_eps)

                self.t0_norm_cast = fd.ops.cast(self.t0_norm, dtype=DataType.BFloat16)
                self.add_output(self.t0_norm_cast)

            def schedule(self):
                smem_cache_op = (
                    LoadStoreOpType.tma if use_tma_ops else LoadStoreOpType.set
                )
                t0_smem = fd.sched.cache_after(self.t0, smem_cache_op)
                fd.sched.set_memory_type(t0_smem, MemoryType.shared)
                tma_tvs = [t0_smem]

                t0_lmem = fd.sched.cache_after(t0_smem)
                cache_before_t0_norm = fd.sched.cache_before(self.t0_norm_cast)

                def _is_not_tma_tensor(a):
                    return a not in tma_tvs

                all_tvs_except_tma = list(
                    filter(_is_not_tma_tensor, fd.sched.tensors())
                )

                tma_width = 256
                vectorize = 8
                elem_per_compute_thread = tensor_size // tma_width // vectorize

                # Define TMA Box
                fd.sched.split(t0_smem, dim=-1, factor=tma_width)

                reference_tv = self.t0_norm_cast

                # Schedule Reference
                # root domain: [I1, I2]
                # split: [I1, I2/V, V]
                fd.sched.split(reference_tv, dim=-1, factor=vectorize)
                # NOTE use outer-split to have constant register allocation
                # split: [I1, EPCT, I2/V/EPCT (block_x), V]
                fd.sched.split(
                    reference_tv,
                    dim=-2,
                    factor=elem_per_compute_thread,
                    inner_split=False,
                )
                # split: [I1, EPCT, I2/V/EPCT (block_x), U, V]
                fd.sched.split(reference_tv, dim=-2, factor=1)
                # split: [I1, I2/V/EPCT (block_x), EPCT, U, V]
                fd.sched.reorder(reference_tv, {-4: -3, -3: -4})

                # Transform all tensors
                fd.sched.transform_like(reference_tv, all_tvs_except_tma)

                # rfactor reduction tensors
                reduction_tvs = list(filter(fd.sched.is_reduction, fd.sched.tensors()))
                rfactor_tvs = [
                    fd.sched.rfactor(tv, dims=[-3, -2, -1]) for tv in reduction_tvs
                ]

                # Apply general parallelization
                fd.sched.parallelize(reference_tv, axis := 0, ParallelType.grid_x)
                fd.sched.parallelize(reference_tv, axis := 1, ParallelType.block_x)
                fd.sched.parallelize(reference_tv, axis := -2, ParallelType.unroll)
                fd.sched.parallelize_like(reference_tv)

                # vectorize store output
                fd.sched.parallelize(
                    self.t0_norm_cast, axis := -1, ParallelType.vectorize
                )

                # tma load input
                if use_tma_ops:
                    fd.sched.parallelize(t0_smem, axis := -1, ParallelType.tma)

                # computeAt
                fd.sched.inline_at(
                    reference_tv,
                    pos=-1,
                    best_effort=True,
                    selected_tensors=all_tvs_except_tma,
                )
                fd.sched.inline_at(
                    t0_lmem,
                    pos=-1,
                    best_effort=True,
                    selected_tensors=[t0_smem],
                )

        fd = VarMean()
        nvf_out = fd.execute(inputs)
        var, mean = torch.var_mean(inputs[0], dim=-1, correction=0, keepdim=True)
        eager_out = (inputs[0] - mean) / torch.sqrt(var + 1e-6)
        self.assertTrue(torch.allclose(eager_out, nvf_out[0], atol=1e-1))

    def test_pointwise_auto_scheduler(self):
        """
        Implement a simple pointwise kernel with user defined schedule
         * Uses nvfuser's PointwiseScheduler
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
                self.t3 = self.ops.exp(self.t2)
                self.add_output(self.t3)

            def schedule(self):
                # Apply selected scheduler
                _apply_scheduler_helper(fd.sched, SchedulerType.pointwise)

        fd = Pointwise()
        nvf_out = fd.execute(inputs)
        eager_out = torch.exp(inputs[0] + inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_reduction_auto_scheduler(self):
        """
        Implement a simple reduction kernel with user defined schedule
         * Expects failure with PointwiseScheduler
         * Uses nvfuser's ReductionScheduler
        """
        inputs = [
            torch.randn(4, 4, device="cuda"),
        ]

        class Reduction(FusionDefinition):
            def definition(self):
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.ops.sum(self.t0, dims=[1])
                self.t2 = self.ops.exp(self.t1)
                self.add_output(self.t2)

            def schedule(self):
                # Test error msg for can_schedule
                pointwise_status, error_msg = fd.sched.can_schedule(
                    SchedulerType.pointwise
                )
                assert not pointwise_status
                assert (
                    error_msg.strip()
                    == "Scheduler _pointwise_ ***rejected*** because : cannot find reference tensor"
                )

                # Apply selected scheduler
                _apply_scheduler_helper(fd.sched, SchedulerType.reduction)

        fd = Reduction()
        nvf_out = fd.execute(inputs)
        eager_out = torch.exp(inputs[0].sum(1))
        self.assertEqual(eager_out, nvf_out[0])

    def test_inner_persistent_auto_scheduler(self):
        """
        Implement a simple normalization kernel with a user defined schedule
         * Uses nvfuser's InnerPersistentScheduler
        """
        tensor_size = 4
        inputs = [torch.randn(tensor_size, tensor_size, device="cuda")]

        class VarMean(FusionDefinition):
            def definition(self):
                self.t0 = fd.from_pytorch(inputs[0])
                self.s0 = fd.define_scalar(1e-6, dtype=DataType.Double)
                self.norm_const = fd.define_scalar(tensor_size, dtype=DataType.Int)

                self.bcast_sum0 = fd.ops.sum(self.t0, dims=[-1], keepdim=True)
                self.mean = fd.ops.div(self.bcast_sum0, self.norm_const)

                self.diff = fd.ops.sub(self.t0, self.mean)
                self.diff_sq = fd.ops.mul(self.diff, self.diff)
                self.bcast_sum1 = fd.ops.sum(self.diff_sq, dims=[-1], keepdim=True)
                self.var = fd.ops.div(self.bcast_sum1, self.norm_const)

                self.t0_diff = fd.ops.sub(self.t0, self.mean)
                self.var_eps = fd.ops.sqrt(fd.ops.add(self.var, self.s0))
                self.t0_norm = fd.ops.div(self.t0_diff, self.var_eps)
                self.add_output(self.t0_norm)

            def schedule(self):
                # Apply selected scheduler
                _apply_scheduler_helper(fd.sched, SchedulerType.inner_persistent)

        fd = VarMean()
        nvf_out = fd.execute(inputs)
        var, mean = torch.var_mean(inputs[0], dim=-1, correction=0, keepdim=True)
        eager_out = (inputs[0] - mean) / torch.sqrt(var + 1e-6)
        self.assertEqual(eager_out, nvf_out[0])

    def test_batch_norm_auto_scheduler(self):
        batch_size = 16
        num_channels = 128
        height = 12
        width = 76
        momentum = 1e-1
        eps = 1e-5
        inputs = [
            torch.randn((batch_size, num_channels, height, width), device="cuda"),
            torch.randn((num_channels,), device="cuda"),
            torch.randn((num_channels,), device="cuda"),
            torch.randn((num_channels,), device="cuda"),
            torch.randn((num_channels,), device="cuda"),
            momentum,
            eps,
        ]

        class BatchNorm(FusionDefinition):
            def definition(self):
                a = fd.from_pytorch(inputs[0])
                w = fd.from_pytorch(inputs[1])
                b = fd.from_pytorch(inputs[2])
                running_mean = fd.from_pytorch(inputs[3])
                running_invstd = fd.from_pytorch(inputs[4])
                momentum = fd.define_scalar(dtype=DataType.Double)
                eps = fd.define_scalar(dtype=DataType.Double)
                a_norm, new_mean, new_invstd = fd.ops.batch_norm(
                    a,
                    w,
                    b,
                    running_mean,
                    running_invstd,
                    momentum,
                    eps,
                    training := True,
                    channels_last := False,
                )
                fd.add_output(a_norm)

            def schedule(self):
                # Apply selected scheduler
                _apply_scheduler_helper(fd.sched, SchedulerType.inner_persistent)

        fd = BatchNorm()
        nvf_out = fd.execute(inputs)
        torch_ref = torch.nn.functional.batch_norm(
            inputs[0],
            running_mean := inputs[3],
            running_var := inputs[4],
            weight := inputs[1],
            bias := inputs[2],
            training=True,
            momentum=momentum,
            eps=eps,
        )
        self.assertEqual(nvf_out[0], inputs[3])
        self.assertEqual(nvf_out[1], inputs[4])
        self.assertEqual(nvf_out[2], torch_ref)

    @pytest.mark.skip(
        reason="Disable test, the scheduler is not actually sending to ExprEvalExec but is sending to KernelExecutor which will correctly error."
    )
    def test_matmul_auto_scheduler(self):
        """
        Implement a simple matmul kernel with a user defined schedule
         * Uses nvfuser's ExprEvalScheduler
        """
        m = 24
        n = 16
        k = 8
        inputs_tt = [
            torch.randn(m, k, device="cuda", dtype=torch.float16),
            torch.randn(k, n, device="cuda", dtype=torch.float16),
        ]
        inputs_tn = [
            inputs_tt[0].clone(),
            inputs_tt[1].clone().as_strided(size=[k, n], stride=[1, k]),
        ]
        inputs_nt = [
            inputs_tt[0].clone().as_strided(size=[m, k], stride=[1, m]),
            inputs_tt[1].clone(),
        ]

        inputs_tn = [inputs_tt[0].clone(), inputs_tn[1].clone()]
        inputs_nn = [inputs_nt[0].clone(), inputs_tn[1].clone()]

        class Matmul(FusionDefinition):
            def __init__(self, inputs):
                super().__init__()
                self.inps = inputs

            def definition(self):
                t0 = fd.from_pytorch(self.inps[0])
                t1 = fd.from_pytorch(self.inps[1])
                t2 = fd.ops.matmul(t0, t1)
                fd.add_output(t2)

            def schedule(self):
                # Apply selected scheduler
                _apply_scheduler_helper(fd.sched, SchedulerType.expr_eval)

        for inputs in [inputs_tt, inputs_tn, inputs_nt, inputs_nn]:
            fd = Matmul(inputs)
            nvf_out = fd.execute(inputs)
            eager_out = torch.matmul(inputs[0], inputs[1])
            self.assertEqual(eager_out, nvf_out[0])

    def test_concretize_reshape_pointwise(self):
        input0_shape = [5, 10, 12]
        input1_shape = [2, 25, 3, 4]
        inputs = [
            torch.randn(input0_shape, device="cuda"),
            torch.randn(input1_shape, device="cuda"),
            *input1_shape,
        ]

        class Reshape(FusionDefinition):
            def definition(self):
                x = fd.from_pytorch(inputs[0])
                bias = fd.from_pytorch(inputs[1])
                S0 = fd.define_scalar(dtype=DataType.Int)
                S1 = fd.define_scalar(dtype=DataType.Int)
                S2 = fd.define_scalar(dtype=DataType.Int)
                S3 = fd.define_scalar(dtype=DataType.Int)
                bias_shape = fd.define_vector([S0, S1, S2, S3], dtype=DataType.Int)

                tv1 = fd.ops.abs(x)
                self.x_reshape = fd.ops.reshape(tv1, new_shape=bias_shape)
                y = fd.ops.add(self.x_reshape, bias)
                fd.add_output(y)

            def schedule(self):
                assert len(fd.sched.tensors()) == 5
                # check that we do not get Segmentation Fault when accessing a
                # tensor that was transformed from symbolic to concrete
                assert len(fd.sched.to_string(self.x_reshape)) > 0

                # Apply selected scheduler
                _apply_scheduler_helper(fd.sched, SchedulerType.pointwise)

        fd = Reshape()
        nvf_out = fd.execute(inputs)
        torch_ref = torch.abs(inputs[0]).reshape(inputs[1].shape) + inputs[1]
        self.assertEqual(nvf_out[0], torch_ref)

    # @pytest.mark.skipif(
    #     torch.cuda.device_count() < 2, reason="More than 1 GPU required"
    # )
    @pytest.mark.skip(
        reason="Disable test, not clear what nvFuser behavior should be with mixed devices in a fusion."
    )
    def test_inputs_with_different_devices(self):
        """
        Test case for issue 2056. Run the same fusion definition with inputs on
        different devices. The python frontend should create a new user
        schedule for inputs on different devices.
        """

        class FDScheduler(FusionDefinition):
            def definition(self):
                self.t0 = fd.define_tensor(
                    shape=[-1, -1, -1],
                    contiguity=[True, True, True],
                    dtype=DataType.Float,
                    is_cpu=False,
                )
                self.t1 = self.ops.sum(self.t0, dim=-1)
                self.add_output(self.t1)

            def schedule(self):
                # Apply reduction schedule
                _apply_scheduler_helper(fd.sched, SchedulerType.reduction)

        # Create Definition
        fd = FDScheduler()

        # Execute FusionDefinition with device 0 and 1
        devices = ["cuda:0", "cuda:1"]
        for device in devices:
            inputs = [
                torch.randn(8, 8, 8, dtype=torch.float32, device=device),
            ]
            torch_ref = inputs[0].sum(-1)
            nvf_out = fd.execute(inputs)
            self.assertEqual(nvf_out[0], torch_ref)

    def test_rfactor_twice(self):
        class Model(FusionDefinition):
            def definition(self):
                self.inp = fd.define_tensor([30])
                self.out = fd.ops.sum(self.inp, [0])
                self.add_output(self.out)

            def schedule(self):
                self.sched.split(self.out, 0, 2, False)
                self.sched.split(self.out, -1, 5, True)
                self.sched.rfactor(self.out, [-1])
                self.sched.rfactor(self.out, [0])

        fd = Model()
        inp = torch.randint(5, [30], dtype=torch.float32, device="cuda")
        (out,) = fd.execute([inp])
        self.assertEqual(out, inp.sum())

    def test_rfactor_allocation(self):
        class Model(FusionDefinition):
            def definition(self):
                self.inp = fd.define_tensor([4, 12], stride_order=[0, 1])
                self.out = fd.ops.sum(self.inp, [0])
                self.add_output(self.out)

            def schedule(self):
                print(fd._user_schedule_ir(True))
                self.sched.split(self.out, 0, 2, False)
                self.sched.rfactor(self.out, [0])
                print(fd._user_schedule_ir(True))
      
        fd = Model()
        print (fd.__repr__())
        inp = torch.ones((4, 12), dtype=torch.float, device="cuda")
        _ = fd.execute([inp])

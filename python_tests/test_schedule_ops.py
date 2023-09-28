# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from typing import Callable
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.jit_utils import RUN_CUDA

# Will only create the nvfuser module if CUDA is available
try:
    from nvfuser import (
        FusionDefinition,
    )
except ImportError:
    pass

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
            fd.t1 = fd.ops.sum(fd.t0, axis=-1)
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
            fd.t1 = fd.ops.sum(fd.t0, axis=-1)
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
            "Invalid merge detected, either one or both axes are outside of TensorView's range.",
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
            "Cannot merge axes within compute at position. Either axis -1 or 0 are within computePosition = 0",
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
            "Tried to access position . in domain",
        )
        self.check_input_error(
            lambda fd: fd.sched.split(fd.t1, -4, 2),
            "Split axis is less than 0 even after adjusting for nDims",
        )

        # Error checking split factor.
        # NOTE: ceildiv will always turn a split greater than the dimension
        # size into 1.
        self.check_input_error(
            lambda fd: fd.sched.split(fd.t1, 1, 0),
            "Expected rhs != 0 to be true, but got false",
        )
        # NOTE: While a negative split is not allowed, it does not make sense
        # why the error is a TypeError given -1 is a valid int
        self.check_input_error(
            lambda fd: fd.sched.split(fd.t1, 1, -1),
            "incompatible function arguments",
            TypeError,
        )

        self.valid_use(lambda fd: fd.sched.split(fd.t1, 1, 2))
        self.valid_use(lambda fd: fd.sched.split(fd.t1, -1, 2))


if __name__ == "__main__":
    run_tests()

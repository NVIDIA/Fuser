# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import os
from copy import deepcopy
from typing import Callable, Optional
import tempfile
import torch
import pytest
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase
from looseversion import LooseVersion

# flake8 complains about DataType being unused in this file but it is necessary
# to run captured fusion definition.
# flake8: noqa
from nvfuser import FusionCache, FusionDefinition, DataType, clone, Tensor


def is_pre_volta():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def is_pre_ampere():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 8


def is_pre_hopper():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 9


def is_pre_blackwell():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 10


def verify_stride_order(output_strides, stride_order):
    sorted_stride = list(output_strides)
    rank = len(output_strides)
    for idx, axis in enumerate(stride_order):
        sorted_stride[rank - 1 - axis] = output_strides[idx]
    assert sorted(sorted_stride, reverse=True) == sorted_stride


# Get string representation for FusionDefinition
# Run captured python definition
# Check that the result of captured python definition matches original results
def check_captured_python_definition(reference_outputs, fd, inputs, device=None):
    import re

    try:
        fd_str = fd.__repr__()
        func_name = re.findall("(nvfuser_fusion_id\\d+)", fd_str.split("\n")[1])[0]
        exec(fd_str)

        # Execute the python definition that was captured
        with FusionDefinition() as fd_cap:
            eval(func_name)(fd_cap)

        torch.manual_seed(0)
        captured_outputs = fd_cap.execute(inputs, device=device)
        # Make sure the original and captured definitions match
        # torch.allclose does not work with fp8 datatype, so cast to fp64.
        return all(
            [
                torch.allclose(
                    ref_out.to(torch.float64),
                    captured_outputs[idx].to(torch.float64),
                    equal_nan=True,
                )
                for idx, ref_out in enumerate(reference_outputs)
            ]
        )
    except Exception as err:
        print("\nException For Printed FusionDefinition:")
        print(
            "(A failure here suggests a mismatch in functionality between the original definition and the printed definition.)"
        )
        print(fd_str)
        raise err


# Run original FusionDefinition
# Clone FusionDefinition
# Apply segmentation if it supported for this FusionDefinition
# Run cloned python definition
# Check that the result of cloned python definition matches original results
def check_cpp_translation(
    reference_outputs, fd, inputs, supports_segmentation, device=None
):
    try:
        torch.manual_seed(0)

        # Clone
        cloned_fd = FusionDefinition()
        clone(fd, cloned_fd)

        # Segment
        if supports_segmentation:
            cloned_fd.segment(inputs)

        # Run
        cloned_outputs = cloned_fd.execute(inputs, device=device)

        # Make sure the results of original and cloned definitions match.
        # torch.allclose does not work with fp8 datatype, so cast to fp64.
        return all(
            [
                torch.allclose(
                    ref_out.to(torch.float64),
                    cloned_outputs[idx].to(torch.float64),
                    equal_nan=True,
                )
                for idx, ref_out in enumerate(reference_outputs)
            ]
        )
    except Exception as err:
        print("\nException For CPP Translation:")
        print(
            "(A failure here suggests a mismatch in functionality between the original and cloned definitions.)"
        )
        print("Does FusionDefinition supports segmentation?\t", supports_segmentation)
        print(fd._repro_error_str("executing", inputs))
        raise err


# This DEBUG_SERDE environment flag is used to debug serialization failures.
#
# If DEBUG_SERDE=debug
# 1) It disables automatically saving FusionCache upon program exit. Therefore,
# it has to be a global flag not per-test.
#
# 2) It resets the FusionCache after each test, which is useful for isolating
# failures. Note, some failures only occur when running multiple tests
# together and accumulating fusions in the cache.
#
# 3) It keeps the temporary files that are created during serde_check.
# Normally, these files are deleted after each test.
#
# DEBUG_SERDE=disable
# 1) It disables the @nvfusertest_serde_check decorator. This disables checking
# that serde round-trips preserve the definition during testing.
env_var_debug_serde = os.getenv("DEBUG_SERDE", "").lower()
debug_serde: bool = env_var_debug_serde == "debug"
disable_serde: bool = env_var_debug_serde == "disable"
del env_var_debug_serde


# The pytest framework and test_python_frontend.py use different arguments for
# testing, so we need specific `serde_check` decorators for both frameworks.
# basic_serde_check is the common part between them. It serializes the cache,
# deletes it, and then deserialized to recreate the cache.
def basic_serde_check():
    # If DEBUG_SERDE is enabled, the temporary file is not deleted
    # automatically
    with tempfile.NamedTemporaryFile(delete=(not debug_serde)) as tmp:
        try:
            # Serialize FusionCache
            fc = FusionCache.get()
            fc.serialize(tmp.name)

            FusionCache.reset()

            # Get new FusionCache because the previous one was destroyed by
            # the reset call.
            fc = FusionCache.get()
            assert fc.num_fusions() == 0
            fc.deserialize(tmp.name)
        except Exception as e:
            if debug_serde:
                raise RuntimeError(
                    f"***** {tmp.name} contains the serialized binary for this failure."
                )
            else:
                raise RuntimeError(
                    "***** Use DEBUG_SERDE=debug to debug serialization failure."
                )


# Enable automatic serialization upon program exit and test deserializing the
# default workspace. NOTE: Serializing error test cases corrupts the serialized
# binary. Call FusionCache.reset() to clear the cache after running an error
# test in `test_python_frontend.py'.
def atexit_serde_check():
    if disable_serde:
        # Ignore FusionCache and automatic serialization if serde check is
        # disabled
        return

    from nvfuser import FusionCache

    if not debug_serde:
        from nvfuser import enable_automatic_serialization

        # Turn on default serialization upon program exit
        enable_automatic_serialization()

    # Automatically load common workplace
    fc = FusionCache.get()
    # Clear FusionCache because the tests expect a new fusion to be generated.
    FusionCache.reset()


def nvfusertest_serde_check(test_fn: Callable):
    """
    A decorator to verify that serialization works with the given exec_nvfuser
    function. Currently, it uses serialization to rebuild the FusionCache
    structure.
    """
    if disable_serde:

        def inner_fn(*args, **kwargs):
            # Remove skip_serde_check if it was given
            kwargs.pop("skip_serde_check", None)
            return test_fn(*args, **kwargs)

        return inner_fn

    def inner_fn(*args, **kwargs):
        self, fusion_func, inputs = args

        # NOTE: For debug purposes, clear FusionCache before running first test
        # so the behavior is more deterministic (PR #1848).
        is_new_fusion_expected = kwargs.get("new_fusion_expected", True)
        if debug_serde and is_new_fusion_expected:
            FusionCache.reset()
            assert FusionCache.get().num_fusions() == 0

        # skip_serde_check is only used by the decorator so remove it before
        # running test_fn
        skip_serde_check = kwargs.pop("skip_serde_check", False)
        if skip_serde_check:
            return test_fn(self, fusion_func, inputs, **kwargs)

        # Run test to populate FusionCache. Deep copy inputs for this run but
        # not the final run. When a fusion output aliases an input, it will
        # change the input value for subsequent function calls. Therefore, only
        # the final run should take the original tensors and potentially update
        # their values.
        inputs_copy = deepcopy(inputs)
        test_fn(self, fusion_func, inputs_copy, **kwargs)

        # Serialize and Deserialize FusionCache
        basic_serde_check()

        # Run test with repopulated FusionCache
        kwargs["new_fusion_expected"] = False
        return test_fn(self, fusion_func, inputs, **kwargs)

    return inner_fn


UPDATED_SDPA = LooseVersion(torch.__version__) >= LooseVersion("2.7.0")


def define_sdpa_rng_state(fd: FusionDefinition) -> tuple[Tensor, Tensor]:
    dtype = DataType.UInt64 if UPDATED_SDPA else DataType.Int
    is_cpu = False if UPDATED_SDPA else True
    philox_shape = [2] if UPDATED_SDPA else []
    philox_seed = fd.define_tensor(
        shape=philox_shape,
        dtype=dtype,
        is_cpu=is_cpu,
    )
    philox_offset = fd.define_tensor(
        shape=[],
        dtype=dtype,
        is_cpu=is_cpu,
    )
    return philox_seed, philox_offset


def create_sdpa_rng_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.uint64 if UPDATED_SDPA else torch.int64
    device = "cuda" if UPDATED_SDPA else "cpu"
    philox_shape = (2,) if UPDATED_SDPA else ()
    philox_seed = torch.testing.make_tensor(philox_shape, device=device, dtype=dtype)
    philox_offset = torch.testing.make_tensor((), device=device, dtype=dtype)
    return philox_seed, philox_offset


"""
Base class for any test class that needs to verify serialization
and run captured string representations of FusionDefinition.
"""


class NVFuserTest(TestCase):
    @classmethod
    def setup_class(cls):
        """
        Setup is run once at the class level, before running any tests of the class.
        `atexit_serde_check` enables automatic serialization at the end of the test suite.
        """
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

        atexit_serde_check()

    # Helper function to verify the nvfuser output and make sure the string
    # definition based on the FusionDefinition is executable and matches the
    # original definition
    @nvfusertest_serde_check
    def exec_nvfuser(
        self,
        fusion_func,
        inputs,
        *,
        _enable_options=[],
        _disable_options=[],
        new_fusion_expected=True,
        device=None,
        is_clonable=True,
        supports_segmentation=True,
    ):
        fc = FusionCache.get()
        before_fusions = fc.num_fusions()
        # Copy inputs because aliased outputs can modify inputs when running
        # FusionDefinition
        inputs_captured = deepcopy(inputs)
        if is_clonable:
            inputs_cloned = deepcopy(inputs)

        # Execute a fusion function and capture the string python definition
        with FusionDefinition() as fd:
            fusion_func(fd)
        torch.manual_seed(0)
        if "id_model_extra_validation" not in _enable_options:
            _enable_options.append("id_model_extra_validation")
        out = fd.execute(
            inputs,
            device=device,
            _enable_options=_enable_options,
            _disable_options=_disable_options,
        )

        self.assertTrue(
            check_captured_python_definition(out, fd, inputs_captured, device)
        )
        if not disable_serde:
            self.assertEqual(
                fc.num_fusions() - before_fusions, int(new_fusion_expected)
            )

        if is_clonable:
            self.assertTrue(
                check_cpp_translation(out, fd, inputs_cloned, supports_segmentation)
            )
        return out, fd

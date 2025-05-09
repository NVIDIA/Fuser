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
from functools import wraps
from enum import Enum, auto
from torch.testing._internal.common_utils import TestCase
from looseversion import LooseVersion

# flake8 complains about DataType being unused in this file but it is necessary
# to run captured fusion definition.
# flake8: noqa
from nvfuser import FusionCache, FusionDefinition, DataType, clone, Tensor

try:
    # flake8: noqa
    import jax

    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    pass


def requiresJAX(fn):
    @wraps(fn)
    def _fn(*args, **kwargs):
        if not JAX_AVAILABLE:
            pytest.xfail("Requires JAX")
        return fn(*args, **kwargs)

    return _fn


class ArgumentType(Enum):
    # a symbolic value requires an input argument during kernel execution
    Symbolic = auto()
    # scalar with constant value
    ConstantScalar = auto()
    # python number - int, float, complex, bool
    Constant = auto()


bool_dtypes = (torch.bool,)

int_dtypes = (
    torch.int32,
    torch.int64,
)

half_precision_float_dtypes = (
    torch.bfloat16,
    torch.float16,
)

full_precision_float_dtypes = (
    torch.float32,
    torch.float64,
)

complex_dtypes = (
    torch.complex64,
    torch.complex128,
)

# Half-precision float dtypes bf16, fp16 are skipped because nvfuser upcasts those dtypes to fp32
# but does not return the original type.
bool_int_dtypes = bool_dtypes + int_dtypes
float_dtypes = half_precision_float_dtypes + full_precision_float_dtypes
int_float_dtypes = int_dtypes + full_precision_float_dtypes
float_complex_dtypes = full_precision_float_dtypes + complex_dtypes
all_dtypes_except_reduced = int_dtypes + full_precision_float_dtypes + complex_dtypes
all_dtypes_except_bool = all_dtypes_except_reduced + half_precision_float_dtypes
all_dtypes = all_dtypes_except_bool + bool_dtypes

map_dtype_to_str = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}

torch_to_jax_dtype_map = None
if JAX_AVAILABLE:
    import jax.numpy as jnp

    torch_to_jax_dtype_map = {
        torch.bool: jnp.bool_,
        torch.uint8: jnp.uint8,
        torch.int8: jnp.int8,
        torch.int16: jnp.int16,
        torch.int32: jnp.int32,
        torch.int64: jnp.int64,
        torch.bfloat16: jnp.bfloat16,
        torch.float16: jnp.float16,
        torch.float32: jnp.float32,
        torch.float64: jnp.float64,
        torch.complex64: jnp.complex64,
        torch.complex128: jnp.complex128,
    }

torch_to_python_dtype_map = {
    torch.bool: bool,
    torch.uint8: int,
    torch.int8: int,
    torch.int16: int,
    torch.int32: int,
    torch.int64: int,
    torch.bfloat16: float,
    torch.float16: float,
    torch.float32: float,
    torch.float64: float,
    torch.complex64: complex,
    torch.complex128: complex,
}


def make_tensor_like(a):
    # type: (torch.Tensor) -> torch.Tensor
    """Returns a tensor with the same properties as the given tensor.

    Args:
        a (torch.Tensor): The tensor to copy properties from.

    Returns:
        torch.Tensor: A tensor with the same properties as :attr:`a`.
    """
    return torch.testing.make_tensor(
        a.shape, device=a.device, dtype=a.dtype, requires_grad=a.requires_grad
    )


def make_number(
    dtype: torch.dtype, low: Optional[float] = None, high: Optional[float] = None
):
    """Returns a random number with desired dtype

    Args:
        dtype (torch.dtype): Desired dtype for number.
        low (Optional[Number]): Sets the lower limit (inclusive) of the given range.
        high (Optional[Number]): Sets the upper limit (exclusive) of the given range.

    Returns:
        (Scalar): The scalar number with specified dtype.
    """
    return make_tensor([1], device="cpu", dtype=dtype, low=low, high=high).item()


def find_nonmatching_dtype(dtype: torch.dtype):
    if dtype in int_float_dtypes:
        return torch.complex128
    elif dtype in complex_dtypes:
        return torch.double
    elif dtype is torch.bool:
        return torch.float32
    return None


def is_complex_dtype(dtype: torch.dtype):
    return dtype in complex_dtypes


def is_floating_dtype(dtype: torch.dtype):
    return dtype in float_dtypes


def is_integer_dtype(dtype: torch.dtype):
    return dtype in int_dtypes


def is_tensor(a):
    return isinstance(a, torch.Tensor)


def is_pre_volta():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def is_pre_ampere():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 8


def is_pre_hopper():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 9


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

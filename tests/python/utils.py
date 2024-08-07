# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import os
from copy import deepcopy
from typing import Callable
import tempfile
import torch

# flake8 complains about DataType being unused in this file but it is necessary
# to run captured fusion definition.
# flake8: noqa
from nvfuser import FusionCache, FusionDefinition, DataType


def is_pre_volta():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def is_pre_ampere():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 8


def is_pre_hopper():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 9


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


# This DEBUG_SERDE environment flag is used to debug serialization failures.
#
# 1) It disables automatically saving FusionCache upon program exit. Therefore,
# it has to be a global flag not per-test.
#
# 2) It resets the FusionCache after each test, which is useful for isolating
# failures. Note, some failures only occur when running multiple tests
# together and accumulating fusions in the cache.
#
# 3) It keeps the temporary files that are created during serde_check.
# Normally, these files are deleted after each test.
env_var_debug_serde = os.getenv("DEBUG_SERDE")
debug_serde: bool = env_var_debug_serde in ("true", "1")


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
                    "***** Use DEBUG_SERDE=true to debug serialization failure."
                )


# Enable automatic serialization upon program exit and test deserializing the
# default workspace. NOTE: Serializing error test cases corrupts the serialized
# binary. Call FusionCache.reset() to clear the cache after running an error
# test in `test_python_frontend.py'.
def atexit_serde_check():
    from nvfuser import FusionCache

    if not debug_serde:
        from nvfuser import enable_automatic_serialization

        # Turn on default serialization upon program exit
        enable_automatic_serialization()

    # Automatically load common workplace
    fc = FusionCache.get()
    # Clear FusionCache because the tests expect a new fusion to be generated.
    FusionCache.reset()

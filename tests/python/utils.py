# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import os
from copy import deepcopy
from typing import Callable
import tempfile

import torch
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.jit_utils import RUN_CUDA

# flake8: noqa
from nvfuser import FusionCache, FusionDefinition, DataType

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM


def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def is_pre_ampere():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 8


def is_pre_hopper():
    if not RUN_NVFUSER:
        return False
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
        return all(
            [
                torch.allclose(ref_out, captured_outputs[idx], equal_nan=True)
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


def serde_check(test_fn: Callable):
    """
    A decorator to verify that serialization works with the given exec_nvfuser function.
    Currently, it uses serialization to rebuild the FusionCache structure.
    """

    def inner_fn(*args, **kwargs):
        self, fusion_func, inputs = args

        # NOTE: For debug purposes, clear FusionCache before running first test
        # so the behavior is more deterministic (PR #1848).
        is_new_fusion_expected = kwargs.get("new_fusion_expected", True)
        if debug_serde and is_new_fusion_expected:
            FusionCache.reset()
            assert FusionCache.get().num_fusions() == 0

        # skip_serde_check is only used by the decorator so remove it before running test_fn
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

        # If DEBUG_SERDE is enabled, the temporary file is not deleted automatically
        with tempfile.NamedTemporaryFile(delete=(not debug_serde)) as tmp:
            try:
                # Serialize FusionCache
                fc = FusionCache.get()
                fc.serialize(tmp.name)

                FusionCache.reset()

                # Get new FusionCache because the previous one was destroyed by the reset call.
                fc = FusionCache.get()
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

        # Run test with repopulated FusionCache
        kwargs["new_fusion_expected"] = False
        return test_fn(self, fusion_func, inputs, **kwargs)

    return inner_fn

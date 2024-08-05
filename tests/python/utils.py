# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch

# flake8 complains about DataType being unused in this file but it is necessary
# to run captured fusion definition.
# flake8: noqa
from nvfuser import FusionDefinition, DataType


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

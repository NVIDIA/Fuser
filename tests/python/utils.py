# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import (FusionDefinition, DataType)

def check_captured_python_definition(fd, inputs, device):
    import re

    # Execute the python definition that was captured
    try:
        fd_str = fd.__repr__()
        func_name = re.findall("(nvfuser_fusion_id\\d+)", fd_str.split("\n")[1])[0]
        exec(fd_str)
        with FusionDefinition() as fd_cap:
            eval(func_name)(fd_cap)
        torch.manual_seed(0)
        return fd_cap.execute(inputs, device=device)
    except Exception as err:
        print("\nException For Printed FusionDefinition:")
        print(
            "(A failure here suggests a mismatch in functionality between the original definition and the printed definition.)"
        )
        print(fd_str)
        raise err

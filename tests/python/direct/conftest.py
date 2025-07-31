# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
from copy import deepcopy
import torch
from torch.testing._internal.common_utils import TestCase

from nvfuser_direct import FusionDefinition
from python.direct_utils import is_pre_volta, check_captured_python_definition


class NVFuserTest(TestCase):
    # Helper function to verify the nvfuser output and make sure the string
    # definition based on the FusionDefinition is executable and matches the
    # original definition
    def exec_nvfuser(
        self,
        fusion_func,
        inputs,
        *,
        expected_fd_str=None,
        device=None,
    ):
        # Copy inputs because aliased outputs can modify inputs when running
        # FusionDefinition
        inputs_captured = deepcopy(inputs)

        # Execute a fusion function and capture the string python definition
        with FusionDefinition() as fd:
            fusion_func(fd)
        torch.manual_seed(0)
        out = fd.execute(
            inputs,
            device=device,
        )

        assert check_captured_python_definition(out, fd, inputs_captured, device)
        assert expected_fd_str is None or expected_fd_str in repr(fd)
        return out, fd


# Migrated tests to new direct python bindings use this.
@pytest.fixture
def nvfuser_direct_test():
    if is_pre_volta():
        pytest.skip("Only supported on Volta and newer devices.")
    yield NVFuserTest()

# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
from copy import deepcopy
import torch
from torch.testing._internal.common_utils import TestCase

from nvfuser_direct import FusionDefinition, LRUCache
from python.direct_utils import is_pre_volta, check_captured_python_definition


class NVFuserTest(TestCase):
    def __init__(self, cache=None):
        super().__init__()
        self.cache = cache

    # Helper function to verify the nvfuser output and make sure the string
    # definition based on the FusionDefinition is executable and matches the
    # original definition
    def exec_nvfuser(
        self,
        fusion_func,
        inputs,
        *,
        new_fusion_expected=True,
        expected_fd_str=None,
        device=None,
        validate_results=False,
    ):
        # Copy inputs because aliased outputs can modify inputs when running
        # FusionDefinition
        inputs_captured = deepcopy(inputs)

        if self.cache is not None:
            prev_size = self.cache.num_fusions()

            # Run compilation twice to test lru cache; The number of fusions
            # should not increase during the second round.
            for _ in range(2):
                with FusionDefinition() as fd:
                    fusion_func(fd)

                fd.fec = self.cache.cache_compile(fd.fusion)
                del fd._fusion

            # The LRU cache is shared across all tests. If you run all the tests
            # together, a fusion can be cached from an earlier test. If you run
            # a test standalone, then the fusion will not exist. Skip this
            # assertion for this case but check that cache_compile works
            # correctly.
            if new_fusion_expected is not None:
                assert self.cache.num_fusions() == prev_size + int(new_fusion_expected)
        else:
            with FusionDefinition() as fd:
                fusion_func(fd)

        # Execute a fusion function and capture the string python definition
        # Set manual_seed before running fusion to avoid mismatch results
        # with rng kernels
        torch.manual_seed(0)
        if validate_results:
            out = fd.validate(inputs)
        else:
            out = fd.execute(
                inputs,
                device=device,
            )

        assert check_captured_python_definition(out, fd, inputs_captured, device)
        assert expected_fd_str is None or expected_fd_str in repr(fd)
        return out, fd


# Migrated tests to new direct python bindings use this.
@pytest.fixture(params=["lru_cache", "eager"])
def nvfuser_direct_test(request):
    if is_pre_volta():
        pytest.skip("Only supported on Volta and newer devices.")

    cache_type = request.param
    if cache_type == "lru_cache":
        if not hasattr(nvfuser_direct_test, "cache"):
            nvfuser_direct_test.cache = LRUCache(max_fusions=16384)
        yield NVFuserTest(nvfuser_direct_test.cache)
    else:
        yield NVFuserTest()

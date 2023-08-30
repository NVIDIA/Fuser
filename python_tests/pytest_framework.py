# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import inspect
import torch
from typing import Callable
from pytest_utils import map_dtype_to_str
import pytest


def _instantiate_opinfo_test_template(
    template: Callable, *, opinfo, dtype: torch.dtype
) -> Callable:
    """Instantiates a test template for an operator."""

    def test():
        # Ref: https://github.com/pytorch/pytorch/blob/aa8ea1d787a9d21b064b664c5344376265feea6c/torch/testing/_internal/common_utils.py#L2251-L2263
        # > CUDA device side error will cause subsequence test cases to fail.
        # > stop entire test suite if catches RuntimeError during torch.cuda.synchronize().
        if torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize()
            except RuntimeError as rte:
                pytest.exit("TEST SUITE EARLY TERMINATION due to torch.cuda.synchronize() failure")

        return template(opinfo, dtype)

    test.__name__ = "_".join((template.__name__, opinfo.name, map_dtype_to_str[dtype]))
    test.__module__ = test.__module__
    return test


class create_op_test:
    def __init__(self, opinfos, *, scope=None):
        self.opinfos = opinfos

        # Acquires the caller's global scope
        if scope is None:
            previous_frame = inspect.currentframe().f_back
            scope = previous_frame.f_globals
        self.scope = scope

    def __call__(self, test_template):
        # NOTE Unlike a typical decorator, this __call__ does not return a function, because it may
        #   (and typically does) instantiate multiple functions from the template it consumes.
        #   Since Python doesn't natively support one-to-many function decorators, the produced
        #   functions are directly assigned to the requested scope (the caller's global scope by default)
        for opinfo in self.opinfos:
            for dtype in sorted(opinfo.dtypes, key=lambda t: repr(t)):
                test = _instantiate_opinfo_test_template(
                    test_template,
                    opinfo=opinfo,
                    dtype=dtype,
                )
                # Adds the instantiated test to the requested scope
                self.scope[test.__name__] = test

# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import inspect
import torch
from typing import Callable
from pytest_utils import map_dtype_to_str


def _instantiate_opinfo_test_template(
    template: Callable, *, opinfo, dtype: torch.dtype
) -> Callable:
    """Instantiates a test template for an operator."""

    def test():
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

# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

assert (
    "nvfuser" not in sys.modules
), "Cannot import nvfuser_next if nvfuser module is already imported."

import os
import torch

# This is needed when libnvfuser_next.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

from . import _C_NEXT  # noqa: F401,F403
from ._C_NEXT import *  # noqa: F401,F403


class FusionDefinition(_C_NEXT._FusionDefinition):
    """
    A class for defining and executing fused operations in nvFuser.
    This class provides a context manager interface for defining fused operations,
    managing inputs/outputs, and executing the fusion with PyTorch tensors.
    Examples
    --------
    >>> with FusionDefinition() as fd:
    ...     t0 = fd.define_tensor()
    ...     t1 = fd.ops.relu(t0)
    ...     fd.add_output(t1)
    >>> outputs = fd.execute([input_tensor])
    """

    def __init__(self):
        """
        Initialize a new FusionDefinition instance.
        """
        super(FusionDefinition, self).__init__()
        self.profiled = False
        self.fusion = _C_NEXT.Fusion()
        self.fusion_guard = _C_NEXT.FusionGuard(self.fusion)

    def define_tensor(self, *args, **kwargs):
        """
        Define a new tensor input for the fusion.
        Parameters
        ----------
        *args
            Positional arguments passed to _C_NEXT.define_tensor
        **kwargs
            Keyword arguments passed to _C_NEXT.define_tensor
        Returns
        -------
        Tensor
            The defined tensor
        """
        tv = _C_NEXT.define_tensor(*args, **kwargs)
        self.fusion.add_input(tv)
        return tv

    def add_output(self, *args, **kwargs):
        """
        Add an output to the fusion.
        Parameters
        ----------
        *args
            Positional arguments passed to fusion.add_output
        **kwargs
            Keyword arguments passed to fusion.add_output
        """
        self.fusion.add_output(*args, **kwargs)

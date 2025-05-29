# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

assert (
    "nvfuser" not in sys.modules
), "Cannot import nvfuser_direct if nvfuser module is already imported."

import os
import torch
import traceback

# This is needed when libnvfuser_direct.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

from . import _C_DIRECT  # noqa: F401,F403
from ._C_DIRECT import *  # noqa: F401,F403


class FusionDefinition:
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
        # Monkey patching nvfuser_direct.ops submodule to mimic python_frontend
        # FusionDefinition.ops API. This is to maintain backwards compatibilty.
        self.ops = _C_DIRECT.ops
        self._fusion = None
        self._fusion_guard = None

    @property
    def fusion(self):
        if not hasattr(self, "fec"):
            return self._fusion
        else:
            return self.fec.fusion()

    def __enter__(self):
        """
        Enter the context manager.

        Returns
        -------
        FusionDefinition
            The FusionDefinition instance
        """
        self._fusion = _C_DIRECT.Fusion()
        self._fusion_guard = _C_DIRECT.FusionGuard(self._fusion)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Exit the context manager and handle any exceptions.
        This method is called when exiting the 'with' block, whether normally or due to an exception.
        The arguments provide information about any exception that occurred:

        Parameters
        ----------
        excecption_type : type or None
            The type of the exception (e.g., ValueError, TypeError).
            None if no exception occurred.
        exception_value : Exception or None
            The actual exception object.
            None if no exception occurred.
        exception_traceback : traceback or None
            The traceback object containing the call stack.
            None if no exception occurred.
        """
        del self._fusion_guard
        if exception_type is not None:
            print(f"Exception occurred: {exception_type.__name__}: {exception_value}")
            if exception_traceback is not None:
                # Format the traceback and print it
                print("Traceback (most recent call last):")
                traceback.print_tb(exception_traceback)

    def define_tensor(self, *args, **kwargs):
        """
        Define a new tensor input for the fusion.

        Parameters
        ----------
        *args
            Positional arguments passed to _C_DIRECT.define_tensor
        **kwargs
            Keyword arguments passed to _C_DIRECT.define_tensor

        Returns
        -------
        Tensor
            The defined tensor
        """
        tv = _C_DIRECT.define_tensor(*args, **kwargs)
        self._fusion.add_input(tv)
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
        self._fusion.add_output(*args, **kwargs)

    def execute(self, inputs, *, device=None, auto_schedule=True) -> list[torch.Tensor]:
        """
        Execute the fusion with the given inputs.

        Parameters
        ----------
        inputs : list of torch.Tensor
            Input tensors and scalars to the fusion
        device : torch.device, optional
            Device to execute the fusion on
        auto_schedule : bool, default=True
            Whether to use automatic scheduling

        Returns
        -------
        list of torch.Tensor
            Output tensors from the fusion
        """
        if auto_schedule:
            if not hasattr(self, "fec"):
                self.fec = _C_DIRECT.FusionExecutorCache(self._fusion)
                # A copy of fusion is created after construction FusionExecutorCache
                # Delete the _fusion and reference the fusion inside FusionExecutorCache
                del self._fusion
            return self.fec.execute(inputs)
        else:
            raise RuntimeError("Manual scheduling is not supported yet.")

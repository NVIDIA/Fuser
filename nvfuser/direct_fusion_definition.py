# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import direct
import traceback


class DirectFusionDefinition(direct._DirectFusionDefinition):
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
        super(DirectFusionDefinition, self).__init__()
        self.profiled = False
        self.fusion = direct.Fusion()
        self.fusion_guard = direct.FusionGuard(self.fusion)

    def __repr__(self):
        """
        Return a string representation of the DirectFusionDefinition.

        Returns
        -------
        str
            A string representation of the DirectFusionDefinition
        """
        return direct.translate_fusion(self.fusion)

    def __enter__(self):
        """
        Enter the context manager.

        Returns
        -------
        FusionDefinition
            The FusionDefinition instance
        """
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
        if exception_type is not None:
            print(f"Exception occurred: {exception_type.__name__}: {exception_value}")
            if exception_traceback is not None:
                # Format the traceback and print it
                print("Traceback (most recent call last):")
                traceback.print_tb(exception_traceback)

    def execute(self, inputs, *, device=None, auto_schedule=True) -> list[torch.Tensor]:
        """
        Execute the fusion with the given inputs.

        Parameters
        ----------
        inputs : list of torch.Tensor
            Input tensors to the fusion
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
                self.fec = direct.FusionExecutorCache(self.fusion)
            return self.fec.execute(inputs)
        else:
            if not hasattr(self, "ke"):
                self.ke = direct.KernelExecutor()
                self.ke.compile(self.fusion, inputs)
            return self.ke.run(inputs)

    def define_tensor(self, *args, **kwargs):
        """
        Define a new tensor input for the fusion.

        Parameters
        ----------
        *args
            Positional arguments passed to direct.define_tensor
        **kwargs
            Keyword arguments passed to direct.define_tensor

        Returns
        -------
        Tensor
            The defined tensor
        """
        tv = direct.define_tensor(*args, **kwargs)
        self.add_input(tv)
        return tv

    def define_scalar(self, *args, **kwargs):
        """
        Define a new scalar input for the fusion.

        Parameters
        ----------
        *args
            Positional arguments passed to direct.define_scalar
        **kwargs
            Keyword arguments passed to direct.define_scalar

        Returns
        -------
        Scalar
            The defined scalar
        """
        scalar = direct.define_scalar(*args, **kwargs)
        if scalar.is_symbolic():
            self.add_input(scalar)
        return scalar

    def add_input(self, *args, **kwargs):
        """
        Add an input to the fusion.

        Parameters
        ----------
        *args
            Positional arguments passed to fusion.add_input
        **kwargs
            Keyword arguments passed to fusion.add_input
        """
        self.fusion.add_input(*args, **kwargs)

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

    def from_pytorch(self, tensor, static_sizes=False):
        """
        Define an nvFuser input tensor from a PyTorch tensor.

        This method creates a symbolic tensor for dynamic shape usage by default.

        Parameters
        ----------
        tensor : torch.Tensor
            Input PyTorch tensor to convert
        static_sizes : bool, default=False
            Whether to interpret sizes as static rather than symbolic
            for dynamic shape usage

        Returns
        -------
        Tensor
            The defined nvFuser tensor

        Raises
        ------
        ValueError
            If a CPU non-scalar tensor is provided
        """
        try:
            from .pytorch_utils import torch_dtype_to_nvfuser_dtype
        except ImportError:
            raise ImportError("Unable to import pytorch_utils!")

        if not tensor.is_cuda and len(tensor.size()) != 0:
            raise ValueError("CPU non-scalar tensor is not supported!")

        tv = direct.define_tensor(
            sizes=tensor.size(),
            strides=tensor.stride(),
            dtype=torch_dtype_to_nvfuser_dtype(tensor.dtype),
            static_sizes=static_sizes,
            is_cpu=tensor.is_cpu,
        )
        self.fusion.add_input(tv)
        return tv

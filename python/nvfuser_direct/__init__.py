# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import warnings

if "nvfuser" in sys.modules:
    warnings.warn(
        "Be careful! You've imported nvfuser_direct when the nvfuser module is already imported.",
        UserWarning,
    )

import os
import torch
import traceback

# This is needed when libnvfuser_direct.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

from . import _C_DIRECT  # noqa: F401,F403
from ._C_DIRECT import *  # noqa: F401,F403


def execute_with_dtensors(fd, in_dtensors):
    """
    Execute a fusion on a list of DTensor inputs.

    Parameters
    ----------
    fd : FusionDefinition
        The fusion definition to execute
    in_dtensors : list of DTensor
        The list of DTensor inputs to the fusion

    Returns
    -------
    list of DTensor
        The list of DTensor outputs from the fusion
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Placement, Shard, Replicate

    inputs = [in_dtensor.to_local() for in_dtensor in in_dtensors]
    out_tensors = fd.execute(inputs, auto_schedule=True)
    out_shardings = fd.fec.get_output_shardings()
    assert len(out_tensors) == len(out_shardings)

    out_dtensors: list[DTensor] = []
    for out_tensor, out_sharding in zip(out_tensors, out_shardings):
        mesh = dist.device_mesh.init_device_mesh("cuda", (out_sharding.mesh.size,))
        placements: list[Placement] = []
        for parallel_type in [_C_DIRECT.ParallelType.mesh_x]:
            axis: int = out_sharding.axis_sharded_on(parallel_type)
            placements.append(Replicate() if axis == -1 else Shard(axis))
        out_dtensors.append(DTensor.from_local(out_tensor, mesh, placements))
    return out_dtensors


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

    def __repr__(self):
        """
        Return a string representation of the FusionDefinition.
        Returns
        -------
        str
            A string representation of the FusionDefinition
        """
        return _C_DIRECT.translate_fusion(self.fusion)

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

    def define_scalar(self, *args, **kwargs):
        """
        Define a new scalar input for the fusion.
        It is added as a fusion input if it is a symbolic value.
        Parameters
        ----------
        *args
            Positional arguments passed to _C_DIRECT.define_scalar
        **kwargs
            Keyword arguments passed to _C_DIRECT.define_scalar
        Returns
        -------
        Scalar
            The defined scalar
        """
        scalar = _C_DIRECT.define_scalar(*args, **kwargs)
        if scalar.is_symbolic():
            self._fusion.add_input(scalar)
        return scalar

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

        supported_tensor = tensor.is_cuda or (tensor.is_cpu and len(tensor.size()) == 0)
        if not supported_tensor:
            raise ValueError(
                f"Found unsupported device {tensor.device}, only scalar CPU or CUDA tensors are supported"
            )

        tv = _C_DIRECT.define_tensor(
            sizes=tensor.size(),
            strides=tensor.stride(),
            dtype=torch_dtype_to_nvfuser_dtype(tensor.dtype),
            static_sizes=static_sizes,
            is_cpu=tensor.is_cpu,
        )
        self.fusion.add_input(tv)
        return tv

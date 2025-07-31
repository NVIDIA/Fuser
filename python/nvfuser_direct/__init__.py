# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import warnings
import functools

if "nvfuser" in sys.modules:
    warnings.warn(
        "Be careful! You've imported nvfuser_direct when the nvfuser module is already imported.",
        UserWarning,
    )

from typing import Optional
import os
import torch
import traceback

# This is needed when libnvfuser_direct.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

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


class LruFusionCache:
    """
    A class that caches the FusionExecutorCache with the Fusion as the key.
    The cache is a LRU cache that evicts the least recently used FusionExecutorCache.
    The cache is used to avoid recompiling the same FusionExecutorCache.
    """

    def __init__(self, max_fusions=16384):
        """
        Initialize a new LruFusionCache instance.
        """
        self.cache = _C_DIRECT.LRUCache(max_fusions)

    def __call__(self, create_fusion_definition):
        """
        A decorator that caches the FusionExecutorCache with the Fusion as the key.
        It returns the compiled fusion definition.
        The cache is a LRU cache that evicts the least recently used FusionExecutorCache.
        The cache is used to avoid recompiling the same FusionExecutorCache.
        """

        @functools.wraps(create_fusion_definition)
        def wrapper(*args, **kwargs):
            fusion_definition = create_fusion_definition(*args, **kwargs)
            if not hasattr(fusion_definition, "fec"):
                fusion_definition.fec = self.cache.cache_compile(
                    fusion_definition.fusion
                )
                # A copy of fusion is created after construction FusionExecutorCache
                # Delete the _fusion and reference the fusion inside FusionExecutorCache
                del fusion_definition._fusion
            return fusion_definition

        def stats():
            """
            Get the stats of the cache.
            """
            return self.cache.stats()

        def num_fusions():
            """
            Get the number of fusions in the cache.
            """
            return self.cache.num_fusions()

        wrapper.stats = stats
        wrapper.num_fusions = num_fusions
        return wrapper


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

    def define_vector(self, size):
        """
        Define a new vector input for the fusion.

        Parameters
        ----------
        size : int
            The size of the vector

        Returns
        -------
        list of Scalar
            The defined vector
        """
        return [self.define_scalar(None, DataType.Int) for i in range(size)]

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

    def _get_device_index(self, device_arg: Optional[torch.device | int | str]) -> int:
        """
        Get the integer index for device_arg.

        Parameters
        ----------
        device_arg : Optional[torch.device | int | str]

        Returns
        -------
        int
            The index for cuda device
        """
        if device_arg is None or isinstance(device_arg, int):
            return device_arg

        # NOTE: torch.device(torch.device) is still a torch.device
        device = torch.device(device_arg)
        assert (
            device.type == "cuda"
        ), "If device argument is passed it must be a CUDA device"
        return device.index

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
            return self.fec.execute(inputs, device=self._get_device_index(device))
        else:
            raise RuntimeError("Manual scheduling is not supported yet.")

    def repro_script_for(self, inputs: list | None = None) -> str:
        """
        Generate a repro script for the fusion.

        Parameters
        ----------
        inputs : list of torch.Tensor
            The list of torch.Tensor inputs to the fusion

        Returns
        -------
        str
            The repro script for the fusion
        """

        msg = "# CUDA devices:\n"
        for i in range(torch.cuda.device_count()):
            msg += f"#  {i}: {torch.cuda.get_device_name(i)}\n"
        msg += (
            f"# torch version: {torch.__version__}\n"
            f"# cuda version: {torch.version.cuda}\n"
            f"import torch\n"
            "from nvfuser_direct import FusionDefinition, DataType\n"
            f"{self}"
            "with FusionDefinition() as fd:\n"
            f"    nvfuser_fusion(fd)\n"
        )
        if inputs is not None:
            msg += "\ninputs = [\n"
            for i in inputs:
                if isinstance(i, torch.Tensor):
                    if i.is_contiguous():
                        msg += f"    torch.testing.make_tensor({tuple(i.size())}, dtype={i.dtype}, device='{i.device}'),\n"
                    else:
                        # max linear index determines number of elements to generate
                        sz = 1
                        for szi, stri in zip(i.size(), i.stride()):
                            if szi == 0:
                                sz = 0
                                break
                            sz += (szi - 1) * stri
                        if i.dtype.is_floating_point:
                            msg += (
                                f"    torch.randn({sz}, dtype={i.dtype}, device='{i.device}')"
                                f".as_strided({tuple(i.size())}, {tuple(i.stride())}),\n"
                            )
                        else:
                            upper_bound = 2 if i.dtype == torch.bool else 10
                            msg += (
                                f"    torch.randint(0, {upper_bound}, ({sz},), dtype={i.dtype}, device='{i.device}')"
                                f".as_strided({tuple(i.size())}, {tuple(i.stride())}),\n"
                            )
                else:
                    input_as_string = str(i)
                    # `nan` and `inf` are stringified as is, which are not
                    # defined in Python. So we replace them with `float("nan")`
                    # and `float("inf")`. `-inf` is replaced with
                    # `-float("inf")`, which equals `float("-inf")`.
                    input_as_string = re.sub(
                        r"\binf\b", 'float("inf")', input_as_string
                    )
                    input_as_string = re.sub(
                        r"\bnan\b", 'float("nan")', input_as_string
                    )
                    msg += f"    {input_as_string},\n"
            msg += "]"
            msg += "\nfd.execute(inputs)\n"

        return msg

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

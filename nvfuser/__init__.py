# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import re
import sys
from typing import Optional, Union, List  # noqa: F401

import torch

# This is needed when libnvfuser.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

# we need to import _C here to avoid confusing error message generated from failure in this python script ended up with
# complaining on `_C` not defined for `_C._FusionDefinition`
from . import _C
from ._C import *  # noqa: F401,F403

from . import contrib  # noqa: F401


logger = logging.getLogger("nvfuser")


# Register automatic serialization of Nvfuser cache hierarchy and cuda kernels.
def enable_automatic_serialization():
    import atexit

    atexit.register(_C.serialize)

    # A separate process is created for each device in a distributed setting.
    # Each FusionCache becomes associated with a single device.
    # Automatic serialization saves a separate cache for each device.
    # Set the FusionCache id to the ddp local rank.
    env_var_ddp_local_rank = os.environ.get("LOCAL_RANK", None)
    if env_var_ddp_local_rank is not None:
        env_var_ddp_local_rank = int(env_var_ddp_local_rank)
    _C.FusionCache.get(max_fusions := 8192, env_var_ddp_local_rank)


# Unregister automatic serialization of Nvfuser cache hierarchy and cuda kernels.
def disable_automatic_serialization():
    import atexit

    atexit.unregister(_C.serialize)


class FusionDefinition(_C._FusionDefinition):
    def __init__(self, id=None, max_length=1024):
        super(FusionDefinition, self).__init__(id, max_length)
        self.profiled = False

    def __enter__(self):
        return self._setup_definition()

    def __exit__(self, type, value, traceback):
        try:
            self._finalize_definition()
        except Exception as err:
            logger.exception(self.getReproErrorString("defining"))
            raise

    def getReproErrorString(self, section: str, inputs: list | None = None):
        msg = (
            f"An error occurred while {section} nvFuser FusionDefinition {self.id()}.\n"
            "If you believe this is a bug or need assistance, please file an issue at "
            "https://github.com/NVIDIA/Fuser/issues/new\n"
            f"Here's a script to reproduce the error:\n"
            "```python\n"
            "# CUDA devices:\n"
        )
        for i in range(torch.cuda.device_count()):
            msg += f"#  {0}: {torch.cuda.get_device_name(i)}\n"
        msg += (
            f"# torch version: {torch.__version__}\n"
            f"# cuda version: {torch.version.cuda}\n"
            f"# nvfuser version: {version()}\n"
            "import torch\n"
            "from nvfuser import FusionDefinition, DataType\n"
            f"{self}"
            "with FusionDefinition() as fd:\n"
            f"    nvfuser_fusion_id{self.id()}(fd)\n"
        )
        if inputs is not None:
            msg += "\ninputs = [\n"
            for i in inputs:
                if isinstance(i, torch.Tensor):
                    # max linear index determines number of elements to generate
                    sz = 1
                    for szi, stri in zip(i.size(), i.stride()):
                        if szi == 0:
                            sz = 0
                            break
                        sz += (szi - 1) * stri
                    if i.dtype.is_floating_point:
                        msg += (
                            f"    torch.randn(({sz},), dtype={i.dtype}, device='{i.device}')"
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
        msg += "```\n"
        return msg

    def definition(self):
        raise NotImplementedError("definition() should be implemented by child class!")

    def schedule(self):
        raise NotImplementedError("schedule() should be implemented by child class!")

    def execute(
        self,
        inputs,
        *,
        device=None,
        override_user_schedule=False,
        capture_debug_output=False,
        profile=False,
    ):
        """
        Executes an nvFuser set of kernels for a given Fusion

        The FusionDefinition will be executed on a single CUDA device.
        Typically, which device to run on is determined by the devices where
        the input tensors reside. However, if the Fusion is defined such that
        none of the inputs are tensors, we are not able to infer a device from
        the inputs. For example, the following FusionDefinition will be unable
        to unambiguously infer the device of its output:

            with FusionDefinition() as fd:
                tv1 = fd.ops.full([5])
                fd.add_output(tv1)

        In that case, we default to selecting the first CUDA
        device, i.e. `torch.device("cuda:0")`. This method enables selecting an
        alternative preferred device.

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.

        Kwargs:
            override_user_schedule (bool): For a user defined schedule,
                override with auto-generated schedule (default: False)
            device (Optional[Union[int, str, torch.device]]): This is a hint to run
                the Fusion on the given CUDA device. This is not typically
                necessary, as the device is usually inferred from the locations
                of input tensors. However, for some fusion definitions, no
                tensors will be input (for example when all tensors are
                generated with `full` or `uniform` ops). In these cases, we
                must either tell NVFuser where to run the resulting kernel, or
                let it default to 0. Note that passing this option providing
                and input tensors that lie on another device is an error.
            capture_debug_output (bool): Whether to capture any printed
                debugging information as a string. If True, the string can be
                retrieved after execution using :meth:`get_debug_output`. If False,
                then that method will return None when called.

        Returns:
            List[Tensor]
        """
        func_based_def = False
        self.profiled = profile

        if device is not None:
            if not isinstance(device, torch.device):
                device = torch.device(device)
            assert (
                device.type == "cuda"
            ), "If device argument is passed it must be a CUDA device"
            device = device.index

        # if definition is not defined by a context manager, try a child class
        if self.id() is None:
            self._setup_definition()
            self.definition()
            self._finalize_definition()
            func_based_def = True

        # If schedule is defined by child class, make a schedule for inputs
        if func_based_def and (super(type(self), self).schedule != self.schedule):
            self._setup_schedule(inputs)
            self.schedule()
            self._finalize_schedule(inputs)

        result = None
        try:
            result = self._execute(
                inputs,
                device=device,
                override_user_schedule=override_user_schedule,
                capture_debug_output=capture_debug_output,
                profile=profile,
            )
        except Exception as err:
            logger.exception(self.getReproErrorString("executing", inputs))
            raise

        return result

    def debug_output(self):
        """
        Retrieve string of captured debug information from the previous execution.

        Note that `capture_debug_output=True` must be passed to `execute()` in
        order to enable capturing this output. Otherwise, this method will
        return `None`.

        Returns:
            Optional[String] : the captured debug output for the previous call
            to execute(). If the `capture_debug_output` argument to that call
            was False, returns None. Otherwise, returns the output as a string.
        """
        return self._debug_output()

    def from_pytorch(self, tensor, static_sizes=False):
        """
        Defines an nvfuser input tensor from a pytorch tensor and defaults
        to definining a symbolic tensor for dynamic shape usage.

        Args:
            tensor (torch.Tensor): Input tensor to nvFuser
            static_sizes (bool)  : Interprets sizes as static rather than
                                   as symbolic for dynamic shape usage

        Returns:
            nvfuser.Tensor
        """
        try:
            from .pytorch_utils import torch_dtype_to_nvfuser_dtype
        except ImportError:
            raise ImportError("Unable to import pytorch_utils!")

        if not tensor.is_cuda:
            raise ValueError("Tensor should be on a cuda device!")

        return self.define_tensor(
            sizes=tensor.size(),
            strides=tensor.stride(),
            dtype=torch_dtype_to_nvfuser_dtype(tensor.dtype),
            static_sizes=static_sizes,
        )

    def fusion_ir(self):
        """
        Returns the uscheduled Fusion IR for the given definition that corresponds to all scheduled inputs.

        Returns:
            String
        """
        return self._fusion_ir()

    def last_cuda_code(self, intrinsic_code=False, **kwargs):
        """
        Returns the Cuda Code for the last executed set of inputs

        Args:
            intrinsic_code (Bool): Include all the additional code required to run kernel(s). (default: False)

        Kwargs:
            override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            String
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        return self._last_cuda_code(intrinsic_code, override_user_schedule)

    def cuda_code_for(self, inputs, intrinsic_code=False, **kwargs):
        """
        Returns the Cuda Code for the given inputs

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.
            intrinsic_code (Bool): Include all the additional code required to run kernel(s). (default: False)

        Kwargs:
            override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            String
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        return self._cuda_code_for(inputs, intrinsic_code, override_user_schedule)

    def last_scheduled_fusion_ir(self, tensor_transforms=False, **kwargs):
        """
        Returns the Scheduled Fusion IR for the last executed set of inputs

        Args:
            tensor_transforms (Bool): Include tensor transforms that were applied through scheduling. (default: False)

        Kwargs:
            override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            String
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        return self._last_scheduled_fusion_ir(tensor_transforms, override_user_schedule)

    def scheduled_fusion_ir_for(self, inputs, tensor_transforms=False, **kwargs):
        """
        Returns the Scheduled Fusion IR for the last executed set of inputs

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.
            tensor_transforms (Bool): Include tensor transforms that were applied through scheduling. (default: False)

        Kwargs:
            override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            String
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        return self._scheduled_fusion_ir_for(
            inputs, tensor_transforms, override_user_schedule
        )

    def profile(self):
        """
        Returns the FusionProfile object from the CUPTI based FusionProfiler

        Returns:
            FusionProfile
        """
        if not self.profiled:
            raise ValueError(
                "The execute() method was not previously called with profiling enabled!"
            )

        fp = self._profile()

        if fp.fusion_id < 0:
            raise ValueError(
                "Something went wrong with Fusion Profiling as an illegal fusion_id was returned! "
                + str(fp.fusion_id)
            )
        if fp.segments < 1:
            raise ValueError(
                "Something went wrong with Fusion Profiling as no kernel segments were profiled!"
                + str(fp.segments)
            )

        return fp

    def validate(
        self,
        inputs: List[torch.Tensor],
        reference_outputs: List[torch.Tensor],
        kwargs=None,
    ):
        """
        Validates the fusion outputs against the provided reference outputs, using variable tolerances determined based on datatype and reduction size.

        Inputs:
            inputs: A list of inputs expected by the fusion definition
            reference_outputs: A list of reference outputs to validate against
        """
        fusion_outputs = self.execute(inputs)
        assert len(fusion_outputs) == len(
            reference_outputs
        ), f"Expected {len(fusion_outputs)} reference outputs for validation."

        tolerance_values = self.getValTolerances(inputs)
        assert len(tolerance_values) == len(
            fusion_outputs
        ), f"Missing tolerance values, expected {len(fusion_outputs)}, got {len(tolerance_values)}"

        for inx, fusion_output in enumerate(fusion_outputs):
            atol, rtol = tolerance_values[inx]
            reference_output = reference_outputs[inx]

            assert (
                reference_output.shape == fusion_output.shape
            ), "Mismatch in reference and fusion output dimensions"
            if torch.is_floating_point(fusion_output) or torch.is_complex(
                fusion_output
            ):
                assert torch.allclose(
                    fusion_output, reference_output, atol=atol, rtol=rtol
                ), f"Max error: {torch.abs(torch.max(fusion_output - reference_output))}, \
                    Absolute tolerance: {atol}, Relative tolerance: {rtol}"

            else:
                assert torch.equal(
                    fusion_output, reference_output
                ), "Mismatch in reference and fusion output values, datatype is not float/complex."


from .nvfuser_version import __version__


def version():
    r"""returns nvfuser version in format of a string 'm.n.p+git[7d-sha]'.

    We strip the git[7d-sha] and convert the string to
    `nvfuser_version.Version` for comparison. e.g. you can use it as:
        import nvfuser
        print(nvfuser.version())              # 0.0.1+git21df524
        nvfuser.version() == '0.0.1`          # True
        nvfuser.version() > '0.0.0`           # True

        from nvfuser_version import Version
        nvfuser.version() < Version('1.0.0')  # True
    """
    return __version__
